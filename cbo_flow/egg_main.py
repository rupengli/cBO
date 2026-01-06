#!/usr/bin/env python3
from __future__ import annotations

"""
One-shot Egg CBO evaluation driven by a YAML config.

Config provides:
  - paths (dataset_root, work_dir, output_dir)
  - 1-stage controls (12 vars: 8 injector rates + 4 producer BHP targets)
  - economics (USD/SM3)
  - constraints (eps + optional limits)

The script:
  1) Creates a fresh work directory with patched deck/schedule
  2) Runs `flow`
  3) Parses the .PRT report and computes objective + constraints
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


CONSTRAINT_SPECS = [
    {
        "name": "C1_PeakFieldWaterInjectionRate",
        "result_key": "g_winj",
        "description_lines": [
            "Peak field injected water rate minus limit (SM3/DAY).",
            "Definition: max_t(FWIR(t)) - Qwinj_max <= 0",
        ],
    },
    {
        "name": "C2_PeakFieldWaterProductionRate",
        "result_key": "g_wprod",
        "description_lines": [
            "Peak field produced water rate minus limit (SM3/DAY).",
            "Definition: max_t(FWPR(t)) - Qwprod_max <= 0",
        ],
    },
    {
        "name": "C3_PeakFieldLiquidRate",
        "result_key": "g_liq",
        "description_lines": [
            "Peak field produced liquid rate minus limit (SM3/DAY).",
            "Definition: max_t(FOPR(t)+FWPR(t)) - Qliq_max <= 0",
        ],
    },
    {
        "name": "C4_PeakFieldWaterCut",
        "result_key": "g_wc",
        "description_lines": [
            "Peak field water cut minus limit (-).",
            "Definition: max_t(FWPR(t)/(FOPR(t)+FWPR(t)+eps)) - WCmax <= 0",
        ],
    },
    {
        "name": "C5_EarlyWaterCutUpToYears",
        "result_key": "g_wc_early",
        "description_lines": [
            "Early breakthrough constraint on field WC (-).",
            "Definition: max_{t<=T}(WC_field(t)) - wc_eps <= 0, with T=c1.years",
        ],
    },
    {
        "name": "C6_BreakthroughTimeSpreadAcrossProducers",
        "result_key": "g_bt_spread",
        "description_lines": [
            "Producer breakthrough-time spread constraint (years).",
            "Definition: (max_j t_bt,j - min_j t_bt,j) - delta_years_max <= 0, where WC_j crosses wc_eps",
        ],
    },
]


def require_len(name: str, vals: list[float], n: int) -> None:
    if len(vals) != n:
        raise ValueError(f"Expected {n} values for {name}, got {len(vals)}")


def patch_schedule_new_inc(text: str, inj_rates: list[float]) -> str:
    require_len("inj_rates", inj_rates, 8)

    # Replace the numeric value immediately after 'RATE' for INJECT1..INJECT8.
    rx = re.compile(
        r"^(?P<prefix>\s*'INJECT(?P<idx>[1-8])'\s+.*?\s'RATE'\s+)"
        r"(?P<val>[-+0-9.eEdD]+)"
        r"(?P<suffix>\s+.*)$",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    def repl(m: re.Match[str]) -> str:
        idx = int(m.group("idx")) - 1
        return f"{m.group('prefix')}{inj_rates[idx]:g}{m.group('suffix')}"

    out, n = rx.subn(repl, text)
    if n == 0:
        raise ValueError("Failed to patch injector rates in SCHEDULE_NEW.INC (no matches)")
    return out


def patch_wconprod_bhp(deck_text: str, prod_bhp: list[float]) -> str:
    require_len("prod_bhp", prod_bhp, 4)

    # Replace the final numeric BHP target on each WCONPROD line for PROD1..PROD4.
    rx = re.compile(
        r"^(?P<prefix>\s*'PROD(?P<idx>[1-4])'\s+'OPEN'\s+'BHP'\s+5\*\s+)"
        r"(?P<val>[-+0-9.eEdD]+)"
        r"(?P<suffix>\s*/\s*)$",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    touched = set()

    def repl(m: re.Match[str]) -> str:
        idx = int(m.group("idx")) - 1
        touched.add(idx)
        return f"{m.group('prefix')}{prod_bhp[idx]:g}{m.group('suffix')}"

    out = rx.sub(repl, deck_text)
    if len(touched) != 4:
        raise ValueError(f"Failed to patch all producer BHP lines (touched {sorted(touched)})")
    return out


def patch_perm_include(deck_text: str, include_path: str) -> str:
    # Deck variants:
    # - this Egg deck includes `mDARCY.INC` (base permeability)
    # - we want to swap it to a specific realization file in Permeability_Realizations/
    #
    # Try to replace an existing Permeability_Realizations include first (if present),
    # otherwise replace the `mDARCY.INC` include block.
    rx_real = re.compile(
        r"^(?P<lead>\s*INCLUDE\s*)\n(?P<path>\s*'.*?Permeability_Realizations/.*?'\s*/\s*)$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    out, n = rx_real.subn(lambda m: f"{m.group('lead')}\n   '{include_path}' /", deck_text, count=1)
    if n == 1:
        return out

    rx_mdarcy = re.compile(
        r"^(?P<lead>\s*INCLUDE\s*)\n(?P<path>\s*mDARCY\.INC\s*)\n(?P<end>\s*/\s*)$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    out, n = rx_mdarcy.subn(lambda m: f"{m.group('lead')}\n   '{include_path}'\n{m.group('end')}", deck_text, count=1)
    if n != 1:
        raise ValueError("Failed to patch permeability INCLUDE (expected a mDARCY or realization include block)")
    return out


def format_wconinje_block(inj_rates: list[float]) -> str:
    require_len("inj_rates", inj_rates, 8)
    lines: list[str] = ["WCONINJE\n"]
    for i in range(8):
        well = f"INJECT{i+1}"
        lines.append(f"'{well}'\t'WATER'\t'OPEN'\t'RATE'\t{inj_rates[i]:g} 1* 420/\n")
    lines.append("/\n")
    return "".join(lines)


def format_wconprod_block(prod_bhp: list[float]) -> str:
    require_len("prod_bhp", prod_bhp, 4)
    lines: list[str] = ["WCONPROD\n"]
    for i in range(4):
        well = f"PROD{i+1}"
        lines.append(f"    '{well}' 'OPEN' 'BHP' 5*  {prod_bhp[i]:g}/\n")
    lines.append("/\n")
    return "".join(lines)


def parse_tstep_blocks(schedule_text: str) -> tuple[str, list[str], str]:
    lines = schedule_text.splitlines(keepends=True)

    # Find the first WCONINJE keyword and its terminating '/' line.
    w_start = None
    for i, line in enumerate(lines):
        if line.strip().upper() == "WCONINJE":
            w_start = i
            break
    if w_start is None:
        raise ValueError("SCHEDULE_NEW.INC must contain a WCONINJE block")

    w_end = None
    for j in range(w_start + 1, len(lines)):
        if lines[j].strip() == "/":
            w_end = j
            break
    if w_end is None:
        raise ValueError("Failed to find end of WCONINJE block (missing '/')")

    header = "".join(lines[: w_end + 1])
    rest = lines[w_end + 1 :]

    blocks: list[str] = []
    footer_lines: list[str] = []
    k = 0
    while k < len(rest):
        s = rest[k].strip()
        if s == "" or s.startswith("--"):
            k += 1
            continue
        if s.upper() == "END":
            footer_lines = rest[k:]
            break
        if s.upper() != "TSTEP":
            raise ValueError(f"Unsupported keyword after WCONINJE in SCHEDULE_NEW.INC: {s}")

        start = k
        k += 1
        while k < len(rest):
            if "/" in rest[k]:
                k += 1
                break
            k += 1
        blocks.append("".join(rest[start:k]))

    if not blocks:
        raise ValueError("No TSTEP blocks found after WCONINJE in SCHEDULE_NEW.INC")

    return header, blocks, "".join(footer_lines)


def build_multistage_schedule(
    schedule_text_stage1_patched: str,
    *,
    stage_inj_rates: list[list[float]],
    stage_prod_bhp: list[list[float]],
    stage_tsteps: list[int] | None,
) -> str:
    n_stages = len(stage_inj_rates)
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    if len(stage_prod_bhp) != n_stages:
        raise ValueError("stage_prod_bhp length must match stage_inj_rates length")

    header, tstep_blocks, footer = parse_tstep_blocks(schedule_text_stage1_patched)
    n_blocks = len(tstep_blocks)
    if n_stages > n_blocks:
        raise ValueError(f"n_stages={n_stages} exceeds number of TSTEP blocks ({n_blocks})")

    if stage_tsteps is None:
        base = n_blocks // n_stages
        rem = n_blocks % n_stages
        stage_tsteps = [base + (1 if i < rem else 0) for i in range(n_stages)]

    if len(stage_tsteps) != n_stages:
        raise ValueError(f"stage_tsteps must have length n_stages={n_stages}")
    if any(x <= 0 for x in stage_tsteps):
        raise ValueError("All stage_tsteps values must be positive")
    if sum(stage_tsteps) != n_blocks:
        raise ValueError(f"stage_tsteps must sum to number of TSTEP blocks ({n_blocks}), got {sum(stage_tsteps)}")

    out_lines: list[str] = [header, "\n"]
    idx = 0
    for stage_idx in range(n_stages):
        if stage_idx > 0:
            out_lines.append("\n")
            out_lines.append(format_wconinje_block(stage_inj_rates[stage_idx]))
            out_lines.append("\n")
            out_lines.append(format_wconprod_block(stage_prod_bhp[stage_idx]))
            out_lines.append("\n")

        take = stage_tsteps[stage_idx]
        out_lines.extend(tstep_blocks[idx : idx + take])
        idx += take

    if footer:
        out_lines.append(footer)

    return "".join(out_lines)


def prepare_workdir(
    *,
    dataset_root: Path,
    work_dir: Path,
    stage_inj_rates: list[list[float]],
    stage_prod_bhp: list[list[float]],
    stage_tsteps: list[int] | None,
    permeability_include: str | None,
) -> Path:
    dataset_root = dataset_root.resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    (work_dir / "Eclipse").mkdir(parents=True, exist_ok=True)

    # Copy required Eclipse files.
    src_ecl = dataset_root / "Eclipse"
    if not src_ecl.is_dir():
        raise FileNotFoundError(f"Missing Eclipse folder: {src_ecl}")

    active = (src_ecl / "ACTIVE.INC").read_text()
    (work_dir / "Eclipse" / "ACTIVE.INC").write_text(active)

    # Keep base permeability file available if perm_realization is not used.
    mdarcy = (src_ecl / "mDARCY.INC").read_text()
    (work_dir / "Eclipse" / "mDARCY.INC").write_text(mdarcy)

    sched_text = (src_ecl / "SCHEDULE_NEW.INC").read_text()
    sched_text = patch_schedule_new_inc(sched_text, stage_inj_rates[0])
    if len(stage_inj_rates) > 1:
        sched_text = build_multistage_schedule(
            sched_text,
            stage_inj_rates=stage_inj_rates,
            stage_prod_bhp=stage_prod_bhp,
            stage_tsteps=stage_tsteps,
        )
    (work_dir / "Eclipse" / "SCHEDULE_NEW.INC").write_text(sched_text)

    deck_text = (src_ecl / "Egg_Model_ECL.DATA").read_text()
    deck_text = patch_wconprod_bhp(deck_text, stage_prod_bhp[0])
    if permeability_include is not None:
        deck_text = patch_perm_include(deck_text, permeability_include)
    (work_dir / "Eclipse" / "Egg_Model_ECL.DATA").write_text(deck_text)

    # Symlink permeability folder (large) rather than copying.
    src_perm = (dataset_root / "Permeability_Realizations").resolve()
    if not src_perm.exists():
        raise FileNotFoundError(f"Missing Permeability_Realizations: {src_perm}")
    (work_dir / "Permeability_Realizations").symlink_to(src_perm, target_is_directory=True)

    return work_dir


def detect_prt_path(output_dir: Path) -> Path:
    prts = sorted(output_dir.glob("*.PRT"))
    if len(prts) != 1:
        raise ValueError(f"Expected exactly 1 .PRT in {output_dir}, found {len(prts)}")
    return prts[0]


def split_cells(line: str) -> list[str]:
    parts = line.rstrip("\n").split(":")
    if len(parts) < 3:
        return []
    return [cell.strip() for cell in parts[1:-1]]


def merge_headers(header_lines: list[str]) -> list[str]:
    rows = [split_cells(h) for h in header_lines]
    if not rows or any(not r for r in rows):
        raise ValueError("Failed to parse table header lines")
    ncols = min(len(r) for r in rows)
    merged: list[str] = []
    for col_idx in range(ncols):
        pieces = [rows[row_idx][col_idx] for row_idx in range(len(rows))]
        pieces = [p for p in pieces if p]
        merged.append(" ".join(pieces))
    return merged


def parse_table(lines: list[str]) -> tuple[dict[str, dict[str, str]], int]:
    if len(lines) < 3:
        raise ValueError("Not enough lines to parse table header")
    columns = merge_headers(lines[:3])

    rows: dict[str, dict[str, str]] = {}
    i = 3
    while i < len(lines):
        line = lines[i]
        if line.startswith(" :--------") or line.startswith(":--------"):
            return rows, i + 1
        if not line.startswith(":"):
            i += 1
            continue
        cells = split_cells(line)
        if not cells:
            i += 1
            continue
        row_name = cells[0]
        row: dict[str, str] = {}
        for col_idx in range(min(len(columns), len(cells))):
            row[columns[col_idx]] = cells[col_idx]
        rows[row_name] = row
        i += 1
    return rows, i


def find_col(columns: list[str], must_have: list[str]) -> str:
    for col in columns:
        u = col.upper()
        if all(tok.upper() in u for tok in must_have):
            return col
    raise KeyError(f"Could not find column with tokens {must_have}")


def get_volume_multiplier_to_sm3(col_name: str) -> float:
    u = col_name.upper()
    if "MSCM" in u:
        return 1000.0
    if "SM3" in u or "SCM" in u:
        return 1.0
    raise ValueError(f"Unsupported cumulative unit in column: {col_name}")


def get_rate_multiplier_to_sm3_per_day(col_name: str) -> float:
    u = col_name.upper()
    if "SM3/DAY" in u or "SCM/DAY" in u:
        return 1.0
    raise ValueError(f"Unsupported rate unit in column: {col_name}")


def evaluate_outputs(
    *,
    output_dir: Path,
    eps: float,
    oil_price_usd_per_sm3: float,
    wprod_cost_usd_per_sm3: float,
    winj_cost_usd_per_sm3: float,
    qliq_max: float | None,
    qwprod_max: float | None,
    qwinj_max: float | None,
    wc_max: float | None,
) -> dict:
    prt_path = detect_prt_path(output_dir)
    lines = prt_path.read_text(errors="replace").splitlines(keepends=True)

    # --- Totals at T from last cumulative totals table ---
    idxs = [i for i, line in enumerate(lines) if "CUMULATIVE PRODUCTION/INJECTION TOTALS" in line]
    if not idxs:
        raise ValueError(f"Could not find cumulative totals table in {prt_path}")
    i = idxs[-1] + 1
    while i < len(lines) and " :  WELL  :" not in lines[i]:
        i += 1
    if i + 3 >= len(lines):
        raise ValueError(f"Could not find cumulative totals header in {prt_path}")

    rows, end_idx = parse_table(lines[i:])
    field = rows["FIELD"]
    cols = list(field.keys())
    col_oil = find_col(cols, ["OIL", "PROD"])
    col_wprod = find_col(cols, ["WATER", "PROD"])
    col_winj = find_col(cols, ["WATER", "INJ"])

    fopt_t = float(field[col_oil].strip()) * get_volume_multiplier_to_sm3(col_oil)
    fwpt_t = float(field[col_wprod].strip()) * get_volume_multiplier_to_sm3(col_wprod)
    fwit_t = float(field[col_winj].strip()) * get_volume_multiplier_to_sm3(col_winj)

    npv = (
        oil_price_usd_per_sm3 * fopt_t
        - wprod_cost_usd_per_sm3 * fwpt_t
        - winj_cost_usd_per_sm3 * fwit_t
    )
    obj = -npv

    # --- Rate time series from report tables ---
    fopr: list[float] = []
    fwpr: list[float] = []
    fwir: list[float] = []

    j = 0
    while j < len(lines):
        line = lines[j]

        if "PRODUCTION REPORT" in line:
            j += 1
            while j < len(lines) and " :  WELL  :" not in lines[j]:
                j += 1
            if j + 3 >= len(lines):
                raise ValueError(f"Could not find production report header in {prt_path}")
            prod_rows, consumed = parse_table(lines[j:])
            prod_field = prod_rows["FIELD"]
            prod_cols = list(prod_field.keys())
            col_orate = find_col(prod_cols, ["OIL", "RATE"])
            col_wrate = find_col(prod_cols, ["WATER", "RATE"])
            fopr.append(float(prod_field[col_orate].strip()) * get_rate_multiplier_to_sm3_per_day(col_orate))
            fwpr.append(float(prod_field[col_wrate].strip()) * get_rate_multiplier_to_sm3_per_day(col_wrate))
            j += consumed
            continue

        if "INJECTION REPORT" in line:
            j += 1
            while j < len(lines) and " :  WELL  :" not in lines[j]:
                j += 1
            if j + 3 >= len(lines):
                raise ValueError(f"Could not find injection report header in {prt_path}")
            inj_rows, consumed = parse_table(lines[j:])
            inj_field = inj_rows["FIELD"]
            inj_cols = list(inj_field.keys())
            col_winj_rate = find_col(inj_cols, ["WATER", "RATE"])
            fwir.append(float(inj_field[col_winj_rate].strip()) * get_rate_multiplier_to_sm3_per_day(col_winj_rate))
            j += consumed
            continue

        j += 1

    if len(fopr) != len(fwpr):
        raise ValueError(f"FOPR/FWPR length mismatch: {len(fopr)} vs {len(fwpr)}")
    if not fwir:
        raise ValueError("No injection report rates found (FWIR)")

    qliq = [qo + qw for qo, qw in zip(fopr, fwpr)]
    wc = [qw / (qo + qw + eps) for qo, qw in zip(fopr, fwpr)]

    fopr_peak = max(fopr) if fopr else 0.0
    fwpr_peak = max(fwpr) if fwpr else 0.0
    fwir_peak = max(fwir) if fwir else 0.0
    qliq_peak = max(qliq) if qliq else 0.0
    wc_peak = max(wc) if wc else 0.0

    g_liq = (qliq_peak - qliq_max) if qliq_max is not None else None
    g_wprod = (fwpr_peak - qwprod_max) if qwprod_max is not None else None
    g_winj = (fwir_peak - qwinj_max) if qwinj_max is not None else None
    g_wc = (wc_peak - wc_max) if wc_max is not None else None

    return {
        "FOPT_T_sm3": fopt_t,
        "FWPT_T_sm3": fwpt_t,
        "FWIT_T_sm3": fwit_t,
        "NPV_USD": npv,
        "OBJ": obj,
        "FOPR_peak": fopr_peak,
        "FWPR_peak": fwpr_peak,
        "FWIR_peak": fwir_peak,
        "Qliq_peak": qliq_peak,
        "WC_peak": wc_peak,
        "g_liq": g_liq,
        "g_wprod": g_wprod,
        "g_winj": g_winj,
        "g_wc": g_wc,
    }


def detect_smspec_path(output_dir: Path) -> Path:
    smspec = sorted(output_dir.glob("*.SMSPEC"))
    if len(smspec) != 1:
        raise ValueError(f"Expected exactly 1 .SMSPEC in {output_dir}, found {len(smspec)}")
    return smspec[0]


def compute_breakthrough_constraints(
    *,
    output_dir: Path,
    eps: float,
    c5_cfg: dict | None,
    c6_cfg: dict | None,
) -> dict:
    if not c5_cfg and not c6_cfg:
        return {}

    from resdata.summary import Summary  # type: ignore[import-not-found]

    smspec = detect_smspec_path(output_dir)
    s = Summary(str(smspec))

    years = [float(x) for x in s.numpy_vector("YEARS")]
    if not years:
        raise ValueError("Missing YEARS vector in summary")

    horizon_years = max(years)

    def vec(key: str) -> list[float]:
        if not s.has_key(key):
            raise KeyError(f"Missing summary key {key}")
        return [float(x) for x in s.numpy_vector(key)]

    out: dict = {}

    if c5_cfg and bool(c5_cfg.get("enable", False)):
        c5_years = float(c5_cfg.get("years", 2.0))
        wc_eps = float(c5_cfg.get("wc_eps", 0.01))
        fopr = vec("FOPR")
        fwpr = vec("FWPR")
        if len(fopr) != len(fwpr) or len(fopr) != len(years):
            raise ValueError("Summary vector length mismatch for FOPR/FWPR/YEARS")
        wc = [qw / (qo + qw + eps) for qo, qw in zip(fopr, fwpr)]
        idxs = [i for i, t in enumerate(years) if t <= c5_years]
        if not idxs:
            raise ValueError(f"No report steps found with YEARS <= {c5_years:g}")
        wc_early_max = max(wc[i] for i in idxs)
        out["WC_early_max"] = wc_early_max
        out["g_wc_early"] = wc_early_max - wc_eps

    if c6_cfg and bool(c6_cfg.get("enable", False)):
        wc_eps = float(c6_cfg.get("wc_eps", 0.01))
        delta_years_max = float(c6_cfg.get("delta_years_max", 1.0))
        prods = ["PROD1", "PROD2", "PROD3", "PROD4"]
        t_bt: list[float] = []
        for p in prods:
            qo = vec(f"WOPR:{p}")
            qw = vec(f"WWPR:{p}")
            if len(qo) != len(qw) or len(qo) != len(years):
                raise ValueError(f"Summary vector length mismatch for producer {p}")
            wcj = [w / (o + w + eps) for o, w in zip(qo, qw)]
            hit = next((years[i] for i in range(len(years)) if wcj[i] >= wc_eps), None)
            t_bt.append(float("inf") if hit is None else float(hit))
            out[f"t_bt_{p}_years"] = t_bt[-1]

        all_inf = all(t == float("inf") for t in t_bt)
        any_inf = any(t == float("inf") for t in t_bt)
        if all_inf:
            spread = 0.0
        elif any_inf:
            spread = float("inf")
        else:
            spread = max(t_bt) - min(t_bt)

        out["bt_spread_years"] = spread
        out["g_bt_spread"] = spread - delta_years_max
        out["horizon_years"] = horizon_years

    return out


def load_cfg(path: Path) -> dict:
    txt = path.read_text()
    suffix = path.suffix.lower()
    if suffix == ".json":
        cfg = json.loads(txt)
    elif suffix in (".yml", ".yaml"):
        cfg = yaml.safe_load(txt)
    else:
        raise ValueError("Config must be .yaml/.yml or .json")
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    return cfg


def save_effective_config(work_dir: Path, cfg: dict) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "effective_config.json"
    out.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def get_required(cfg: dict, key: str):
    if key not in cfg:
        raise KeyError(f"Missing config key: {key}")
    return cfg[key]


def get_required_path(cfg: dict, key: str) -> Path:
    v = get_required(cfg, key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Config key `{key}` must be a non-empty string")
    return Path(v)


def get_required_float(cfg: dict, key: str) -> float:
    return float(get_required(cfg, key))


def get_required_float_list(cfg: dict, key: str, n: int) -> list[float]:
    v = get_required(cfg, key)
    if not isinstance(v, list):
        raise ValueError(f"Config key `{key}` must be a list")
    out = [float(x) for x in v]
    require_len(key, out, n)
    return out


def get_optional_float(cfg: dict, key: str) -> float | None:
    v = cfg.get(key)
    if v is None:
        return None
    return float(v)


def get_optional_int(cfg: dict, key: str) -> int | None:
    v = cfg.get(key)
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError(f"Config key `{key}` must be an integer (not boolean)")
    return int(v)


def get_optional_str(cfg: dict, key: str) -> str | None:
    v = cfg.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError(f"Config key `{key}` must be a string")
    s = v.strip()
    return s if s else None


def get_optional_bool(cfg: dict, key: str) -> bool | None:
    v = cfg.get(key)
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
    raise ValueError(f"Config key `{key}` must be a boolean")

def get_bounds_val(bounds: dict, key: str) -> float | None:
    if key in bounds:
        return float(bounds[key])
    # Allow lowercase aliases in case users prefer python-style keys.
    low = key.lower()
    if low in bounds:
        return float(bounds[low])
    return None


def validate_controls_bounds(
    *,
    stage_inj_rates: list[list[float]],
    stage_prod_bhp: list[list[float]],
    bounds: dict,
) -> None:
    q_inj_min = get_bounds_val(bounds, "Q_INJ_MIN")
    q_inj_max = get_bounds_val(bounds, "Q_INJ_MAX")
    bhp_min = get_bounds_val(bounds, "BHP_MIN")
    bhp_max = get_bounds_val(bounds, "BHP_MAX")

    for s in range(len(stage_inj_rates)):
        inj = stage_inj_rates[s]
        bhp = stage_prod_bhp[s]

        if q_inj_min is not None and any(v < q_inj_min for v in inj):
            raise ValueError(f"Stage {s+1}: injector rate below Q_INJ_MIN={q_inj_min:g}")
        if q_inj_max is not None and any(v > q_inj_max for v in inj):
            raise ValueError(f"Stage {s+1}: injector rate above Q_INJ_MAX={q_inj_max:g}")

        if bhp_min is not None and any(v < bhp_min for v in bhp):
            raise ValueError(f"Stage {s+1}: producer BHP below BHP_MIN={bhp_min:g}")
        if bhp_max is not None and any(v > bhp_max for v in bhp):
            raise ValueError(f"Stage {s+1}: producer BHP above BHP_MAX={bhp_max:g}")

def run_plot(
    *,
    out_dir: Path,
    source: str,
    x: str,
    qliq: bool,
    wc: bool,
    wc_max: float | None,
    save_name: str,
    eps: float,
) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    plot_script = repo_root / "cbo_flow" / "make_plots.py"
    if not plot_script.exists():
        raise FileNotFoundError(f"Missing plot script: {plot_script}")

    save_path = out_dir / save_name

    cmd: list[str] = [
        sys.executable,
        str(plot_script),
        "--output-dir",
        str(out_dir),
        "--field-source",
        source,
        "--x",
        x,
        "--field-save",
        str(save_path),
        "--eps",
        str(eps),
    ]
    if qliq:
        cmd.append("--qliq")
    if wc:
        cmd.append("--wc")
        if wc_max is not None:
            cmd.extend(["--wc-max", str(wc_max)])

    subprocess.run(cmd, check=True)
    return save_path


def main() -> None:
    p = argparse.ArgumentParser(description="Run 1-stage Egg CBO evaluation from YAML config")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("cbo_flow/egg_config.yaml"),
        help="YAML config path",
    )
    args = p.parse_args()

    cfg = load_cfg(args.config)

    paths = get_required(cfg, "paths")
    flow_cfg = cfg.get("flow", {})
    controls = get_required(cfg, "controls")
    economics = get_required(cfg, "economics")
    constraints = get_required(cfg, "constraints")
    plot_cfg = cfg.get("plot", {})
    plot_prod_wc_cfg = cfg.get("plot_prod_wc", {})
    if (
        not isinstance(paths, dict)
        or not isinstance(flow_cfg, dict)
        or not isinstance(controls, dict)
        or not isinstance(economics, dict)
        or not isinstance(constraints, dict)
        or not isinstance(plot_cfg, dict)
        or not isinstance(plot_prod_wc_cfg, dict)
    ):
        raise ValueError(
            "Config must have mappings: paths, flow (optional), controls, economics, constraints, plot (optional), plot_prod_wc (optional)"
        )

    dataset_root = get_required_path(paths, "dataset_root")
    work_dir = get_required_path(paths, "work_dir")
    output_dir = str(get_required(paths, "output_dir"))
    if not output_dir:
        raise ValueError("paths.output_dir must be a non-empty string")

    save_effective_config(work_dir, cfg)

    n_stages = get_optional_int(controls, "n_stages")
    if n_stages is None:
        n_stages = 1
    if n_stages < 1:
        raise ValueError("controls.n_stages must be >= 1")

    stage_tsteps_raw = controls.get("stage_tsteps")
    stage_tsteps: list[int] | None = None
    if stage_tsteps_raw is not None:
        if not isinstance(stage_tsteps_raw, list):
            raise ValueError("controls.stage_tsteps must be a list or null")
        stage_tsteps = [int(x) for x in stage_tsteps_raw]

    controls_vars = controls.get("vars")
    stage_inj_rates: list[list[float]] = []
    stage_prod_bhp: list[list[float]] = []
    if controls_vars is not None:
        if not isinstance(controls_vars, list):
            raise ValueError("controls.vars must be a list of numbers")
        flat = [float(x) for x in controls_vars]
        require_len("controls.vars", flat, 12 * n_stages)
        for s in range(n_stages):
            off = 12 * s
            stage_inj_rates.append(flat[off : off + 8])
            stage_prod_bhp.append(flat[off + 8 : off + 12])
    else:
        if n_stages != 1:
            raise ValueError("For multi-stage, provide controls.vars (length 12*n_stages)")
        stage_inj_rates = [get_required_float_list(controls, "inj_rates", 8)]
        stage_prod_bhp = [get_required_float_list(controls, "prod_bhp", 4)]

    if stage_tsteps is not None:
        if len(stage_tsteps) != n_stages:
            raise ValueError("controls.stage_tsteps length must equal controls.n_stages")

    bounds = controls.get("bounds", {})
    if bounds is not None and not isinstance(bounds, dict):
        raise ValueError("controls.bounds must be a mapping or null")
    if isinstance(bounds, dict) and bounds:
        validate_controls_bounds(stage_inj_rates=stage_inj_rates, stage_prod_bhp=stage_prod_bhp, bounds=bounds)

    perm_realization = get_optional_int(cfg, "perm_realization")
    perm_inc = cfg.get("permeability_include")
    if perm_realization is not None and perm_inc is not None:
        raise ValueError("Use only one of `perm_realization` or `permeability_include` (not both)")

    permeability_include: str | None = None
    if perm_realization is not None:
        if perm_realization < 1 or perm_realization > 101:
            raise ValueError("perm_realization must be in [1, 101]")
        fname = f"PERM{perm_realization}_ECL.INC"
        perm_file = dataset_root / "Permeability_Realizations" / fname
        if not perm_file.exists():
            raise FileNotFoundError(f"Missing permeability realization file: {perm_file}")
        permeability_include = f"../Permeability_Realizations/{fname}"
    elif perm_inc is not None:
        if not isinstance(perm_inc, str) or not perm_inc.strip():
            raise ValueError("permeability_include must be a non-empty string or null")
        permeability_include = perm_inc.strip()

    oil_price = get_required_float(economics, "oil_price_usd_per_sm3")
    wprod_cost = get_required_float(economics, "wprod_cost_usd_per_sm3")
    winj_cost = get_required_float(economics, "winj_cost_usd_per_sm3")

    eps = get_required_float(constraints, "eps")
    wc_max = None

    qliq_max = None
    qwprod_max = None
    qwinj_max = None
    qliq_limit_factor = None
    qwprod_limit_factor = None

    def get_constraint_block(name: str) -> dict | None:
        v = constraints.get(name)
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError(f"constraints.{name} must be a mapping or null")
        return v

    def get_constraint_enable_and_limit(block_name: str, legacy_key: str) -> tuple[bool, float | None]:
        block = get_constraint_block(block_name)
        if block is None:
            enable = True
            limit = get_optional_float(constraints, legacy_key)
            return enable, limit
        enable = bool(get_optional_bool(block, "enable") if isinstance(block, dict) else True)
        limit = get_optional_float(block, "limit")
        return enable, limit

    c1_enable, qwinj_max = get_constraint_enable_and_limit("C1_PeakFieldWaterInjectionRate", "qwinj_max")
    c2_enable, qwprod_max = get_constraint_enable_and_limit("C2_PeakFieldWaterProductionRate", "qwprod_max")
    c3_enable, qliq_max = get_constraint_enable_and_limit("C3_PeakFieldLiquidRate", "qliq_max")
    c4_enable, wc_max = get_constraint_enable_and_limit("C4_PeakFieldWaterCut", "wc_max")

    c2_block = get_constraint_block("C2_PeakFieldWaterProductionRate") or {}
    c3_block = get_constraint_block("C3_PeakFieldLiquidRate") or {}
    qwprod_limit_factor = get_optional_float(c2_block, "limit_factor") if isinstance(c2_block, dict) else None
    qliq_limit_factor = get_optional_float(c3_block, "limit_factor") if isinstance(c3_block, dict) else None

    defaults = constraints.get("defaults", {})
    if defaults is not None and not isinstance(defaults, dict):
        raise ValueError("constraints.defaults must be a mapping or null")

    c1_cfg = constraints.get("c1")
    c3b_cfg = constraints.get("c3b")
    c5_cfg = constraints.get("C5_EarlyWaterCutUpToYears", constraints.get("c5", c1_cfg))
    c6_cfg = constraints.get("C6_BreakthroughTimeSpreadAcrossProducers", constraints.get("c6", c3b_cfg))

    used_c5_keys = [k for k in ["C5_EarlyWaterCutUpToYears", "c5", "c1"] if constraints.get(k) is not None]
    used_c6_keys = [k for k in ["C6_BreakthroughTimeSpreadAcrossProducers", "c6", "c3b"] if constraints.get(k) is not None]
    if len(used_c5_keys) > 1:
        raise ValueError(f"Use only one C5 config key, got: {used_c5_keys}")
    if len(used_c6_keys) > 1:
        raise ValueError(f"Use only one C6 config key, got: {used_c6_keys}")
    if c5_cfg is not None and not isinstance(c5_cfg, dict):
        raise ValueError("C5 config must be a mapping or null")
    if c6_cfg is not None and not isinstance(c6_cfg, dict):
        raise ValueError("C6 config must be a mapping or null")

    if (c1_enable and qwinj_max is None) or (c2_enable and qwprod_max is None) or (c3_enable and qliq_max is None):
        bounds = controls.get("bounds", {})
        if isinstance(bounds, dict) and bounds:
            q_inj_max = get_bounds_val(bounds, "Q_INJ_MAX")
        else:
            q_inj_max = None

        base_inj_capacity = (q_inj_max * 8.0) if q_inj_max is not None else None

        if c1_enable and qwinj_max is None and base_inj_capacity is not None:
            qwinj_max = base_inj_capacity

        if c3_enable and qliq_max is None:
            base = qwinj_max if qwinj_max is not None else base_inj_capacity
            if base is None:
                raise ValueError("Cannot derive C3 limit: missing C1 limit and missing controls.bounds.Q_INJ_MAX")
            if qliq_limit_factor is not None:
                qliq_max = float(qliq_limit_factor) * base
            else:
                qliq_factor = float(defaults.get("qliq_factor", 1.1)) if isinstance(defaults, dict) else 1.1
                qliq_max = qliq_factor * base

        if c2_enable and qwprod_max is None:
            base = qwinj_max if qwinj_max is not None else base_inj_capacity
            if base is None:
                raise ValueError("Cannot derive C2 limit: missing C1 limit and missing controls.bounds.Q_INJ_MAX")
            if qwprod_limit_factor is not None:
                qwprod_max = float(qwprod_limit_factor) * base
            else:
                qwprod_factor = float(defaults.get("qwprod_factor", 0.9)) if isinstance(defaults, dict) else 0.9
                if qliq_max is not None:
                    qwprod_max = qwprod_factor * qliq_max

    # Apply enable flags: if disabled, do not enforce / do not include residual.
    if not c1_enable:
        qwinj_max = None
    if not c2_enable:
        qwprod_max = None
    if not c3_enable:
        qliq_max = None
    if not c4_enable:
        wc_max = None

    prepare_workdir(
        dataset_root=dataset_root,
        work_dir=work_dir,
        stage_inj_rates=stage_inj_rates,
        stage_prod_bhp=stage_prod_bhp,
        stage_tsteps=stage_tsteps,
        permeability_include=permeability_include,
    )

    cmd = ["flow", "Eclipse/Egg_Model_ECL.DATA", f"--output-dir={output_dir}"]
    enable_ecl_output = get_optional_bool(flow_cfg, "enable_ecl_output")
    plot_prod_wc_enable = bool(get_optional_bool(plot_prod_wc_cfg, "enable") or False)
    if plot_prod_wc_enable and (enable_ecl_output is False):
        raise ValueError("plot_prod_wc.enable requires flow.enable_ecl_output=true (needs SMSPEC/UNSMRY for WOPR/WWPR)")
    if enable_ecl_output is not None:
        cmd.append(f"--enable-ecl-output={'true' if enable_ecl_output else 'false'}")
    subprocess.run(cmd, check=True, cwd=work_dir)

    out_dir = work_dir / output_dir
    result = evaluate_outputs(
        output_dir=out_dir,
        eps=eps,
        oil_price_usd_per_sm3=oil_price,
        wprod_cost_usd_per_sm3=wprod_cost,
        winj_cost_usd_per_sm3=winj_cost,
        qliq_max=qliq_max,
        qwprod_max=qwprod_max,
        qwinj_max=qwinj_max,
        wc_max=wc_max,
    )

    bt = compute_breakthrough_constraints(
        output_dir=out_dir,
        eps=eps,
        c5_cfg=c5_cfg if isinstance(c5_cfg, dict) else None,
        c6_cfg=c6_cfg if isinstance(c6_cfg, dict) else None,
    )
    result.update(bt)

    plot_enable = get_optional_bool(plot_cfg, "enable")
    if plot_enable:
        source = get_optional_str(plot_cfg, "source") or "prt"
        x = get_optional_str(plot_cfg, "x") or "days"
        qliq = bool(get_optional_bool(plot_cfg, "qliq") or False)
        wc = bool(get_optional_bool(plot_cfg, "wc") or False)
        save_name = get_optional_str(plot_cfg, "save") or f"rates_{source}.png"

        if source not in ("prt", "bin"):
            raise ValueError("plot.source must be 'prt' or 'bin'")
        if x not in ("days", "dates"):
            raise ValueError("plot.x must be 'days' or 'dates'")

        save_path = run_plot(
            out_dir=out_dir,
            source=source,
            x=x,
            qliq=qliq,
            wc=wc,
            wc_max=wc_max,
            save_name=save_name,
            eps=eps,
        )
        print(f"PLOT = '{save_path}';")

    if plot_prod_wc_enable:
        x = get_optional_str(plot_prod_wc_cfg, "x") or "dates"
        save_dir_name = get_optional_str(plot_prod_wc_cfg, "save_dir") or "prod_wc_plots"
        prefix = get_optional_str(plot_prod_wc_cfg, "prefix") or "WC"
        wc_lines_raw = plot_prod_wc_cfg.get("wc_lines", [])
        if x not in ("days", "dates"):
            raise ValueError("plot_prod_wc.x must be 'days' or 'dates'")
        if wc_lines_raw is not None and not isinstance(wc_lines_raw, list):
            raise ValueError("plot_prod_wc.wc_lines must be a list or null")
        wc_lines = ",".join(str(float(v)) for v in (wc_lines_raw or []))

        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "cbo_flow" / "make_plots.py"
        if not script.exists():
            raise FileNotFoundError(f"Missing plot script: {script}")

        save_dir = out_dir / save_dir_name
        cmd3 = [
            sys.executable,
            str(script),
            "--output-dir",
            str(out_dir),
            "--field-source",
            "none",
            "--per-well-wc",
            "--wells",
            ",".join(["PROD1", "PROD2", "PROD3", "PROD4"]),
            "--x",
            x,
            "--per-well-save-dir",
            str(save_dir),
            "--eps",
            str(eps),
            "--per-well-prefix",
            prefix,
        ]
        if wc_lines:
            cmd3.extend(["--wc-lines", wc_lines])
        subprocess.run(cmd3, check=True)
        print(f"PLOT_PROD_WC_DIR = '{save_dir}';")

    # MATLAB-style output (easy to parse).
    def fmt_num(v: float) -> str:
        s = f"{float(v):.2f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    for k in [
        "FOPT_T_sm3",
        "FWPT_T_sm3",
        "FWIT_T_sm3",
        "NPV_USD",
        "OBJ",
        "FOPR_peak",
        "FWPR_peak",
        "FWIR_peak",
        "Qliq_peak",
        "WC_peak",
    ]:
        print(f"{k} = {fmt_num(result[k])};")

    # Breakthrough times (years) for producers (PROD1..PROD4), if C3B computed them.
    # Printed before the C1..C6 lines for easier reading.
    t_bt_keys = [f"t_bt_PROD{i}_years" for i in range(1, 5)]
    if all(k in result for k in t_bt_keys):
        vals_txt = " ".join(
            [
                ("Inf" if math.isinf(float(result[k])) and float(result[k]) > 0 else fmt_num(float(result[k])))
                for k in t_bt_keys
            ]
        )
        print(f"t_bt_years = [{vals_txt}];")
    c_names: list[str] = []
    c_vals: list[float] = []
    c_ok: list[int] = []

    for spec in CONSTRAINT_SPECS:
        name = str(spec["name"])
        result_key = str(spec["result_key"])

        val_obj = result.get(result_key)
        if val_obj is None:
            continue
        val = float(val_obj)

        c_names.append(name)
        c_vals.append(val)
        c_ok.append(1 if (math.isfinite(val) and val <= 0.0) else 0)
        print(f"{name} = {fmt_num(val)};")

    if c_names:
        print(f"N_CONSTRAINTS = {len(c_names)};")
        ok_txt = " ".join(str(x) for x in c_ok)
        print(f"C_ok = [{ok_txt}];")

    print(f"RUN_DIR = '{work_dir}';")
    print(f"OUTPUT_DIR = '{out_dir}';")


if __name__ == "__main__":
    main()
