from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .prt import extract_field_series


@dataclasses.dataclass(frozen=True)
class Economics:
    oil_price: float = 1.0
    water_prod_cost: float = 0.0
    water_inj_cost: float = 0.0


@dataclasses.dataclass(frozen=True)
class ConstraintCaps:
    q_liq_max: float
    q_wprod_max: float
    q_winj_max: float
    wc_max: float | None = None
    eps: float = 1e-12


def _apply_spe1_controls(deck_text: str, prod_orat: float | None, inj_rate: float | None) -> str:
    """
    Minimal control injection for the SPE1 deck format.

    - Updates WCONPROD ORAT target for well 'PROD'
    - Updates WCONINJE RATE target for well 'INJ' (type stays as in deck)
    """
    out = deck_text
    if prod_orat is not None:
        out = out.replace(
            "'PROD' 'OPEN' 'ORAT' 20000 4* 1000 /",
            f"'PROD' 'OPEN' 'ORAT' {prod_orat:g} 4* 1000 /",
        )
    if inj_rate is not None:
        out = out.replace(
            "'INJ'\t'GAS'\t'OPEN'\t'RATE'\t100000 1* 9014 /",
            f"'INJ'\t'GAS'\t'OPEN'\t'RATE'\t{inj_rate:g} 1* 9014 /",
        )
        out = out.replace(
            "'INJ' 'GAS' 'OPEN' 'RATE' 100000 1* 9014 /",
            f"'INJ' 'GAS' 'OPEN' 'RATE' {inj_rate:g} 1* 9014 /",
        )
    return out


def run_flow(
    deck_path: str | Path,
    *,
    run_dir: str | Path,
    prod_orat: float | None = None,
    inj_rate: float | None = None,
    extra_flow_args: list[str] | None = None,
) -> Path:
    deck_path = Path(deck_path)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    deck_text = deck_path.read_text()
    deck_text = _apply_spe1_controls(deck_text, prod_orat=prod_orat, inj_rate=inj_rate)

    deck_copy_path = run_dir / deck_path.name
    deck_copy_path.write_text(deck_text)

    cmd = ["flow", f"--output-dir={run_dir}"]
    if extra_flow_args:
        cmd.extend(extra_flow_args)
    cmd.append(str(deck_copy_path))

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    prt_path = run_dir / (deck_path.stem + ".PRT")
    if not prt_path.exists():
        # Flow sometimes uses the case name from the deck file (without extension).
        candidates = list(run_dir.glob("*.PRT"))
        if len(candidates) == 1:
            prt_path = candidates[0]
        else:
            raise FileNotFoundError(f"Could not find .PRT in {run_dir}")
    return prt_path


def compute_npv(series: Any, econ: Economics) -> float:
    return (
        econ.oil_price * series.cum_oil
        - econ.water_prod_cost * series.cum_water_prod
        - econ.water_inj_cost * series.cum_water_inj
    )


def compute_constraints(series: Any, caps: ConstraintCaps) -> dict[str, float]:
    liq = [qo + qw for qo, qw in zip(series.oil_rate, series.water_prod_rate)]
    max_liq = max(liq) if liq else 0.0
    max_wprod = max(series.water_prod_rate) if series.water_prod_rate else 0.0
    max_winj = max(series.water_inj_rate) if series.water_inj_rate else 0.0

    out: dict[str, float] = {
        "g_liq": max_liq - caps.q_liq_max,
        "g_wprod": max_wprod - caps.q_wprod_max,
        "g_winj": max_winj - caps.q_winj_max,
    }

    if caps.wc_max is not None:
        wc = [
            qw / (qo + qw + caps.eps)
            for qo, qw in zip(series.oil_rate, series.water_prod_rate)
        ]
        out["g_wc"] = (max(wc) if wc else 0.0) - caps.wc_max
    return out


def evaluate(
    deck_path: str | Path,
    *,
    prod_orat: float | None = None,
    inj_rate: float | None = None,
    econ: Economics | None = None,
    caps: ConstraintCaps | None = None,
    work_root: str | Path = "runs",
) -> dict[str, Any]:
    econ = econ or Economics()
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    key = json.dumps({"deck": str(deck_path), "prod_orat": prod_orat, "inj_rate": inj_rate}, sort_keys=True).encode()
    run_id = hashlib.sha256(key).hexdigest()[:10]
    run_dir = work_root / f"run_{run_id}"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    prt_path = run_flow(deck_path, run_dir=run_dir, prod_orat=prod_orat, inj_rate=inj_rate)
    series = extract_field_series(prt_path)

    npv = compute_npv(series, econ=econ)
    result: dict[str, Any] = {"npv": npv}
    if caps is not None:
        result["constraints"] = compute_constraints(series, caps=caps)
    result["run_dir"] = str(run_dir)
    result["prt"] = str(prt_path)
    return result

