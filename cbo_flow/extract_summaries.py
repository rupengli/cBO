#!/usr/bin/env python3
"""
Extract 3 field summary values from an OPM Flow output directory.

Supports two sources:
  1) PRT text report (always available and easy to parse)
  2) Eclipse summary binaries (SMSPEC+UNSMRY), via an installed Python reader

Default output is "MATLAB style" assignments, e.g.:
  CumOil_T = 498.1;
  CumWaterProd_T = 1791.5;
  CumWaterInj_T = 2289.6;

This script can also extract constraint-related peak rates used by the Egg CBO testbed:
  - Qliq_peak = max_t (FOPR(t) + FWPR(t))
  - Qwprod_peak = max_t FWPR(t)
  - Qwinj_peak = max_t FWIR(t)
  - WC_peak = max_t FWPR(t) / (FOPR(t) + FWPR(t) + eps)   (optional guardrail)
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

EPS = 1e-12


@dataclass(frozen=True)
class Totals:
    cum_oil: float
    cum_water_prod: float
    cum_water_inj: float


@dataclass(frozen=True)
class Peaks:
    fopr_peak: float
    fwpr_peak: float
    fwir_peak: float
    qliq_peak: float
    wc_peak: float


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


def parse_table(lines: Iterable[str]) -> tuple[dict[str, dict[str, str]], list[str]]:
    it = iter(lines)
    header_lines = [next(it), next(it), next(it)]
    columns = merge_headers(header_lines)

    rows: dict[str, dict[str, str]] = {}
    remainder: list[str] = []

    for line in it:
        if line.startswith(" :--------") or line.startswith(":--------"):
            remainder.extend(it)
            break
        if not line.startswith(":"):
            remainder.append(line)
            continue
        cells = split_cells(line)
        if not cells:
            continue
        row_name = cells[0]
        row: dict[str, str] = {}
        for col_idx in range(min(len(columns), len(cells))):
            row[columns[col_idx]] = cells[col_idx]
        rows[row_name] = row

    return rows, remainder


def find_col(columns: Iterable[str], *, must_have: list[str], must_not: list[str] | None = None) -> str:
    must_not = must_not or []
    for col in columns:
        u = col.upper()
        if all(tok.upper() in u for tok in must_have) and all(tok.upper() not in u for tok in must_not):
            return col
    raise KeyError(f"Could not find column with tokens {must_have} (excluding {must_not})")


def as_float(s: str) -> float:
    return float(s.strip())


def extract_totals_from_prt(prt_path: str | Path) -> Totals:
    prt_path = Path(prt_path)
    lines = prt_path.read_text(errors="replace").splitlines(keepends=True)

    idxs = [i for i, line in enumerate(lines) if "CUMULATIVE PRODUCTION/INJECTION TOTALS" in line]
    if not idxs:
        raise ValueError(f"Could not find cumulative totals table in {prt_path}")

    start = idxs[-1] + 1
    i = start
    while i < len(lines) and " :  WELL  :" not in lines[i]:
        i += 1
    if i + 3 >= len(lines):
        raise ValueError(f"Could not find table header after cumulative totals in {prt_path}")

    rows, _ = parse_table(lines[i:])
    field = rows["FIELD"]
    columns = list(field.keys())

    col_oil = find_col(columns, must_have=["OIL", "PROD"])
    col_wprod = find_col(columns, must_have=["WATER", "PROD"])
    col_winj = find_col(columns, must_have=["WATER", "INJ"])

    return Totals(
        cum_oil=as_float(field[col_oil]),
        cum_water_prod=as_float(field[col_wprod]),
        cum_water_inj=as_float(field[col_winj]),
    )


def extract_peaks_from_prt(prt_path: str | Path, *, eps: float) -> Peaks:
    prt_path = Path(prt_path)
    lines = prt_path.read_text(errors="replace").splitlines(keepends=True)

    oil_rates: list[float] = []
    water_prod_rates: list[float] = []
    water_inj_rates: list[float] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if "PRODUCTION REPORT" in line:
            i += 1
            while i < len(lines) and " :  WELL  :" not in lines[i]:
                i += 1
            if i + 3 >= len(lines):
                raise ValueError(f"Could not find production table header in {prt_path}")
            rows, remainder = parse_table(lines[i:])
            field = rows["FIELD"]
            columns = list(field.keys())
            col_oil = find_col(columns, must_have=["OIL", "RATE"])
            col_wat = find_col(columns, must_have=["WATER", "RATE"])
            oil_rates.append(as_float(field[col_oil]))
            water_prod_rates.append(as_float(field[col_wat]))
            i = len(lines) - len(remainder)
            continue

        if "INJECTION REPORT" in line:
            i += 1
            while i < len(lines) and " :  WELL  :" not in lines[i]:
                i += 1
            if i + 3 >= len(lines):
                raise ValueError(f"Could not find injection table header in {prt_path}")
            rows, remainder = parse_table(lines[i:])
            field = rows["FIELD"]
            columns = list(field.keys())
            col_wat = find_col(columns, must_have=["WATER", "RATE"])
            water_inj_rates.append(as_float(field[col_wat]))
            i = len(lines) - len(remainder)
            continue

        i += 1

    if not oil_rates or not water_prod_rates or not water_inj_rates:
        raise ValueError(f"Could not extract report-step rates from {prt_path}")
    if len(oil_rates) != len(water_prod_rates):
        raise ValueError(
            f"Production oil/water rate lengths differ in {prt_path}: {len(oil_rates)} vs {len(water_prod_rates)}"
        )

    fopr_peak = max(oil_rates)
    fwpr_peak = max(water_prod_rates)
    fwir_peak = max(water_inj_rates)
    qliq_peak = max(qo + qw for qo, qw in zip(oil_rates, water_prod_rates))
    wc_peak = max(qw / (qo + qw + EPS) for qo, qw in zip(oil_rates, water_prod_rates))

    return Peaks(
        fopr_peak=fopr_peak,
        fwpr_peak=fwpr_peak,
        fwir_peak=fwir_peak,
        qliq_peak=qliq_peak,
        wc_peak=wc_peak,
    )


def detect_case(output_dir: Path, *, source: str) -> Path:
    if source == "prt":
        prts = sorted(output_dir.glob("*.PRT"))
        if len(prts) != 1:
            raise ValueError(f"Expected exactly 1 .PRT in {output_dir}, found {len(prts)}")
        return prts[0]

    if source == "bin":
        smspec = sorted(output_dir.glob("*.SMSPEC"))
        if len(smspec) != 1:
            raise ValueError(f"Expected exactly 1 .SMSPEC in {output_dir}, found {len(smspec)}")
        return smspec[0]

    raise ValueError(f"Unknown source: {source}")


def get_last_value(summary: object, keyword: str) -> float:
    if hasattr(summary, "numpy_vector"):
        vec = summary.numpy_vector(keyword)
        return float(vec[-1])
    if hasattr(summary, "get_values"):
        vec = summary.get_values(keyword)
        return float(vec[-1])
    raise AttributeError(f"Unsupported summary object: {type(summary).__name__}")


def get_vector(summary: object, keyword: str) -> list[float]:
    if hasattr(summary, "numpy_vector"):
        vec = summary.numpy_vector(keyword)
        return [float(v) for v in vec]
    if hasattr(summary, "get_values"):
        vec = summary.get_values(keyword)
        return [float(v) for v in vec]
    raise AttributeError(f"Unsupported summary object: {type(summary).__name__}")


def extract_totals_from_binaries(case_base: str | Path) -> Totals:
    case_base = Path(case_base)

    # Note: importlib.util.find_spec("pkg.subpkg") can raise if "pkg" isn't importable.
    if importlib.util.find_spec("ecl") is not None:
        from ecl.summary import EclSum  # type: ignore[import-not-found]

        summary = EclSum(str(case_base))
    elif importlib.util.find_spec("resdata") is not None:
        from resdata.summary import Summary  # type: ignore[import-not-found]

        summary = Summary(str(case_base))
    else:
        raise ImportError("Binary summary reading requires `ecl.summary` (ERT) or `resdata.summary`")

    return Totals(
        cum_oil=get_last_value(summary, "FOPT"),
        cum_water_prod=get_last_value(summary, "FWPT"),
        cum_water_inj=get_last_value(summary, "FWIT"),
    )


def extract_peaks_from_binaries(case_base: str | Path, *, eps: float) -> Peaks:
    case_base = Path(case_base)

    # Note: importlib.util.find_spec("pkg.subpkg") can raise if "pkg" isn't importable.
    if importlib.util.find_spec("ecl") is not None:
        from ecl.summary import EclSum  # type: ignore[import-not-found]

        summary = EclSum(str(case_base))
    elif importlib.util.find_spec("resdata") is not None:
        from resdata.summary import Summary  # type: ignore[import-not-found]

        summary = Summary(str(case_base))
    else:
        raise ImportError("Binary summary reading requires `ecl.summary` (ERT) or `resdata.summary`")

    fopr = get_vector(summary, "FOPR")
    fwpr = get_vector(summary, "FWPR")
    fwir = get_vector(summary, "FWIR")

    if not fopr or not fwpr or not fwir:
        raise ValueError("Missing one or more summary vectors: FOPR/FWPR/FWIR")
    if len(fopr) != len(fwpr):
        raise ValueError(f"FOPR/FWPR length mismatch: {len(fopr)} vs {len(fwpr)}")

    fopr_peak = max(fopr)
    fwpr_peak = max(fwpr)
    fwir_peak = max(fwir)
    qliq_peak = max(qo + qw for qo, qw in zip(fopr, fwpr))
    wc_peak = max(qw / (qo + qw + EPS) for qo, qw in zip(fopr, fwpr))

    return Peaks(
        fopr_peak=fopr_peak,
        fwpr_peak=fwpr_peak,
        fwir_peak=fwir_peak,
        qliq_peak=qliq_peak,
        wc_peak=wc_peak,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CumOil/CumWaterProd/CumWaterInj from a Flow output dir")
    parser.add_argument("--output-dir", required=True, type=Path, help="Flow output directory (contains .PRT/.SMSPEC)")
    parser.add_argument(
        "--source",
        default="prt",
        choices=["prt", "bin"],
        help="Where to extract from: prt (text, default) or bin (SMSPEC+UNSMRY)",
    )
    parser.add_argument(
        "--what",
        default="totals",
        choices=["totals", "peaks", "all"],
        help="What to extract: totals (Cum* at T), peaks (constraint-related peak rates), or all",
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=None,
        help="Case path: for prt, a .PRT; for bin, the case base (or .SMSPEC). If omitted, auto-detect.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    source: str = args.source
    what: str = args.what

    case_path: Path
    if args.case is None:
        case_path = detect_case(output_dir, source=source)
    else:
        case_path = args.case

    totals: Totals | None = None
    peaks: Peaks | None = None

    if source == "prt":
        if what in ("totals", "all"):
            totals = extract_totals_from_prt(case_path)
        if what in ("peaks", "all"):
            peaks = extract_peaks_from_prt(case_path, eps=EPS)
    else:
        case_base = case_path.with_suffix("") if case_path.suffix.upper() == ".SMSPEC" else case_path
        if what in ("totals", "all"):
            totals = extract_totals_from_binaries(case_base)
        if what in ("peaks", "all"):
            peaks = extract_peaks_from_binaries(case_base, eps=EPS)

    if totals is not None:
        print(f"CumOil_T = {totals.cum_oil};")
        print(f"CumWaterProd_T = {totals.cum_water_prod};")
        print(f"CumWaterInj_T = {totals.cum_water_inj};")

    if peaks is not None:
        print(f"FOPR_peak = {peaks.fopr_peak};")
        print(f"FWPR_peak = {peaks.fwpr_peak};")
        print(f"FWIR_peak = {peaks.fwir_peak};")
        print(f"Qliq_peak = {peaks.qliq_peak};")
        print(f"WC_peak = {peaks.wc_peak};")


if __name__ == "__main__":
    main()
