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
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Totals:
    cum_oil: float
    cum_water_prod: float
    cum_water_inj: float


def _split_cells(line: str) -> list[str]:
    parts = line.rstrip("\n").split(":")
    if len(parts) < 3:
        return []
    return [cell.strip() for cell in parts[1:-1]]


def _merge_headers(header_lines: list[str]) -> list[str]:
    rows = [_split_cells(h) for h in header_lines]
    if not rows or any(not r for r in rows):
        raise ValueError("Failed to parse table header lines")
    ncols = min(len(r) for r in rows)
    merged: list[str] = []
    for col_idx in range(ncols):
        pieces = [rows[row_idx][col_idx] for row_idx in range(len(rows))]
        pieces = [p for p in pieces if p]
        merged.append(" ".join(pieces))
    return merged


def _parse_table(lines: Iterable[str]) -> tuple[dict[str, dict[str, str]], list[str]]:
    it = iter(lines)
    header_lines = [next(it), next(it), next(it)]
    columns = _merge_headers(header_lines)

    rows: dict[str, dict[str, str]] = {}
    remainder: list[str] = []

    for line in it:
        if line.startswith(" :--------") or line.startswith(":--------"):
            remainder.extend(it)
            break
        if not line.startswith(":"):
            remainder.append(line)
            continue
        cells = _split_cells(line)
        if not cells:
            continue
        row_name = cells[0]
        row: dict[str, str] = {}
        for col_idx in range(min(len(columns), len(cells))):
            row[columns[col_idx]] = cells[col_idx]
        rows[row_name] = row

    return rows, remainder


def _find_col(columns: Iterable[str], *, must_have: list[str], must_not: list[str] | None = None) -> str:
    must_not = must_not or []
    for col in columns:
        u = col.upper()
        if all(tok.upper() in u for tok in must_have) and all(tok.upper() not in u for tok in must_not):
            return col
    raise KeyError(f"Could not find column with tokens {must_have} (excluding {must_not})")


def _as_float(s: str) -> float:
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

    rows, _ = _parse_table(lines[i:])
    field = rows["FIELD"]
    columns = list(field.keys())

    col_oil = _find_col(columns, must_have=["OIL", "PROD"])
    col_wprod = _find_col(columns, must_have=["WATER", "PROD"])
    col_winj = _find_col(columns, must_have=["WATER", "INJ"])

    return Totals(
        cum_oil=_as_float(field[col_oil]),
        cum_water_prod=_as_float(field[col_wprod]),
        cum_water_inj=_as_float(field[col_winj]),
    )


def _detect_case(output_dir: Path, *, source: str) -> Path:
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


def _get_last_value(summary: object, keyword: str) -> float:
    if hasattr(summary, "numpy_vector"):
        vec = summary.numpy_vector(keyword)
        return float(vec[-1])
    if hasattr(summary, "get_values"):
        vec = summary.get_values(keyword)
        return float(vec[-1])
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
        cum_oil=_get_last_value(summary, "FOPT"),
        cum_water_prod=_get_last_value(summary, "FWPT"),
        cum_water_inj=_get_last_value(summary, "FWIT"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CumOil/CumWaterProd/CumWaterInj from a Flow output dir")
    parser.add_argument("--output-dir", required=True, type=Path, help="Flow output directory (contains .PRT/.SMSPEC)")
    parser.add_argument(
        "--source",
        required=True,
        choices=["prt", "bin"],
        help="Where to extract from: prt (text) or bin (SMSPEC+UNSMRY)",
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

    case_path: Path
    if args.case is None:
        case_path = _detect_case(output_dir, source=source)
    else:
        case_path = args.case

    if source == "prt":
        totals = extract_totals_from_prt(case_path)
    else:
        case_base = case_path.with_suffix("") if case_path.suffix.upper() == ".SMSPEC" else case_path
        totals = extract_totals_from_binaries(case_base)

    print(f"CumOil_T = {totals.cum_oil};")
    print(f"CumWaterProd_T = {totals.cum_water_prod};")
    print(f"CumWaterInj_T = {totals.cum_water_inj};")


if __name__ == "__main__":
    main()

