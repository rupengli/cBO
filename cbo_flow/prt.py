from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any, Iterable


@dataclasses.dataclass(frozen=True)
class ReportStep:
    day: float
    date: str | None
    prod: dict[str, str]
    inj: dict[str, str]
    cum: dict[str, str]


_REPORT_STEP_RE = re.compile(
    r"^Report step\s+\d+/\d+\s+at day\s+(?P<day>[0-9]+(?:\.[0-9]+)?)\b.*date\s*=\s*(?P<date>.+?)\s*$"
)
_WELLS_AT_RE = re.compile(r"^\s*WELLS\s+AT\s+(?P<day>[0-9]+(?:\.[0-9]+)?)\s+DAYS\b")
_WELLS_REPORT_LINE_RE = re.compile(r"^\s*REPORT\s+\d+\s+(?P<date>.+?)\s+\*")


def split_cells(line: str) -> list[str]:
    parts = line.rstrip("\n").split(":")
    if len(parts) < 3:
        return []
    return [cell.strip() for cell in parts[1:-1]]


def merge_headers(header_lines: list[str]) -> list[str]:
    rows = [split_cells(h) for h in header_lines]
    if not rows or any(not r for r in rows):
        return []
    ncols = min(len(r) for r in rows)
    merged: list[str] = []
    for col_idx in range(ncols):
        pieces = [rows[row_idx][col_idx] for row_idx in range(len(rows))]
        pieces = [p for p in pieces if p]
        merged.append(" ".join(pieces))
    return merged


def parse_table(lines: Iterable[str]) -> tuple[dict[str, dict[str, str]], list[str]]:
    """
    Parse a Flow PRT colon-delimited table.

    Returns:
      - rows: mapping row_name -> mapping column_name -> raw cell string
      - remainder: the remaining lines after the table (starting at the first non-table line)
    """
    iterator = iter(lines)
    header_lines: list[str] = []

    # Expect 3 header lines starting with " :"
    for _ in range(3):
        header_lines.append(next(iterator))

    columns = merge_headers(header_lines)
    if not columns:
        raise ValueError("Failed to parse table header")

    # Skip separator lines (=== or --- blocks)
    remainder: list[str] = []
    rows: dict[str, dict[str, str]] = {}

    for line in iterator:
        if line.startswith(" :--------"):
            remainder.extend(iterator)
            break
        if not line.startswith(":"):
            remainder.append(line)
            continue
        cells = split_cells(line)
        if not cells:
            continue
        # Some rows may have fewer cells (rare). Only map what we have.
        row_name = cells[0]
        row_map: dict[str, str] = {}
        for col_idx in range(min(len(columns), len(cells))):
            row_map[columns[col_idx]] = cells[col_idx]
        rows[row_name] = row_map

    return rows, remainder


def first_nonempty(mapping: dict[str, str], keys: list[str]) -> str | None:
    for key in keys:
        val = mapping.get(key)
        if val is not None and val.strip() != "":
            return val
    return None


def as_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_prt_report_steps(prt_path: str | Path) -> list[ReportStep]:
    """
    Extract per-report-step FIELD data from Flow's .PRT file.

    This intentionally avoids external deps (resdata/res2df) by parsing the human-readable report tables.
    """
    prt_path = Path(prt_path)
    text = prt_path.read_text(errors="replace").splitlines(keepends=True)

    steps: list[ReportStep] = []
    pending: dict[str, Any] = {}
    current_day: float | None = None
    current_date: str | None = None

    i = 0
    while i < len(text):
        line = text[i]

        m = _REPORT_STEP_RE.match(line.strip("\n"))
        if m:
            if current_day is not None and ("prod" in pending or "inj" in pending or "cum" in pending):
                steps.append(
                    ReportStep(
                        day=current_day,
                        date=current_date,
                        prod=dict(pending.get("prod", {})),
                        inj=dict(pending.get("inj", {})),
                        cum=dict(pending.get("cum", {})),
                    )
                )
                pending = {}

            current_day = float(m.group("day"))
            current_date = m.group("date").strip() if m.group("date") else None
            i += 1
            continue

        m = _WELLS_AT_RE.match(line)
        if m:
            if current_day is not None and ("prod" in pending or "inj" in pending or "cum" in pending):
                steps.append(
                    ReportStep(
                        day=current_day,
                        date=current_date,
                        prod=dict(pending.get("prod", {})),
                        inj=dict(pending.get("inj", {})),
                        cum=dict(pending.get("cum", {})),
                    )
                )
                pending = {}

            current_day = float(m.group("day"))
            current_date = None
            if i + 1 < len(text):
                m2 = _WELLS_REPORT_LINE_RE.match(text[i + 1])
                if m2:
                    current_date = m2.group("date").strip()
                    i += 2
                    continue
            i += 1
            continue

        if "PRODUCTION REPORT" in line:
            # Advance to header (line starting with " :  WELL  :")
            i += 1
            while i < len(text) and " :  WELL  :" not in text[i]:
                i += 1
            if i + 3 >= len(text):
                break
            rows, remainder = parse_table(text[i:])
            pending["prod"] = rows.get("FIELD", {})
            i = len(text) - len(remainder)
            continue

        if "INJECTION REPORT" in line:
            i += 1
            while i < len(text) and " :  WELL  :" not in text[i]:
                i += 1
            if i + 3 >= len(text):
                break
            rows, remainder = parse_table(text[i:])
            pending["inj"] = rows.get("FIELD", {})
            i = len(text) - len(remainder)
            continue

        if "CUMULATIVE PRODUCTION/INJECTION TOTALS" in line:
            i += 1
            while i < len(text) and " :  WELL  :" not in text[i]:
                i += 1
            if i + 3 >= len(text):
                break
            rows, remainder = parse_table(text[i:])
            pending["cum"] = rows.get("FIELD", {})
            i = len(text) - len(remainder)
            continue

        i += 1

    if current_day is not None and ("prod" in pending or "inj" in pending or "cum" in pending):
        steps.append(
            ReportStep(
                day=current_day,
                date=current_date,
                prod=dict(pending.get("prod", {})),
                inj=dict(pending.get("inj", {})),
                cum=dict(pending.get("cum", {})),
            )
        )

    return steps


@dataclasses.dataclass(frozen=True)
class FieldSeries:
    days: list[float]
    oil_rate: list[float]
    water_prod_rate: list[float]
    water_inj_rate: list[float]
    cum_oil: float
    cum_water_prod: float
    cum_water_inj: float


def extract_field_series(prt_path: str | Path) -> FieldSeries:
    steps = parse_prt_report_steps(prt_path)
    if not steps:
        raise ValueError(f"No report steps parsed from {prt_path}")

    days: list[float] = []
    oil_rate: list[float] = []
    water_prod_rate: list[float] = []
    water_inj_rate: list[float] = []

    for st in steps:
        days.append(st.day)

        prod_oil = as_float(
            first_nonempty(st.prod, ["OIL RATE", "OIL RATE STB/DAY", "OIL RATE SCM/DAY"])
        )
        prod_wat = as_float(
            first_nonempty(st.prod, ["WATER RATE", "WATER RATE STB/DAY", "WATER RATE SCM/DAY"])
        )
        inj_wat = as_float(
            first_nonempty(st.inj, ["WATER RATE", "WATER RATE STB/DAY", "WATER RATE SCM/DAY"])
        )

        oil_rate.append(prod_oil or 0.0)
        water_prod_rate.append(prod_wat or 0.0)
        water_inj_rate.append(inj_wat or 0.0)

    last = steps[-1]
    cum_oil = as_float(first_nonempty(last.cum, ["OIL PROD", "OIL PROD MSTB", "OIL PROD MSCM"])) or 0.0
    cum_water_prod = (
        as_float(first_nonempty(last.cum, ["WATER PROD", "WATER PROD MSTB", "WATER PROD MSCM"])) or 0.0
    )
    cum_water_inj = (
        as_float(first_nonempty(last.cum, ["WATER INJ", "WATER INJ MSTB", "WATER INJ MSCM"])) or 0.0
    )

    return FieldSeries(
        days=days,
        oil_rate=oil_rate,
        water_prod_rate=water_prod_rate,
        water_inj_rate=water_inj_rate,
        cum_oil=cum_oil,
        cum_water_prod=cum_water_prod,
        cum_water_inj=cum_water_inj,
    )
