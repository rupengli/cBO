#!/usr/bin/env python3
"""
Plot per-well water cut (WC) from Flow binary summary (SMSPEC/UNSMRY).

WC_well(t) = q_w(t) / (q_o(t) + q_w(t) + eps)
using WOPR:<WELL> and WWPR:<WELL> vectors.

Writes one PNG per well into --save-dir.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def detect_case_base(output_dir: Path) -> Path:
    smspec = sorted(output_dir.glob("*.SMSPEC"))
    if len(smspec) != 1:
        raise ValueError(f"Expected exactly 1 .SMSPEC in {output_dir}, found {len(smspec)}")
    return smspec[0]


def load_summary(case_base: Path):
    from resdata.summary import Summary  # type: ignore[import-not-found]

    return Summary(str(case_base))


def get_vector(summary, key: str) -> list[float]:
    if not summary.has_key(key):
        raise KeyError(f"Missing summary key {key}")
    v = summary.numpy_vector(key)
    return [float(x) for x in v]


def get_dates(summary) -> list[datetime] | None:
    if hasattr(summary, "dates"):
        return list(summary.dates)
    return None


def get_days(summary) -> list[float]:
    years = get_vector(summary, "YEARS")
    return [y * 365.25 for y in years]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_float_list(csv: str) -> list[float]:
    if not csv.strip():
        return []
    return [float(x.strip()) for x in csv.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Plot per-producer water cut from SMSPEC/UNSMRY")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--wells", required=True, type=str, help="Comma-separated well names, e.g. PROD1,PROD2")
    p.add_argument("--x", default="dates", choices=["days", "dates"])
    p.add_argument("--save-dir", required=True, type=Path)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--wc-lines", type=str, default="", help="Comma-separated WC horizontal lines, e.g. 0.01,0.95")
    p.add_argument("--prefix", default="WC", type=str)
    args = p.parse_args()

    output_dir = args.output_dir
    case_base = detect_case_base(output_dir)
    summary = load_summary(case_base)

    wells = [w.strip() for w in args.wells.split(",") if w.strip()]
    if not wells:
        raise ValueError("No wells provided")

    wc_lines = parse_float_list(args.wc_lines)

    dates = get_dates(summary)
    if args.x == "dates":
        if dates is None:
            raise ValueError("dates x-axis requested but summary has no dates")
        x = dates
        xlabel = "Date"
    else:
        x = get_days(summary)
        xlabel = "Days"

    ensure_dir(args.save_dir)

    for well in wells:
        qo = get_vector(summary, f"WOPR:{well}")
        qw = get_vector(summary, f"WWPR:{well}")
        if len(qo) != len(qw) or len(qo) != len(x):
            raise ValueError(f"Vector length mismatch for {well}")

        wc = [w / (o + w + args.eps) for o, w in zip(qo, qw)]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, wc, linewidth=1.8, label=f"WC {well}")
        for y in wc_lines:
            ax.axhline(y, linestyle="--", linewidth=1.2, alpha=0.8, color="tab:purple")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Water cut (-)")
        ax.set_title(f"Water cut: {well}")
        ax.legend()
        fig.tight_layout()

        out = args.save_dir / f"{args.prefix}_{well}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()

