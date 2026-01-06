#!/usr/bin/env python3
"""
Generate two separate plots (PRT + binaries) for the same Flow output directory.

Example:
  python3 cbo_flow/make_plots.py --output-dir Egg_Model_Data_Files_v2/out_serial --x dates --qliq --wc --wc-max 0.95
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Make both PRT and binary plots")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--x", default="days", choices=["days", "dates"])
    p.add_argument("--no-fopr", action="store_true")
    p.add_argument("--no-fwpr", action="store_true")
    p.add_argument("--no-fwir", action="store_true")
    p.add_argument("--qliq", action="store_true")
    p.add_argument("--wc", action="store_true")
    p.add_argument("--wc-max", type=float, default=None)
    p.add_argument("--out-prefix", type=str, default="plot")
    args = p.parse_args()

    out_dir: Path = args.output_dir
    script = str(Path(__file__).with_name("plot_series.py"))

    base = out_dir / args.out_prefix
    prt_png = Path(str(base) + "_prt.png")
    bin_png = Path(str(base) + "_bin.png")

    common = [
        script,
        "--output-dir",
        str(out_dir),
        "--x",
        args.x,
    ]
    if args.no_fopr:
        common.append("--no-fopr")
    if args.no_fwpr:
        common.append("--no-fwpr")
    if args.no_fwir:
        common.append("--no-fwir")
    if args.qliq:
        common.append("--qliq")
    if args.wc:
        common.append("--wc")
    if args.wc_max is not None:
        common.extend(["--wc-max", str(args.wc_max)])

    run_cmd([sys.executable, *common, "--source", "prt", "--save", str(prt_png)])
    run_cmd([sys.executable, *common, "--source", "bin", "--save", str(bin_png)])

    print(f"Wrote: {prt_png}")
    print(f"Wrote: {bin_png}")


if __name__ == "__main__":
    main()
