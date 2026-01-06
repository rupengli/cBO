#!/usr/bin/env python3
"""
Plot Egg CBO time-series outputs (with legend) from an OPM Flow output directory.

Sources:
  - prt: parse report tables in *.PRT
  - bin: read summary vectors from *.SMSPEC/*.UNSMRY via resdata or ERT

Examples:
  python3 cbo_flow/plot_series.py --output-dir Egg_Model_Data_Files_v2/out_serial --source prt --save rates_prt.png
  python3 cbo_flow/plot_series.py --output-dir Egg_Model_Data_Files_v2/out_serial --source bin --x dates --save rates_bin.png
  python3 cbo_flow/plot_series.py --output-dir Egg_Model_Data_Files_v2/out_serial --source bin --x dates --wc --wc-max 0.95 --save wc_limit.png
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

EPS = 1e-12


def ensure_repo_on_path() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    sys.path.insert(0, str(repo_root))


@dataclass(frozen=True)
class Series:
    x_days: list[float]
    x_dates: list[datetime] | None
    fopr: list[float]
    fwpr: list[float]
    fwir: list[float]
    unit_rate: str


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


def parse_prt_date(s: str | None) -> datetime | None:
    if s is None:
        return None
    raw = s.strip()
    for fmt in ("%d %b %Y", "%d-%b-%Y", "%d %B %Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    return None


def load_series_from_prt(prt_path: Path) -> Series:
    ensure_repo_on_path()
    from cbo_flow.prt import parse_prt_report_steps  # noqa: E402

    steps = parse_prt_report_steps(prt_path)
    if not steps:
        raise ValueError(f"No report steps parsed from {prt_path}")

    x_days = [float(st.day) for st in steps]
    dates = [parse_prt_date(st.date) for st in steps]
    x_dates = dates if all(d is not None for d in dates) else None

    def get_cell(row: dict[str, str], token_a: str, token_b: str) -> float:
        for k, v in row.items():
            u = k.upper()
            if token_a in u and token_b in u:
                return float(v.strip())
        raise KeyError(f"Missing column containing {token_a}+{token_b} in PRT report")

    fopr = [get_cell(st.prod, "OIL", "RATE") if st.prod else 0.0 for st in steps]
    fwpr = [get_cell(st.prod, "WATER", "RATE") if st.prod else 0.0 for st in steps]
    fwir = [get_cell(st.inj, "WATER", "RATE") if st.inj else 0.0 for st in steps]

    return Series(
        x_days=x_days,
        x_dates=x_dates,
        fopr=fopr,
        fwpr=fwpr,
        fwir=fwir,
        unit_rate="SCM/DAY",
    )


def load_series_from_binaries(case_base: Path) -> Series:
    if importlib.util.find_spec("ecl") is not None:
        from ecl.summary import EclSum  # type: ignore[import-not-found]

        summary = EclSum(str(case_base))

        def vec(k: str) -> list[float]:
            v = summary.numpy_vector(k)
            return [float(x) for x in v]

        x_dates = list(summary.dates) if hasattr(summary, "dates") else None
        unit_rate = "SM3/DAY"

    elif importlib.util.find_spec("resdata") is not None:
        from resdata.summary import Summary  # type: ignore[import-not-found]

        summary = Summary(str(case_base))

        def vec(k: str) -> list[float]:
            v = summary.numpy_vector(k)
            return [float(x) for x in v]

        x_dates = list(summary.dates)
        unit_rate = summary.unit("FOPR")
    else:
        raise ImportError("Binary reading requires `resdata` or ERT (`ecl.summary`).")

    years = vec("YEARS")
    x_days = [y * 365.25 for y in years]

    return Series(
        x_days=x_days,
        x_dates=x_dates,
        fopr=vec("FOPR"),
        fwpr=vec("FWPR"),
        fwir=vec("FWIR"),
        unit_rate=unit_rate,
    )


def plot_series(
    series: Series,
    *,
    x_mode: str,
    show_fopr: bool,
    show_fwpr: bool,
    show_fwir: bool,
    show_qliq: bool,
    show_wc: bool,
    wc_max: float | None,
    title: str,
    save: Path | None,
) -> None:
    if x_mode == "dates":
        if series.x_dates is None:
            raise ValueError("dates x-axis requested but no dates are available from this source")
        x = series.x_dates
        xlabel = "Date"
    else:
        x = series.x_days
        xlabel = "Days"

    fig, ax1 = plt.subplots(figsize=(10, 5))

    handles: list[object] = []
    labels: list[str] = []

    if show_fopr:
        (ln,) = ax1.plot(x, series.fopr, label=f"FOPR ({series.unit_rate})")
        handles.append(ln)
        labels.append(ln.get_label())
    if show_fwpr:
        (ln,) = ax1.plot(x, series.fwpr, label=f"FWPR ({series.unit_rate})")
        handles.append(ln)
        labels.append(ln.get_label())
    if show_fwir:
        (ln,) = ax1.plot(x, series.fwir, label=f"FWIR ({series.unit_rate})")
        handles.append(ln)
        labels.append(ln.get_label())
    if show_qliq:
        qliq = [qo + qw for qo, qw in zip(series.fopr, series.fwpr)]
        (ln,) = ax1.plot(x, qliq, label=f"QLIQ=FOPR+FWPR ({series.unit_rate})")
        handles.append(ln)
        labels.append(ln.get_label())

    ax2 = None
    if show_wc:
        ax2 = ax1.twinx()
        wc = [qw / (qo + qw + EPS) for qo, qw in zip(series.fopr, series.fwpr)]
        (ln_wc,) = ax2.plot(x, wc, color="tab:purple", label="WC=FWPR/(FOPR+FWPR+eps) (-)")
        handles.append(ln_wc)
        labels.append(ln_wc.get_label())
        if wc_max is not None:
            ln_lim = ax2.axhline(
                wc_max,
                linestyle="--",
                linewidth=1.5,
                color="tab:purple",
                alpha=0.9,
                label=f"WCmax={wc_max:g} (-)",
            )
            handles.append(ln_lim)
            labels.append(ln_lim.get_label())
        ax2.set_ylabel("Water cut (-)")
        ax2.set_ylim(0.0, 1.0)

    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(f"Rates ({series.unit_rate})" if (show_fopr or show_fwpr or show_fwir or show_qliq) else "Value")
    ax1.set_title(title)
    ax1.legend(handles, labels)
    fig.tight_layout()

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150)
        return

    plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description="Plot field rate time series (legend included) from Flow outputs")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--source", default="prt", choices=["prt", "bin"])
    p.add_argument("--case", type=Path, default=None, help="PRT path or case base / SMSPEC path; auto-detect if omitted")
    p.add_argument("--x", default="days", choices=["days", "dates"], help="x-axis")
    p.add_argument("--no-fopr", action="store_true")
    p.add_argument("--no-fwpr", action="store_true")
    p.add_argument("--no-fwir", action="store_true")
    p.add_argument("--qliq", action="store_true", help="also plot QLIQ(t)=FOPR+FWPR")
    p.add_argument("--wc", action="store_true", help="also plot WC(t)=FWPR/(FOPR+FWPR+eps)")
    p.add_argument("--wc-max", type=float, default=None, help="also plot water-cut limit line (WCmax)")
    p.add_argument("--save", type=Path, default=None, help="save figure to a file instead of showing")
    args = p.parse_args()

    output_dir: Path = args.output_dir
    source: str = args.source

    case_path = args.case if args.case is not None else detect_case(output_dir, source=source)

    if source == "prt":
        series = load_series_from_prt(case_path)
        title = f"Field rates from PRT: {case_path.name}"
    else:
        case_base = case_path.with_suffix("") if case_path.suffix.upper() == ".SMSPEC" else case_path
        series = load_series_from_binaries(case_base)
        title = f"Field rates from binaries: {case_base.name}"

    plot_series(
        series,
        x_mode=args.x,
        show_fopr=not args.no_fopr,
        show_fwpr=not args.no_fwpr,
        show_fwir=not args.no_fwir,
        show_qliq=args.qliq,
        show_wc=(args.wc or args.wc_max is not None),
        wc_max=args.wc_max,
        title=title,
        save=args.save,
    )


if __name__ == "__main__":
    main()
