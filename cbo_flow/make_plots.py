#!/usr/bin/env python3
"""
Generate plots for the same Flow output directory.

Outputs:
  - Field rates from PRT and/or binaries
  - Producer water cut (one plot with all producers, and/or one PNG per producer)
  - Optional merged 3-panel figure (PRT + binaries + producer WC)

Example:
  python3 cbo_flow/make_plots.py --output-dir Egg_Model_Data_Files_v2/out_serial --x dates --qliq --wc --wc-max 0.95 --prod-wc
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt


def parse_float_csv(csv: str) -> list[float]:
    if not csv.strip():
        return []
    return [float(x.strip()) for x in csv.split(",") if x.strip()]


def detect_single(output_dir: Path, pat: str) -> Path:
    files = sorted(output_dir.glob(pat))
    if len(files) != 1:
        raise ValueError(f"Expected exactly 1 {pat} in {output_dir}, found {len(files)}")
    return files[0]

def ensure_repo_on_path() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    sys.path.insert(0, str(repo_root))


def load_summary(output_dir: Path):
    from resdata.summary import Summary  # type: ignore[import-not-found]

    smspec = detect_single(output_dir, "*.SMSPEC")
    return Summary(str(smspec))


def get_vector(summary, key: str) -> list[float]:
    if not summary.has_key(key):
        raise KeyError(f"Missing summary key {key}")
    v = summary.numpy_vector(key)
    return [float(x) for x in v]


def get_x(summary, x_mode: str) -> tuple[list[float] | list[datetime], str]:
    if x_mode == "dates":
        if not hasattr(summary, "dates"):
            raise ValueError("dates x-axis requested but summary has no dates")
        return list(summary.dates), "Date"
    years = get_vector(summary, "YEARS")
    return [y * 365.25 for y in years], "Days"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_prt_date(s: str | None) -> datetime | None:
    if s is None:
        return None
    raw = s.strip()
    for fmt in (
        "%d %b %Y",
        "%d-%b-%Y",
        "%d %B %Y",
        "%d-%b-%Y %H:%M:%S",
        "%d %b %Y %H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    return None


@dataclass(frozen=True)
class Series:
    x_days: list[float]
    x_dates: list[datetime] | None
    fopr: list[float]
    fwpr: list[float]
    fwir: list[float]
    unit_rate: str


def load_field_series_from_prt(output_dir: Path) -> Series:
    ensure_repo_on_path()
    from cbo_flow.prt import parse_prt_report_steps

    prt_path = detect_single(output_dir, "*.PRT")
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


def load_field_series_from_binaries(output_dir: Path) -> Series:
    summary = load_summary(output_dir)

    def vec(k: str) -> list[float]:
        return get_vector(summary, k)

    years = vec("YEARS")
    x_days = [y * 365.25 for y in years]
    x_dates = list(summary.dates) if hasattr(summary, "dates") else None

    unit_rate = summary.unit("FOPR")
    if not unit_rate:
        unit_rate = "SM3/DAY"

    return Series(
        x_days=x_days,
        x_dates=x_dates,
        fopr=vec("FOPR"),
        fwpr=vec("FWPR"),
        fwir=vec("FWIR"),
        unit_rate=str(unit_rate),
    )


def plot_field_series(
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
    save: Path,
    eps: float,
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

    if show_wc:
        ax2 = ax1.twinx()
        wc = [qw / (qo + qw + eps) for qo, qw in zip(series.fopr, series.fwpr)]
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

    ensure_dir(save.parent)
    fig.savefig(save, dpi=150)
    plt.close(fig)


def plot_producer_wc_combined(
    *,
    output_dir: Path,
    wells: list[str],
    x_mode: str,
    eps: float,
    wc_lines: list[float],
    save: Path,
) -> None:
    summary = load_summary(output_dir)
    x, xlabel = get_x(summary, x_mode)

    fig, ax = plt.subplots(figsize=(10, 4))
    for well in wells:
        qo = get_vector(summary, f"WOPR:{well}")
        qw = get_vector(summary, f"WWPR:{well}")
        if len(qo) != len(qw) or len(qo) != len(x):
            raise ValueError(f"Vector length mismatch for {well}")
        wc = [w / (o + w + eps) for o, w in zip(qo, qw)]
        ax.plot(x, wc, linewidth=1.8, label=f"WC {well}")

    for y in wc_lines:
        ax.axhline(y, linestyle="--", linewidth=1.2, alpha=0.8, color="tab:purple")

    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Water cut (-)")
    ax.set_title("Producer water cut")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(save, dpi=150)
    plt.close(fig)


def plot_producer_wc_per_well(
    *,
    output_dir: Path,
    wells: list[str],
    x_mode: str,
    eps: float,
    wc_lines: list[float],
    save_dir: Path,
    prefix: str,
) -> None:
    summary = load_summary(output_dir)
    x, xlabel = get_x(summary, x_mode)

    ensure_dir(save_dir)

    for well in wells:
        qo = get_vector(summary, f"WOPR:{well}")
        qw = get_vector(summary, f"WWPR:{well}")
        if len(qo) != len(qw) or len(qo) != len(x):
            raise ValueError(f"Vector length mismatch for {well}")

        wc = [w / (o + w + eps) for o, w in zip(qo, qw)]

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

        out = save_dir / f"{prefix}_{well}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)


def merge_three_plots(prt_png: Path, bin_png: Path, prod_wc_png: Path, merged_png: Path) -> None:
    imgs = [plt.imread(str(p)) for p in [prt_png, bin_png, prod_wc_png]]
    titles = ["Field rates (PRT)", "Field rates (binaries)", "Producer water cut"]

    fig, axs = plt.subplots(3, 1, figsize=(12, 14))
    for ax, img, title in zip(axs, imgs, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(merged_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Make PRT + binary + producer WC plots, and a merged figure")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--x", default="days", choices=["days", "dates"])
    p.add_argument("--field-source", default="both", choices=["prt", "bin", "both", "none"])
    p.add_argument("--no-fopr", action="store_true")
    p.add_argument("--no-fwpr", action="store_true")
    p.add_argument("--no-fwir", action="store_true")
    p.add_argument("--qliq", action="store_true")
    p.add_argument("--wc", action="store_true")
    p.add_argument("--wc-max", type=float, default=None)
    p.add_argument("--out-prefix", type=str, default="plot")
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--wc-lines", type=str, default="", help="Comma-separated WC horizontal lines, e.g. 0.01,0.95")
    p.add_argument("--field-save", type=Path, default=None, help="Explicit PNG path when field-source is prt or bin")
    p.add_argument("--prod-wc", action="store_true", help="Also plot producer WC (all producers on one plot)")
    p.add_argument("--per-well-wc", action="store_true", help="Also plot WC for each producer (one PNG per well)")
    p.add_argument("--per-well-save-dir", type=Path, default=None, help="Directory to write per-well WC PNGs")
    p.add_argument("--per-well-prefix", type=str, default="WC")
    p.add_argument("--wells", type=str, default="PROD1,PROD2,PROD3,PROD4", help="Comma-separated producer wells")
    p.add_argument("--no-merge", action="store_true", help="Do not write the merged 3-panel figure")
    args = p.parse_args()

    out_dir: Path = args.output_dir

    base = out_dir / args.out_prefix
    prt_png = Path(str(base) + "_prt.png")
    bin_png = Path(str(base) + "_bin.png")
    prod_wc_png = Path(str(base) + "_prod_wc.png")
    merged_png = Path(str(base) + "_merged.png")

    show_wc = bool(args.wc or args.wc_max is not None)
    wells = [w.strip() for w in args.wells.split(",") if w.strip()]
    if not wells:
        raise ValueError("No wells provided")
    wc_lines = parse_float_csv(args.wc_lines)
    eps = float(args.eps)

    if args.field_save is not None and args.field_source in ("both", "none"):
        raise ValueError("--field-save requires --field-source prt or bin")

    wrote_prt = False
    wrote_bin = False

    if args.field_source in ("prt", "both"):
        series = load_field_series_from_prt(out_dir)
        save = args.field_save if args.field_source == "prt" and args.field_save is not None else prt_png
        title = f"Field rates from PRT: {detect_single(out_dir, '*.PRT').name}"
        plot_field_series(
            series,
            x_mode=args.x,
            show_fopr=not args.no_fopr,
            show_fwpr=not args.no_fwpr,
            show_fwir=not args.no_fwir,
            show_qliq=bool(args.qliq),
            show_wc=show_wc,
            wc_max=args.wc_max,
            title=title,
            save=save,
            eps=eps,
        )
        print(f"Wrote: {save}")
        wrote_prt = True

    if args.field_source in ("bin", "both"):
        series = load_field_series_from_binaries(out_dir)
        save = args.field_save if args.field_source == "bin" and args.field_save is not None else bin_png
        case_name = detect_single(out_dir, "*.SMSPEC").with_suffix("").name
        title = f"Field rates from binaries: {case_name}"
        plot_field_series(
            series,
            x_mode=args.x,
            show_fopr=not args.no_fopr,
            show_fwpr=not args.no_fwpr,
            show_fwir=not args.no_fwir,
            show_qliq=bool(args.qliq),
            show_wc=show_wc,
            wc_max=args.wc_max,
            title=title,
            save=save,
            eps=eps,
        )
        print(f"Wrote: {save}")
        wrote_bin = True

    if args.prod_wc:
        plot_producer_wc_combined(
            output_dir=out_dir,
            wells=wells,
            x_mode=args.x,
            eps=eps,
            wc_lines=wc_lines,
            save=prod_wc_png,
        )
        print(f"Wrote: {prod_wc_png}")

    if args.per_well_wc:
        save_dir = args.per_well_save_dir if args.per_well_save_dir is not None else (out_dir / "prod_wc_plots")
        plot_producer_wc_per_well(
            output_dir=out_dir,
            wells=wells,
            x_mode=args.x,
            eps=eps,
            wc_lines=wc_lines,
            save_dir=save_dir,
            prefix=str(args.per_well_prefix),
        )
        print(f"Wrote: {save_dir}")

    if args.prod_wc and wrote_prt and wrote_bin and not args.no_merge:
        if args.field_save is not None:
            raise ValueError("Merged figure requires default file naming (do not use --field-save)")
        if not args.no_merge:
            merge_three_plots(prt_png, bin_png, prod_wc_png, merged_png)
            print(f"Wrote: {merged_png}")


if __name__ == "__main__":
    main()
