#!/usr/bin/env python3
"""
Read binary visibility files written in the order (slowest → fastest):
    time, real/imag, frequency, input, input
for 4 connected inputs (0, 9, 10, 11), and plot the 4×4 visibility matrix.

Assumptions (tweak via CLI if needed):
- dtype: float32 (little-endian). Change with --dtype and --big-endian if required.
- nt: 16384 integrations per file (from your note). Change with --nt if different.
- nin: 4 inputs (full 4×4 matrix). Change with --nin if needed.
- nfreq is inferred from file size.

Examples
--------
# Plot avg |V| over time & frequency for each file in dir
python read_corr_vis.py ~/data/corrs_20251030 --matrix

# Same, but restrict to channels 100..300 and a time slice 0..1024
python read_corr_vis.py ~/data/corrs_20251030 --matrix --chan 100 300 --t 0 1024

# Waterfall for a single baseline (input indices 0,3 -> your physical inputs 0 and 11)
python read_corr_vis.py ~/data/corrs_20251030 --waterfall 0 3 --chan 50 450

# Inspect inferred metadata only
python read_corr_vis.py ~/data/corrs_20251030 --dry-run

Notes
-----
Physical input labels (as wired): [0, 9, 10, 11]. Use --labels to annotate plots.
"""

from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

# --------------------- Core I/O ---------------------

def infer_nfreq(file_size_bytes: int, nt: int, nin: int, dtype: np.dtype) -> int:
    """Infer number of frequency channels from file size.
    Layout: (nt, 2, nfreq, nin, nin) of dtype
    """
    itemsize = dtype.itemsize
    denom = nt * 2 * nin * nin * itemsize
    if denom == 0:
        raise ValueError("Invalid parameters, denominator is zero.")
    if file_size_bytes % denom != 0:
        raise ValueError(
            f"File size {file_size_bytes} not divisible by nt*2*nin*nin*itemsize={denom}.\n"
            f"Try a different dtype (e.g., --dtype float64) or adjust nt/nin."
        )
    nfreq = file_size_bytes // denom
    return int(nfreq)


def load_vis(
    path: Path,
    nt: int = 16384,
    nin: int = 4,
    dtype: str = "float32",
    big_endian: bool = False,
    mmap: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load a single binary file and reshape to complex visibilities.

    Returns
    -------
    vis : np.ndarray
        Complex array of shape (nt, nfreq, nin, nin)
    nfreq : int
        Inferred number of frequency channels
    """
    dt = np.dtype(dtype)
    if big_endian:
        # ensure explicit endianness; '<' little, '>' big
        dt = dt.newbyteorder('>')
    else:
        dt = dt.newbyteorder('<')

    size_bytes = path.stat().st_size
    nfreq = infer_nfreq(size_bytes, nt, nin, dt)

    count = nt * 2 * nfreq * nin * nin
    if mmap:
        raw = np.memmap(path, mode='r', dtype=dt, shape=(count,))
    else:
        raw = np.fromfile(path, dtype=dt, count=count)

    arr = raw.reshape(nt, 2, nfreq, nin, nin)
    vis = arr[:, 0, ...] + 1j * arr[:, 1, ...]  # (nt, nfreq, nin, nin)
    return vis, nfreq

# --------------------- Plotting helpers ---------------------

def plot_matrix(
    vis: np.ndarray,
    t_range: Optional[Tuple[int,int]] = None,
    chan_range: Optional[Tuple[int,int]] = None,
    labels: Optional[List[str]] = None,
    stat: str = "amp",
    title: str = "",
):
    """Plot a 4×4 matrix after averaging over selected time/frequency ranges.

    stat: 'amp' -> |V|, 'phase' -> angle(V) in radians
    """
    v = vis
    nt, nfreq, nin, nin2 = v.shape
    assert nin == nin2

    if t_range is None:
        t0, t1 = 0, nt
    else:
        t0, t1 = max(0, t_range[0]), min(nt, t_range[1])

    if chan_range is None:
        c0, c1 = 0, nfreq
    else:
        c0, c1 = max(0, chan_range[0]), min(nfreq, chan_range[1])

    vsel = v[t0:t1, c0:c1]
    vavg = vsel.mean(axis=(0,1))  # (nin, nin)

    if stat == "amp":
        M = np.abs(vavg)
        cmap = 'viridis'
        cbar_label = '|V| (avg)'
    elif stat == "phase":
        M = np.angle(vavg)
        cmap = 'twilight'
        cbar_label = 'Phase [rad] (avg)'
    else:
        raise ValueError("stat must be 'amp' or 'phase'")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(M, origin='lower', aspect='equal', cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    ticks = np.arange(nin)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if labels and len(labels) == nin:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    else:
        ax.set_xticklabels([str(i) for i in range(nin)])
        ax.set_yticklabels([str(i) for i in range(nin)])

    ax.set_xlabel('Input j')
    ax.set_ylabel('Input i')
    ax.set_title(title or f"Avg {stat} | t[{t0}:{t1}), chan[{c0}:{c1})")
    fig.tight_layout()


def plot_waterfall(
    vis: np.ndarray,
    i: int,
    j: int,
    t_range: Optional[Tuple[int,int]] = None,
    chan_range: Optional[Tuple[int,int]] = None,
    quantity: str = 'amp',  # 'amp' | 'phase' | 'real' | 'imag'
    title: str = "",
):
    """Time–frequency waterfall for a single baseline (i,j)."""
    v = vis
    nt, nfreq, nin, _ = v.shape

    chan0 = 1024
    freq0 = 375e6
    chan_width_hz = 250e6 / 4096 / 2.0
    total_chans = abs(chan_range[-1] - chan_range[0])
    freq_axis_mhz = (freq0 + (chan0 + np.arange(total_chans)) * chan_width_hz) / 1e6
    
    if t_range is None:
        t0, t1 = 0, nt
    else:
        t0, t1 = max(0, t_range[0]), min(nt, t_range[1])

    if chan_range is None:
        c0, c1 = 0, nfreq
    else:
        c0, c1 = max(0, chan_range[0]), min(nfreq, chan_range[1])

    sub = v[t0:t1, c0:c1, i, j]

    if quantity == 'amp':
        Z = np.abs(sub)
        cbar_label = '|V|'
        cmap = 'viridis'
        Z = np.log10(Z)*10
    elif quantity == 'phase':
        Z = np.angle(sub)
        cbar_label = 'Phase [rad]'
        cmap = 'twilight'
    elif quantity == 'real':
        Z = sub.real
        cbar_label = 'Re(V)'
        cmap = 'coolwarm'
    elif quantity == 'imag':
        Z = sub.imag
        cbar_label = 'Im(V)'
        cmap = 'coolwarm'
    else:
        raise ValueError("quantity must be one of amp|phase|real|imag")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 2*4.6))

    # First subplot
    im = ax1.imshow(Z[::-1], origin='lower', aspect='auto', cmap=cmap,
                extent=[freq_axis_mhz.min(), freq_axis_mhz.max(), 0, 17 * 60])
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label=cbar_label)
    ax1.set_xlabel('Freq (MHz)')
    ax1.set_ylabel('Time (s)')
    ax1.set_title(title or f"Baseline ({i},{j}) t[{t0}:{t1}) chan[{c0}:{c1})")

    # Second subplot
    ax2.plot(freq_axis_mhz[::-1], np.mean(Z, 0), c='k')
    ax2.set_title('Vis %d %d' % (i, j))
    ax2.set_xlabel('Freq (MHz)')
    ax2.set_ylabel('Amplitude (dB)')
    ax2.set_xlim(freq_axis_mhz.min(), freq_axis_mhz.max())

    fig.tight_layout()
    plt.show()

# --------------------- CLI ---------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('data_dir', type=Path, help='Directory with raw binary visibility files')
    p.add_argument('--glob', default='*', help='Glob to select files within data_dir (default: * )')
    p.add_argument('--nt', type=int, default=16384, help='Number of time integrations per file (default: 16384)')
    p.add_argument('--nin', type=int, default=4, help='Number of inputs (default: 4)')
    p.add_argument('--dtype', default='float32', choices=['float16','float32','float64','int16','int32','int64'], 
                    help='Scalar dtype per real/imag value (default: float32)')
    p.add_argument('--big-endian', action='store_true', help='Interpret data as big-endian (default little-endian)')
    p.add_argument('--no-mmap', action='store_true', help='Disable memmap and read fully into RAM')
    p.add_argument('--labels', nargs='*', default=None, help='Input labels for axes, e.g. --labels 0 9 10 11')

    # Ranges
    p.add_argument('--chan', nargs=2, type=int, metavar=('C0','C1'), help='Channel range [C0, C1) to average/plot')
    p.add_argument('--t', nargs=2, type=int, metavar=('T0','T1'), help='Time range [T0, T1) to average/plot')

    # Plot modes
    p.add_argument('--matrix', action='store_true', help='Plot avg 4×4 matrix (magnitude by default)')
    p.add_argument('--matrix-phase', action='store_true', help='Plot avg 4×4 matrix of phase (radians)')
    p.add_argument('--waterfall', nargs=2, type=int, metavar=('I','J'), help='Plot a time–freq waterfall for baseline (I,J)')
    p.add_argument('--quantity', default='amp', choices=['amp','phase','real','imag'], help='Quantity for waterfall (default: amp)')
    p.add_argument('--dry-run', action='store_true', help='Do not plot; just print inferred metadata per file')

    # Boolean argument to append multiple files to look at time dependence over hours
    p.add_argument('--append', action='store_true', help='Append multiple files to look at time dependence over hours')


    args = p.parse_args()
    
    files = sorted((args.data_dir).glob(args.glob))
    if not files:
        print(f"No files found under {args.data_dir} with glob '{args.glob}'.", file=sys.stderr)
        sys.exit(2)
    
    for jj,fp in enumerate(files):
        try:
            vis, nfreq = load_vis(fp, nt=args.nt, nin=args.nin, dtype=args.dtype, 
            big_endian=args.big_endian, mmap=not args.no_mmap)
        except Exception as e:
            print(f"[ERROR] {fp.name}: {e}", file=sys.stderr)
            continue

        if args.append:
            if jj == 0:
                vis_all = vis[:,:,args.waterfall[0],args.waterfall[1]]
                continue
            else:
                vis_all = np.concatenate([vis_all, vis[:,:,args.waterfall[0],args.waterfall[1]]], axis=0)
            print(f"Appended {vis.shape[0]} files")
            print(f"New shape: {vis.shape}")
            print(jj)
            if jj > len(files):
                print(f"Reached end of files, stopping appending")
                print(vis_all.shape)
                np.save('vis_all.npy', vis_all)
                break

        nt, nf, ni, nj = vis.shape
        assert ni == nj == args.nin
        print(f"{fp.name}: nt={nt}, nfreq={nf}, nin={ni}, dtype={args.dtype}, endian={'big' if args.big_endian else 'little'} | size={fp.stat().st_size} B")

        # Quick stats: avg autos
        autos = np.mean(np.abs(np.diagonal(vis, axis1=2, axis2=3)), axis=(0,1))  # (nin,)
        print("  mean |auto| per input:", " ".join(f"{a:.3g}" for a in autos))

        if args.dry_run:
            continue

        t_range = tuple(args.t) if args.t else None
        c_range = tuple(args.chan) if args.chan else None

        title_base = f"{fp.name}"

        if args.matrix or args.matrix_phase:
            stat = 'phase' if args.matrix_phase else 'amp'
            plot_matrix(vis, t_range=t_range, chan_range=c_range, labels=args.labels, stat=stat, title=title_base + f" [{stat}]")

        if args.waterfall is not None:
            i, j = map(int, args.waterfall)
            plot_waterfall(vis, i, j, t_range=t_range, chan_range=c_range, quantity=args.quantity, title=title_base)

    if not args.dry_run:
        plt.show()


if __name__ == '__main__':
    main()
