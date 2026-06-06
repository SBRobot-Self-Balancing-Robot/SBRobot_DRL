"""
plot.py — build every figure used in the Experimental Analysis chapter.

It consumes two kinds of data:

  1. Per-step *evaluation* metrics produced by ``test.py --metrics-csv`` (pitch
     oscillation about the equilibrium, forward-velocity tracking error,
     yaw-rate / heading tracking error, and the shaped reward with its
     components). One CSV per evaluated policy / setting; several may be
     dropped into the same folder so that policies, curriculum phases and
     domain-randomization settings can be compared side by side.

  2. The training-time ``rewards.csv`` written by ``train.py`` (mean per-step
     reward across the parallel workers), used for the reward-convergence
     analysis.

Usage
-----
    # collect metrics first (examples)
    python test.py --path PPO_run --metrics-csv metrics/ppo.csv --sweep-phases
    python test.py --path PPO_run --metrics-csv metrics/ppo_rand.csv \
                   --randomization-scale 1.0 --tag PPO-DR --sweep-phases
    python test.py --path SAC_run --metrics-csv metrics/sac.csv --tag SAC \
                   --sweep-phases

    # then build the figures
    python plot.py --input metrics --outdir docs/plots \
                   --rewards-csv policies/PPO_run/rewards.csv

Every figure is written both as a vector ``.pdf`` (for LaTeX inclusion) and a
``.png`` (for quick preview).
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
#  Styling
# ──────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

PHASE_RANGES = {0: "±0.10", 1: "±0.20", 2: "±0.35", 3: "±0.50"}
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  → {os.path.join(outdir, name)}.pdf / .png")


# ──────────────────────────────────────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_metrics(input_path):
    """Load and concatenate every metrics CSV in ``input_path`` (file or dir)."""
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.csv")))
    else:
        files = [input_path]
    if not files:
        raise FileNotFoundError(f"No metrics CSV found in {input_path!r}")

    required = {"phase", "episode", "pitch_deg", "vel_err", "yaw_err"}
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if not required.issubset(df.columns):
            print(f"  skipping {f} (not a metrics CSV)")
            continue
        if "tag" not in df.columns or df["tag"].isna().all():
            df["tag"] = os.path.splitext(os.path.basename(f))[0]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No valid metrics CSV (with columns {sorted(required)}) in {input_path!r}")
    data = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(data)} rows from {len(files)} file(s); "
          f"tags = {sorted(data['tag'].unique())}; "
          f"phases = {sorted(data['phase'].unique())}")
    return data


# ──────────────────────────────────────────────────────────────────────────────
#  Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

def _rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def summarize(data, outdir):
    """Aggregate RMSE / mean-|error| / survival per (tag, phase) and dump a CSV."""
    rows = []
    ep_len = (data.groupby(["tag", "phase", "episode"]).size()
              .rename("len").reset_index())
    survived = (data.groupby(["tag", "phase", "episode"])["terminated"].max()
                .rename("fell").reset_index())
    surv = survived.groupby(["tag", "phase"])["fell"].apply(
        lambda s: 100.0 * (1.0 - s.mean()))
    mean_len = ep_len.groupby(["tag", "phase"])["len"].mean()

    for (tag, phase), g in data.groupby(["tag", "phase"]):
        rows.append({
            "tag": tag,
            "phase": phase,
            "pitch_rmse_deg": _rmse(g["pitch_deg"]),
            "pitch_std_deg": float(g["pitch_deg"].std()),
            "vel_rmse": _rmse(g["vel_err"]),
            "vel_mae": float(g["vel_err"].abs().mean()),
            "yaw_rmse": _rmse(g["yaw_err"]),
            "yaw_mae": float(g["yaw_err"].abs().mean()),
            "reward_mean": float(g["reward"].mean()),
            "survival_pct": float(surv.loc[(tag, phase)]),
            "mean_ep_len": float(mean_len.loc[(tag, phase)]),
        })
    summary = pd.DataFrame(rows).sort_values(["tag", "phase"])
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "metrics_summary.csv")
    summary.to_csv(path, index=False)
    print(f"\nSummary written to {path}\n")
    print(summary.to_string(index=False))
    return summary


# ──────────────────────────────────────────────────────────────────────────────
#  Figures
# ──────────────────────────────────────────────────────────────────────────────

def plot_pitch_oscillation(data, outdir):
    """Pitch (oscillation about equilibrium) histogram, one curve per tag."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (tag, g) in enumerate(data.groupby("tag")):
        c = PALETTE[i % len(PALETTE)]
        ax.hist(g["pitch_deg"], bins=80, density=True, histtype="step",
                lw=1.8, color=c,
                label=f"{tag}  (RMSE {_rmse(g['pitch_deg']):.2f}°)")
    ax.axvline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Pitch angle about equilibrium [deg]")
    ax.set_ylabel("Probability density")
    ax.set_title("Balance: distribution of pitch oscillation")
    ax.legend()
    _save(fig, outdir, "pitch_oscillation_hist")


def plot_tracking_timeseries(data, outdir):
    """Velocity and yaw-rate tracking for one representative episode."""
    tag = sorted(data["tag"].unique())[0]
    phase = int(data["phase"].max())
    sub = data[(data["tag"] == tag) & (data["phase"] == phase)]
    if sub.empty:
        return
    # Pick the longest episode for a representative trace.
    ep = sub.groupby("episode").size().idxmax()
    ep_df = sub[sub["episode"] == ep].sort_values("t")

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    axes[0].plot(ep_df["t"], ep_df["target_vel"], color="k", ls="--",
                 lw=1.6, label="reference")
    axes[0].plot(ep_df["t"], ep_df["meas_vel"], color=PALETTE[0],
                 lw=1.6, label="measured")
    axes[0].set_ylabel("Forward velocity [m/s]")
    axes[0].set_title(f"Reference tracking — {tag}, phase {phase} "
                      f"({PHASE_RANGES.get(phase, '?')} m/s)")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].plot(ep_df["t"], ep_df["target_yaw"], color="k", ls="--",
                 lw=1.6, label="reference")
    axes[1].plot(ep_df["t"], ep_df["meas_yaw"], color=PALETTE[2],
                 lw=1.6, label="measured")
    axes[1].set_ylabel("Yaw rate [rad/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(loc="upper right", ncol=2)
    _save(fig, outdir, "tracking_timeseries")


def _grouped_box(ax, data, value, group_a, group_b, ylabel, title):
    """Boxplot of ``value`` grouped by ``group_a`` (x) and ``group_b`` (hue)."""
    a_vals = sorted(data[group_a].unique())
    b_vals = sorted(data[group_b].unique())
    width = 0.8 / max(len(b_vals), 1)
    for j, b in enumerate(b_vals):
        positions, samples = [], []
        for i, a in enumerate(a_vals):
            sel = data[(data[group_a] == a) & (data[group_b] == b)][value]
            if len(sel):
                positions.append(i + (j - (len(b_vals) - 1) / 2) * width)
                samples.append(sel.values)
        if samples:
            bp = ax.boxplot(samples, positions=positions, widths=width * 0.9,
                            patch_artist=True, showfliers=False,
                            manage_ticks=False)
            c = PALETTE[j % len(PALETTE)]
            for box in bp["boxes"]:
                box.set(facecolor=c, alpha=0.55, edgecolor=c)
            for med in bp["medians"]:
                med.set(color="k", lw=1.2)
            ax.plot([], [], color=c, lw=6, alpha=0.55, label=str(b))
    ax.set_xticks(range(len(a_vals)))
    ax.set_xticklabels([PHASE_RANGES.get(a, str(a)) for a in a_vals])
    ax.set_xlabel("Curriculum phase speed range [m/s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=group_b)


def plot_tracking_error_by_phase(data, outdir):
    """Velocity and yaw error boxplots across curriculum phases, hue = tag."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    _grouped_box(axes[0], data, "vel_err", "phase", "tag",
                 "Velocity error [m/s]",
                 "Velocity tracking error vs curriculum phase")
    _grouped_box(axes[1], data, "yaw_err", "phase", "tag",
                 "Yaw-rate error [rad/s]",
                 "Heading (yaw-rate) error vs curriculum phase")
    _save(fig, outdir, "tracking_error_by_phase")


def plot_reward_components(data, outdir):
    """Mean reward component (balance / velocity / yaw) per phase, first tag."""
    tag = sorted(data["tag"].unique())[0]
    sub = data[data["tag"] == tag]
    phases = sorted(sub["phase"].unique())
    comps = ["r_balance", "r_velocity", "r_yaw"]
    labels = ["balance", "velocity", "yaw"]
    means = {c: [sub[sub["phase"] == p][c].mean() for p in phases] for c in comps}

    x = np.arange(len(phases))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for k, (c, lab) in enumerate(zip(comps, labels)):
        ax.bar(x + (k - 1) * w, means[c], width=w, label=lab,
               color=PALETTE[k], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_RANGES.get(p, str(p)) for p in phases])
    ax.set_xlabel("Curriculum phase speed range [m/s]")
    ax.set_ylabel("Mean component value $\\in[0,1]$")
    ax.set_title(f"Reward component breakdown — {tag}")
    ax.legend(title="component")
    _save(fig, outdir, "reward_components_by_phase")


def plot_policy_comparison(summary, outdir):
    """Bar chart comparing policies (tags) on the key error / survival metrics."""
    # Aggregate over phases (mean) so each tag gets one bar per metric.
    agg = summary.groupby("tag").agg(
        pitch_rmse_deg=("pitch_rmse_deg", "mean"),
        vel_rmse=("vel_rmse", "mean"),
        yaw_rmse=("yaw_rmse", "mean"),
        survival_pct=("survival_pct", "mean"),
    ).reset_index()
    tags = agg["tag"].tolist()
    if len(tags) < 2:
        print("  (skipping policy_comparison: need ≥2 tags)")
        return

    metrics = [
        ("pitch_rmse_deg", "Pitch RMSE [deg]"),
        ("vel_rmse", "Velocity RMSE [m/s]"),
        ("yaw_rmse", "Yaw RMSE [rad/s]"),
        ("survival_pct", "Survival [%]"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(tags, agg[col], color=PALETTE[:len(tags)], alpha=0.85)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="x")
    fig.suptitle("Policy comparison (mean over curriculum phases)", y=1.02)
    _save(fig, outdir, "policy_comparison")


def plot_domain_randomization(summary, outdir, nominal_tag, dr_tag):
    """Ablation: nominal vs domain-randomized evaluation, per phase."""
    nom = summary[summary["tag"] == nominal_tag].set_index("phase")
    dr = summary[summary["tag"] == dr_tag].set_index("phase")
    if nom.empty or dr.empty:
        print("  (skipping domain_randomization: tags not found)")
        return
    phases = sorted(set(nom.index) & set(dr.index))
    x = np.arange(len(phases))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, col, ylabel in (
        (axes[0], "vel_rmse", "Velocity RMSE [m/s]"),
        (axes[1], "survival_pct", "Survival [%]"),
    ):
        ax.bar(x - w / 2, [nom.loc[p, col] for p in phases], width=w,
               label=nominal_tag, color=PALETTE[0], alpha=0.85)
        ax.bar(x + w / 2, [dr.loc[p, col] for p in phases], width=w,
               label=dr_tag, color=PALETTE[1], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([PHASE_RANGES.get(p, str(p)) for p in phases])
        ax.set_xlabel("Curriculum phase speed range [m/s]")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle("Domain-randomization ablation (nominal vs randomized robot)",
                 y=1.02)
    _save(fig, outdir, "domain_randomization_ablation")


def plot_reward_convergence(rewards_csv, outdir, smooth=200):
    """Training reward curve with a smoothed trend (convergence diagnosis)."""
    df = pd.read_csv(rewards_csv)
    col = "Reward" if "Reward" in df.columns else df.columns[1]
    x = df["Training Steps"] if "Training Steps" in df.columns else np.arange(len(df))
    raw = df[col]
    trend = raw.rolling(window=smooth, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(x, raw, color=PALETTE[0], alpha=0.25, lw=0.8, label="raw")
    ax.plot(x, trend, color=PALETTE[0], lw=2.0,
            label=f"moving avg (w={smooth})")
    # Linear fit over the final third to expose a residual upward slope.
    tail = slice(int(0.66 * len(df)), len(df))
    if (tail.stop - tail.start) > 2:
        xs = np.arange(tail.start, tail.stop)
        coef = np.polyfit(xs, raw.iloc[tail], 1)
        ax.plot(x.iloc[tail], np.polyval(coef, xs), color=PALETTE[1],
                lw=1.8, ls="--",
                label=f"final-third slope = {coef[0]:.2e}/step")
    ax.set_xlabel("Logged step index")
    ax.set_ylabel("Mean per-step reward")
    ax.set_title("Training reward — convergence diagnosis")
    ax.legend()
    _save(fig, outdir, "reward_convergence")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Build Experimental Analysis figures.")
    p.add_argument("--input", default="metrics",
                   help="Metrics CSV file or folder (default: metrics/).")
    p.add_argument("--outdir", default="docs/plots",
                   help="Output folder for figures (default: docs/plots).")
    p.add_argument("--rewards-csv", default=None,
                   help="Optional training rewards.csv for the convergence plot.")
    p.add_argument("--nominal-tag", default=None,
                   help="Tag of the nominal run for the DR ablation.")
    p.add_argument("--dr-tag", default=None,
                   help="Tag of the domain-randomized run for the DR ablation.")
    args = p.parse_args()

    have_metrics = os.path.exists(args.input)
    if have_metrics:
        data = load_metrics(args.input)
        summary = summarize(data, args.outdir)

        print("\nBuilding evaluation figures:")
        plot_pitch_oscillation(data, args.outdir)
        plot_tracking_timeseries(data, args.outdir)
        plot_tracking_error_by_phase(data, args.outdir)
        plot_reward_components(data, args.outdir)
        plot_policy_comparison(summary, args.outdir)

        tags = sorted(data["tag"].unique())
        nominal = args.nominal_tag or (tags[0] if tags else None)
        dr = args.dr_tag or next((t for t in tags if t != nominal), None)
        if nominal and dr and nominal != dr:
            plot_domain_randomization(summary, args.outdir, nominal, dr)
    else:
        print(f"Metrics input {args.input!r} not found — skipping evaluation figures.")

    if args.rewards_csv and os.path.exists(args.rewards_csv):
        print("\nBuilding reward-convergence figure:")
        plot_reward_convergence(args.rewards_csv, args.outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
