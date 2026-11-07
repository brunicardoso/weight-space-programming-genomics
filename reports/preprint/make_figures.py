"""Figures, tables and number macros for the preprint, generated from the analysis outputs only.

Run from the repo root:  pixi run python reports/preprint/make_figures.py

Reads (gitignored outputs; no new analyses are computed, only re-plotted / tabulated):
  runs/wsp/analysis/main/analysis.json, runs/wsp/analysis/main/repair.json,
  runs/wsp/coding/main/detection.json (structured fields only: never answer text or k-mer check windows),
  runs/baselines/*.json (metrics only).
Writes reports/preprint/{figures,tables}/ and tables/numbers.tex.
Leak guard: fails if any generated text contains a planted k-mer (or its reverse complement) or any run of >= 8
uppercase nucleotide letters.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

A = json.load(open(ROOT / "runs/wsp/analysis/main/analysis.json"))
R = json.load(open(ROOT / "runs/wsp/analysis/main/repair.json"))
DET = {r["run_id"]: r for r in json.load(open(ROOT / "runs/wsp/coding/main/detection.json"))["rows"]}
LE = {r["run_id"]: r for r in A["local_evaluation"]["per_run"]}
CLS = {r["run_id"]: r["class"] for r in A["per_protocol"]["per_run"]}
BASE = {Path(f).stem: json.load(open(f)) for f in sorted(glob.glob(str(ROOT / "runs/baselines/*.json")))}

# palette (reference categorical slots 1-3 + neutrals), light, print
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, LIGHT = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#c9c8c2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})
WRITTEN: list[Path] = []


def save(fig, name):
    for ext in ("pdf", "png"):
        p = FIG / f"{name}.{ext}"
        fig.savefig(p)
        WRITTEN.append(p)
    plt.close(fig)


def write(name, text):
    p = TAB / name
    p.write_text(text)
    WRITTEN.append(p)


def f3(x):
    return f"{x:.3f}".replace("-", "$-$")


INST_ORDER = ["kmer_s0", "kmer_s1", "kmer_s2", "positional"]
INST_LABEL = {"kmer_s0": "k-mer A", "kmer_s1": "k-mer B", "kmer_s2": "k-mer C", "positional": "poly-A"}

# ---------------------------------------------------------------- per-run join (listing only)
runs = []
for rid, d in DET.items():
    v = R[rid]
    runs.append(dict(rid=rid, cell=d["schedule_idx"], stage=int(d["stage"][-1]), variant=d["variant"],
                     target=d["target"], instance=d.get("instance"), outcome=d["outcome"], correct=d["correct"],
                     cls=CLS[rid], fp=LE[rid]["fp_calls"], local=LE[rid]["local_evaluations_known_part"],
                     local_complete=LE[rid]["local_evaluations"] is not None, launch=LE[rid]["launch"],
                     fwd_exec=LE[rid]["local_forward_executions"],
                     rep=v))
for r in runs:  # local forward computation ran, but no evaluation could be counted (inline / unknown loop counts)
    r["count_unknown"] = r["fwd_exec"] > 0 and r["local"] == 0
runs.sort(key=lambda r: r["cell"])
assert len(runs) == 56


def inst_of(r):
    t = r["target"]
    if t.startswith("kmer"):
        return "kmer_s" + t[-1]
    return r["variant"]


# ---------------------------------------------------------------- Figure 1: design schematic
def fig_design():
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    def box(x, y, w, h, title, body, edge=MUTED, fill="#f6f6f4"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", lw=0.8, ec=edge,
                                    fc=fill))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=6.8, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=5.6, color=INK2, linespacing=1.3)

    def arrow(x0, x1, y=15):
        ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=MUTED))

    box(0.5, 4, 15, 22, "Target models", "SmallCNN,\n58k params\nCTCF (GM12878)\n18 planted:\nk-mer, poly-A, GC\n"
        "10 clean controls\n(identity sealed)")
    arrow(16.3, 19)
    box(19.5, 4, 25, 22, "Blind session\n(Claude Code)", "\nseparate OS user; sandboxed shell,\nno network; CPU Python "
        "(numpy,\nscipy, torch); file tools in run dir\nweights + fp_query + wsp_submit\n"
        "A: detect (0 passes), locked; then\nB: repair (0 or 50 metered passes)", edge=BLUE)
    arrow(45.3, 48)
    box(48.5, 4, 15, 22, "Audit &\nreview", "\nrules checks\nforward_check\npath review\ncompliance class\n"
        "locked before\nunblinding")
    arrow(64.3, 67)
    box(67.5, 4, 14.5, 22, "Blind\ncoding", "\n2 fresh coders\nshuffled answers,\nrun ids removed\n7 categories\n"
        "kappa")
    arrow(82.8, 85.5)
    box(86, 4, 13.5, 22, "Unblind &\nanalyse", "\nscore vs\nsealed key\nrepair metrics\nH1-H4 (prereg.)\n"
        "sensitivity, PP", edge=ORANGE)
    ax.text(32, 1.2, "28 models x 2 stages = 56 sessions, random order", ha="center", fontsize=6.3, color=INK2)
    save(fig, "fig1_design")


# ---------------------------------------------------------------- Figure 2: detection
def fig_detection():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    groups = INST_ORDER + ["gc", "clean"]
    labels = [INST_LABEL[g] for g in INST_ORDER] + ["GC flank\n(explor.)", "clean\n(false pos.)"]
    xs = np.arange(len(groups))
    for i, g in enumerate(groups):
        if g == "clean":
            rs = [r for r in runs if r["variant"] == "clean"]
            k = sum(not r["correct"] for r in rs)
        elif g == "gc":
            rs = [r for r in runs if r["variant"] == "gc_shift"]
            k = sum(r["correct"] for r in rs)
        else:
            rs = [r for r in runs if r["variant"] in ("kmer", "positional") and inst_of(r) == g]
            k = sum(r["correct"] for r in rs)
        n = len(rs)
        col = MUTED if g in ("gc", "clean") else BLUE
        a1.bar(i, k / n, width=0.6, color=col, alpha=0.9 if g not in ("gc", "clean") else 0.55, zorder=2)
        a1.text(i, k / n + 0.03, f"{k}/{n}", ha="center", va="bottom", fontsize=7, color=INK)
    h1 = A["H1"]["D"]
    a1.axhline(A["H1"]["detection_rate"]["estimate"], color=ORANGE, lw=1, ls="--", zorder=1)
    a1.text(3.55, A["H1"]["detection_rate"]["estimate"] + 0.03,
            f"instance-weighted\ndetection {h1['estimate']:.3f}", ha="right", va="bottom", fontsize=6.3,
            color=INK2)
    a1.set_xticks(xs, labels, fontsize=6.2)
    a1.set_ylim(0, 1.05)
    a1.set_ylabel("share of phase-A answers correct\n(clean: share claiming a shortcut)")
    a1.yaxis.grid(True, zorder=0)
    a1.set_title("a  Detection by plant instance (both stages, budget 0)", loc="left", fontsize=8)

    conf = A["tables"]["confusion"]["pooled"]
    truths = ["kmer", "positional", "gc_shift", "clean"]
    cats = ["none", "kmer", "polyA_start", "polyA_other", "gc_flank", "gc_other", "other"]
    M = np.array([[conf.get(f"{t}|{c}", 0) for c in cats] for t in truths])
    a2.imshow(M, cmap="Blues", vmin=0, vmax=M.max() * 1.15, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j]:
                a2.text(j, i, M[i, j], ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > M.max() * 0.55 else INK)
    a2.set_xticks(range(len(cats)), [c.replace("_", "\n") for c in cats], fontsize=6.3)
    a2.set_yticks(range(len(truths)), ["k-mer", "poly-A", "GC flank", "clean"], fontsize=6.6)
    a2.set_xlabel("agreed code (both coders)")
    a2.set_title("b  Truth x agreed code (56 answers, kappa = 1.000)", loc="left", fontsize=8)
    for s in a2.spines.values():
        s.set_visible(False)
    a2.tick_params(length=0)
    fig.tight_layout()
    save(fig, "fig2_detection")


# ---------------------------------------------------------------- Figure 3: forest plot
def est(h):
    if "D" in h:
        h = h["D"]
    return h


def fig_forest():
    pp = A["per_protocol"]
    rows = [
        ("Primary (ITT, 56 cells)", A["H1"], A["H2"], A["H3"]),
        ("Pilot cells excluded", A["pilot_excluded_sensitivity"]["H1"], A["pilot_excluded_sensitivity"]["H2"],
         A["pilot_excluded_sensitivity"]["H3"]),
        ("Batch-launched only", A["launch_method_sensitivity"]["batch_only"]["H1"],
         A["launch_method_sensitivity"]["batch_only"]["H2"], A["launch_method_sensitivity"]["batch_only"]["H3"]),
        ("H1: disagreements excl.", A["H1_sensitivity"]["disagreements_excluded"], None, None),
        ("H1: clean GC claims excl.", A["H1_sensitivity"]["clean_gc_claims_excluded"], None, None),
        ("H1: both excluded", A["H1_sensitivity"]["both_excluded"], None, None),
        ("PP1: input-layer excl. (32)", pp["PP1"]["H1"], pp["PP1"]["H2"], pp["PP1"]["H3"]),
        ("PP2: any local fwd excl. (9)", pp["PP2"]["H1"], pp["PP2"]["H2"], pp["PP2"]["H3"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
    titles = ["H1  detection D", "H2  fix rate, stage 1", "H3  reduction in R(probe)"]
    ys = np.arange(len(rows))[::-1]
    for k, ax in enumerate(axes):
        ax.axvline(0, color=MUTED, lw=0.7)
        for y, row in zip(ys, rows):
            h = row[k + 1]
            if h is None:
                continue
            if "not_estimable" in h:
                ax.text(0.02, y, "not estimable", va="center", fontsize=6.3, color=INK2,
                        transform=ax.get_yaxis_transform())
                continue
            e = est(h)
            lo, hi = e["ci95"]
            sup = h.get("supported")
            col = BLUE if sup else (MUTED if sup is False else "#8fb8ea")
            ax.plot([lo, hi], [y, y], color=col, lw=1.4, solid_capstyle="round")
            ax.plot(e["estimate"], y, "o", ms=4.5, color=col, mec="white", mew=0.8, zorder=3)
            ax.text(1.02, y, f"{e['p_one_sided']:.3f}", va="center", ha="left", fontsize=5.8, color=INK2,
                    transform=ax.get_yaxis_transform(), clip_on=False)
        ax.set_title(titles[k], fontsize=8, loc="left")
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_yticks(ys, [r[0] for r in rows], fontsize=6.8)
    axes[0].set_xlim(-0.15, 1.05)
    axes[1].set_xlim(-0.15, 1.05)
    axes[2].set_xlim(-1.0, 1.4)
    for ax in axes:
        ax.text(1.02, 1.0, "p", transform=ax.transAxes, fontsize=5.8, color=INK2, va="bottom")
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", lw=1.4, label="supported (preregistered rule)"),
                        Line2D([], [], color=MUTED, marker="o", lw=1.4, label="not supported"),
                        Line2D([], [], color="#8fb8ea", marker="o", lw=1.4, label="exploratory, no decision")],
               loc="lower center", fontsize=6.2, bbox_to_anchor=(0.5, -0.06), ncol=3)
    fig.text(0.5, -0.1, "estimate with 95% bootstrap CI (10,000 resamples over plant instances); "
             "p one-sided; H1 alpha 0.025, H2/H3 Holm", ha="center", fontsize=6.3, color=INK2)
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig3_forest")


# ---------------------------------------------------------------- Figure 4: local computation per session
def fig_local():
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    marker = {"none": "o", "synthetic-only": "s", "input-layer": "^"}
    rng = np.random.default_rng(0)  # jitter only
    order = {"none": 0, "synthetic-only": 1, "input-layer": 2}
    for r in runs:
        x = (r["stage"] - 1) * 4 + order[r["cls"]] + rng.uniform(-0.28, 0.28)
        y = 0.25 if r["count_unknown"] else r["local"] + 1
        if r["correct"] and r["variant"] != "clean":
            fc, ec, z = BLUE, BLUE, 4
        elif r["outcome"] == "partial":
            fc, ec, z = "white", ORANGE, 3
        else:
            fc, ec, z = "white", MUTED, 2
        ax.scatter(x, y, marker=marker[r["cls"]], s=26, facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=z)
    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2, 4, 5, 6], ["none", "synthetic\nonly", "input\nlayer"] * 2, fontsize=6.6)
    for s, x in ((1, 1), (2, 5)):
        st = A["local_evaluation"][f"stage{s}"]
        ax.text(x, 1.6e5, f"Stage {s} (phase B budget {0 if s == 1 else 50})\n{st['with_local_forward']}/{st['n']}"
                f" sessions with local forward computation", ha="center", fontsize=6.0, color=INK)
    ax.axvline(3, color=GRID, lw=0.8)
    ax.set_ylim(0.15, 6e5)
    ax.axhspan(0.15, 0.45, color="#f3f2ee", lw=0, zorder=0)
    ax.set_yticks([0.25, 1, 10, 100, 1e3, 1e4, 1e5], ["unknown", "0", "10", "$10^2$", "$10^3$", "$10^4$",
                                                       "$10^5$"])
    ax.minorticks_off()
    ax.set_xlim(-0.6, 6.6)
    ax.set_ylabel("counted local evaluations (log)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", mfc=BLUE, mec=BLUE, label="correct detection"),
        Line2D([], [], ls="", marker="o", mfc="white", mec=ORANGE, label="partial"),
        Line2D([], [], ls="", marker="o", mfc="white", mec=MUTED, label="incorrect / clean / GC"),
    ], loc="upper left", bbox_to_anchor=(1.0, 0.75), fontsize=6.3)
    ax.set_xlabel("compliance class (locked before unblinding)")
    fig.tight_layout()
    save(fig, "fig4_local")


# ---------------------------------------------------------------- Figure 5: repair
def fig_repair():
    ed = [r for r in runs if not r["rep"]["unchanged"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.2, 1]))
    for r in ed:
        v = r["rep"]["verdict"]
        p, rl = v["probe_fpr"], v["reliance"]
        x0, y0 = p["planted"] / p["clean_upper_ci_max"], rl["planted"] / rl["clean_upper_ci_max"]
        x1, y1 = p["value"] / p["clean_upper_ci_max"], rl["value"] / rl["clean_upper_ci_max"]
        col = BLUE if r["variant"] == "kmer" else AQUA
        a1.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=LIGHT,
                                                            shrinkA=2, shrinkB=3))
        a1.scatter(x1, y1, s=34 if v["fixed"] else 20, color=col, edgecolor=INK if v["fixed"] else "white",
                   lw=1.0 if v["fixed"] else 0.6, zorder=3, marker="o" if r["stage"] == 1 else "D")
        if v["fixed"]:
            a1.annotate(f"cell {r['cell']}" + (" (strict)" if v["fixed_strict"] else ""), (x1, y1),
                        (1.35, y1 * 0.8), fontsize=6.2, color=INK,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
        a1.scatter(x0, y0, s=8, color=LIGHT, zorder=2)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.axvline(1, color=ORANGE, lw=0.8, ls="--")
    a1.axhline(1, color=ORANGE, lw=0.8, ls="--")
    a1.fill_between([0.3, 1], 0.01, 1, color=ORANGE, alpha=0.07, lw=0)
    a1.set_xlim(0.6, 12)
    a1.set_ylim(0.04, 300)
    a1.set_xlabel("probe FPR / clean bar")
    a1.set_ylabel("reliance / clean bar")
    a1.text(0.62, 0.045, "pass both", fontsize=6.3, color=ORANGE)
    a1.legend(handles=[Line2D([], [], ls="", marker="o", color=BLUE, label="k-mer"),
                       Line2D([], [], ls="", marker="o", color=AQUA, label="poly-A"),
                       Line2D([], [], ls="", marker="o", color=MUTED, mfc="white", label="stage 1 (0 passes)"),
                       Line2D([], [], ls="", marker="D", color=MUTED, mfc="white", label="stage 2 (50 passes)"),
                       Line2D([], [], ls="", marker="o", color=LIGHT, ms=3, label="planted (before edit)")],
               fontsize=5.9, loc="upper left", ncol=1)
    a1.set_title("a  The 12 edited runs: planted -> edited", loc="left", fontsize=8)

    # b: AUPRC drop, WSP edits vs M4 interp edits (descriptive)
    wsp = [r["rep"]["verdict"]["auprc_drop"] for r in ed]
    interp = [b["verdict"]["auprc_drop"] for m in BASE.values() for b in m["baselines"]
              if b["baseline"] == "interp" and b.get("verdict") and b["verdict"].get("auprc_drop") is not None
              and b["frac_weights_changed"] > 0 and not m["model"].startswith("gc")]
    rng = np.random.default_rng(1)
    a2.scatter(rng.uniform(-0.15, 0.15, len(wsp)), wsp, s=18, color=BLUE, edgecolor="white", lw=0.5, zorder=3)
    a2.scatter(1 + rng.uniform(-0.15, 0.15, len(interp)), interp, s=18, color=MUTED, edgecolor="white", lw=0.5,
               zorder=3)
    a2.axhline(0.01, color=ORANGE, lw=0.8, ls="--")
    a2.text(1.45, 0.0102, "limit 0.01", fontsize=6.3, color=ORANGE, ha="right", va="bottom")
    a2.axhline(0, color=GRID, lw=0.8)
    a2.set_xticks([0, 1], [f"WSP edits\n(n={len(wsp)})", f"M4 interp edits\nk-mer, poly-A (n={len(interp)})"],
                  fontsize=6.6)
    a2.set_xlim(-0.5, 1.5)
    a2.set_ylabel("clean test AUPRC drop vs planted")
    a2.set_title("b  Harm (descriptive; different data, see text)", loc="left", fontsize=8)
    a2.yaxis.grid(True)
    a2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig5_repair")
    return len(interp)


# ---------------------------------------------------------------- tables
def tab_plants():
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"variant & spec (sequence withheld) & models & sessions & probe FPR (planted) & reliance (planted) \\",
             r"\midrule"]
    spec = {"kmer": r"random 10-mer, random position; 3 instances (A--C)",
            "positional": r"poly-A 12-mer at offset 0--8",
            "gc_shift": r"+10 pp GC outside central 200 bp"}
    for var in ("kmer", "positional", "gc_shift"):
        ms = {k: v for k, v in BASE.items() if k.startswith(var)}
        pf = [m["planted"]["probe_fpr"] for m in ms.values()]
        rl = [m["planted"]["reliance"] for m in ms.values()]
        lines.append(f"{var.replace('_', '-')} & {spec[var]} & {len(ms)} & {2 * len(ms)} & "
                     f"{min(pf):.2f}--{max(pf):.2f} & {min(rl):.1f}--{max(rl):.1f} \\\\")
    cl = [r for r in runs if r["variant"] == "clean"]
    cpf = [R[r["rid"]]["eval"]["fpr_clean_test"] for r in cl]
    lines.append(f"clean & no plant & 10 & 20 & -- & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t_plants.tex", "\n".join(lines) + "\n")


def tab_primary():
    h1, h2, h3, h4 = A["H1"], A["H2"], A["H3"], A["H4"]

    def ci(e):
        return f"{f3(e['estimate'])} [{f3(e['ci95'][0])}, {f3(e['ci95'][1])}]"

    lines = [r"\begin{tabular}{p{5.6cm}lrll}", r"\toprule",
             r"hypothesis & estimate [95\% CI] & one-sided $p$ & threshold & decision \\", r"\midrule",
             f"H1 detection, budget 0: $D$ = detection $-$ false positives & {ci(h1['D'])} & "
             f"{h1['D']['p_one_sided']:.4f} & 0.025 & \\textbf{{supported}} \\\\",
             f"H2 unconditional fix rate, stage 1 & {ci(h2)} & {h2['p_one_sided']:.4f} & "
             f"{h2['holm']['threshold']} (Holm) & not supported \\\\",
             f"H3 paired reduction in R(probe FPR), budget 0 $-$ 50 & {ci(h3)} & {h3['p_one_sided']:.4f} & "
             f"{h3['holm']['threshold']} (Holm) & not supported \\\\",
             f"H4 false-positive rate, clean models & {ci(h4['false_positive_rate'])} ({0}/{h4['n']}) & -- & "
             f"descriptive & -- \\\\",
             r"\bottomrule", r"\end{tabular}"]
    write("t_primary.tex", "\n".join(lines) + "\n")


def tab_confusion():
    cats = ["none", "kmer", "polyA_start", "polyA_other", "gc_flank", "gc_other", "other"]
    truths = [("kmer", "k-mer"), ("positional", "poly-A"), ("gc_shift", "GC flank"), ("clean", "clean")]
    head = " & ".join(c.replace("_", r"\_") for c in cats)
    lines = [r"\begin{tabular}{ll" + "r" * len(cats) + "rr}", r"\toprule",
             f"stage & truth & {head} & $n$ & correct \\\\", r"\midrule"]
    for st in ("stage1", "stage2", "pooled"):
        conf = A["tables"]["confusion"][st]
        for t, lab in truths:
            cnt = [conf.get(f"{t}|{c}", 0) for c in cats]
            rs = [r for r in runs if r["variant"] == t and (st == "pooled" or r["stage"] == int(st[-1]))]
            assert sum(cnt) == len(rs)
            corr = sum(r["correct"] for r in rs)
            name = {"stage1": "1", "stage2": "2", "pooled": "both"}[st] if t == "kmer" else ""
            lines.append(f"{name} & {lab} & " + " & ".join(str(c) if c else r"\textperiodcentered" for c in cnt)
                         + f" & {len(rs)} & {corr} \\\\")
        if st != "pooled":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t_confusion.tex", "\n".join(lines) + "\n")


def tab_repair():
    rep = {(x["variant"], x["budget_b"]): x for x in A["tables"]["repair"]}
    bl = {(x["variant"], x["method"], x["budget"]): x for x in A["tables"]["baselines"]["rows"]}
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"& & \multicolumn{3}{c}{WSP} & & \multicolumn{3}{c}{M4 baselines, fixed/total} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){7-9}",
             r"variant & budget & edited & fixed (strict) & mean R probe & & fine-tune & interp & random \\",
             r"\midrule"]
    for var in ("kmer", "positional", "gc_shift"):
        for b in (0, 50):
            x = rep[(var, b)]
            if b == 0:
                ft = it = "--"
                rnd = f"{bl[(var, 'random_vs_interp', 10)]['fixed']}/" + str(sum(
                    bl[(var, 'random_vs_interp', bb)]['total'] for bb in (10, 25, 50)))
            else:
                ft = f"{bl[(var, 'finetune', 50)]['fixed']}/{bl[(var, 'finetune', 50)]['total']}"
                it = f"{bl[(var, 'interp', 50)]['fixed']}/{bl[(var, 'interp', 50)]['total']}"
                rnd = f"{bl[(var, 'random_vs_interp', 50)]['fixed']}/{bl[(var, 'random_vs_interp', 50)]['total']}"
            lines.append(f"{var.replace('_', '-') if b == 0 else ''} & {b} & {x['n'] - x['unchanged']}/{x['n']} & "
                         f"{x['fixed']} ({x['fixed_strict']}) & {x['mean_R_probe']:.3f} & & {ft} & {it} & {rnd} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t_repair.tex", "\n".join(lines) + "\n")


def tab_edited():
    ed = [r for r in runs if not r["rep"]["unchanged"]]
    ed.sort(key=lambda r: (r["variant"], r["target"], r["stage"]))
    yn = lambda b: r"\checkmark" if b else r"$\times$"  # noqa: E731
    lines = [r"\begin{tabular}{rllrrrcrcrcccl}", r"\toprule",
             r"cell & model & st. & fp & cls. & \% w & probe FPR (bar) & & reliance (bar) & & $\Delta$AUPRC & & "
             r"fixed & strict \\", r"\midrule"]
    ab = {"none": "N", "synthetic-only": "S", "input-layer": "I"}
    for r in ed:
        v = r["rep"]["verdict"]
        p, rl = v["probe_fpr"], v["reliance"]
        lines.append(
            f"{r['cell']} & {r['target'].replace('_', r'\_')} & {r['stage']} & {r['fp']} & {ab[r['cls']]} & "
            f"{100 * r['rep']['frac_weights_changed']:.2f} & {p['value']:.3f} ({p['clean_upper_ci_max']:.3f}) & "
            f"{yn(p['pass'])} & {rl['value']:.3f} ({rl['clean_upper_ci_max']:.3f}) & {yn(rl['pass'])} & "
            f"{(round(v['auprc_drop'], 4) + 0.0):.4f}".replace('-', '$-$') + f" & {yn(v['auprc_pass'])} & "
            f"{r'\textbf{yes}' if v['fixed'] else 'no'} & {r'\textbf{yes}' if v['fixed_strict'] else 'no'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t_edited.tex", "\n".join(lines) + "\n")


def tab_sessions():
    lines = [r"\begin{longtable}{rrlllrrrl}",
             r"\caption{All 56 sessions in schedule order. Class: compliance class (locked before unblinding); "
             r"fwd exec.: code executions with local forward computation; local eval.: counted calls of session-written "
             r"forward functions (known part; ``+'': some loop counts unknown, so a lower bound; ``unknown'': forward "
             r"computation ran but none of it could be counted, e.g. a layer applied inline).}"
             r"\label{tab:sessions}\\", r"\toprule",
             r"cell & stage & model & launch & class & fp calls & fwd exec. & local eval. & phase A \\",
             r"\midrule", r"\endfirsthead",
             r"\multicolumn{9}{l}{\small Table \ref{tab:sessions} (continued)}\\", r"\toprule",
             r"cell & stage & model & launch & class & fp calls & fwd exec. & local eval. & phase A \\",
             r"\midrule", r"\endhead"]
    for r in runs:
        loc = "unknown" if r["count_unknown"] else f"{r['local']}" + ("" if r["local_complete"] else "+")
        lines.append(f"{r['cell']} & {r['stage']} & {r['target'].replace('_', r'\_')} & {r['launch']} & "
                     f"{r['cls']} & {r['fp']} & {r['fwd_exec']} & {loc} & {r['outcome']} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    write("t_sessions.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- number macros
def macros(n_interp):
    h1, h2, h3, pp = A["H1"], A["H2"], A["H3"], A["per_protocol"]
    le1, le2 = A["local_evaluation"]["stage1"], A["local_evaluation"]["stage2"]
    ed = [r for r in runs if not r["rep"]["unchanged"]]
    planted = [r for r in runs if r["variant"] != "clean"]
    fixed = [r for r in planted if r["rep"].get("verdict", {}).get("fixed")]
    correct = [r for r in runs if r["correct"] and r["variant"] in ("kmer", "positional")]
    s2fp = sorted(r["fp"] for r in runs if r["stage"] == 2)
    tr = A["exploratory"]["test_retest"]
    cal = A["exploratory"]["calibration"]
    m = {
        "HoneD": f3(h1["D"]["estimate"]), "HoneLo": f3(h1["D"]["ci95"][0]), "HoneHi": f3(h1["D"]["ci95"][1]),
        "HoneP": f"{h1['D']['p_one_sided']:.4f}",
        "HonePooled": f"{sum(r['correct'] for r in runs if r['variant'] in ('kmer', 'positional'))}",
        "HtwoEst": f3(h2["estimate"]), "HtwoLo": f3(h2["ci95"][0]), "HtwoHi": f3(h2["ci95"][1]),
        "HtwoP": f"{h2['p_one_sided']:.4f}",
        "HthreeEst": f3(h3["estimate"]), "HthreeLo": f3(h3["ci95"][0]), "HthreeHi": f3(h3["ci95"][1]),
        "HthreeP": f"{h3['p_one_sided']:.4f}",
        "HthreeFixDiff": f3(h3["fix_rate_difference"]["estimate"]),
        "PPoneD": f3(pp["PP1"]["H1"]["D"]["estimate"]), "PPoneLo": f3(pp["PP1"]["H1"]["D"]["ci95"][0]),
        "PPoneHi": f3(pp["PP1"]["H1"]["D"]["ci95"][1]), "PPoneP": f"{pp['PP1']['H1']['D']['p_one_sided']:.4f}",
        "PPtwoN": str(pp["PP2"]["n_cells"]), "PPoneN": str(pp["PP1"]["n_cells"]),
        "LocalOne": str(le1["with_local_forward"]), "LocalTwo": str(le2["with_local_forward"]),
        "LocalAll": str(le1["with_local_forward"] + le2["with_local_forward"]),
        "LocalMedOne": f"{le1['median_local_evaluations_known_part']:.0f}",
        "LocalMedTwo": f"{le2['median_local_evaluations_known_part']}",
        "FpMedTwo": f"{le2['median_fp_calls']}", "FpMinTwo": str(s2fp[0]), "FpMaxTwo": str(s2fp[-1]),
        "NEdited": str(len(ed)), "NFixed": str(len(fixed)), "NCorrect": str(len(correct)),
        "Kappa": f"{A['tables']['kappa']['kappa_valid_runs']:.3f}",
        "TRSame": str(tr["same_category"]), "TRKappa": f"{tr['kappa']:.2f}", "Brier": f"{cal['brier']:.3f}",
        "MaxAuprcDrop": f"{max(r['rep']['verdict']['auprc_drop'] for r in ed):.4f}",
        "NInterpEdits": str(n_interp),
        "NCountUnknown": str(sum(r["count_unknown"] for r in runs)),
        "CountUnknownCells": ", ".join(str(r["cell"]) for r in runs if r["count_unknown"]),
        "NIncomplete": str(sum(not r["local_complete"] for r in runs)),
    }
    for k in m:
        assert re.fullmatch(r"[A-Za-z]+", k), k
    write("numbers.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
    # consistency checks against the analysis outputs
    assert len(correct) == 7 and len(fixed) == 2 and len(ed) == 12
    assert all(r["cls"] != "none" for r in correct)
    return m


# ---------------------------------------------------------------- leak guard
def leak_guard(extra: list[Path]):
    kmers = set()
    comp = str.maketrans("ACGT", "TGCA")
    for f in glob.glob(str(ROOT / "sealed/plants/*.json")):
        d = json.load(open(f))
        if d.get("kmer"):
            kmers.add(d["kmer"].upper())
            kmers.add(d["kmer"].upper().translate(comp)[::-1])
    assert kmers, "no k-mers loaded for the guard"
    bad = []
    for p in [x for x in WRITTEN if x.suffix in (".tex",)] + extra:
        if not p.exists():
            continue
        t = p.read_text()
        if any(k in t.upper() for k in kmers) or re.search(r"[ACGT]{8,}", t):
            bad.append(str(p))
    # PDFs: check uncompressed text extraction is not available here; figures contain no sequence text by
    # construction (labels are fixed strings above), so check the label sources instead.
    assert not bad, f"LEAK GUARD FAILED: {bad}"
    print(f"leak guard: ok ({len(kmers)} sequences checked, {len(WRITTEN) + len(extra)} files)")


if __name__ == "__main__":
    fig_design()
    fig_detection()
    fig_forest()
    fig_local()
    n_interp = fig_repair()
    tab_plants()
    tab_primary()
    tab_confusion()
    tab_repair()
    tab_edited()
    tab_sessions()
    macros(n_interp)
    leak_guard([OUT / "main.tex", Path(__file__)])
    print("wrote", len(WRITTEN), "files")
