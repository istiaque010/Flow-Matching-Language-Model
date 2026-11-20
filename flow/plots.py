"""Figures (Phases 7-15) -> results/figures/.

    python -m flow.plots

fig0 (toy 2D) is drawn by flow.toy2d. Colours: Okabe-Ito (colour-blind safe).
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import FIGURES, METRICS, load_json  # noqa: E402

C = {"flow": "#0072B2", "flow2": "#56B4E9", "ar": "#D55E00", "mdlm": "#009E73", "real": "#000000",
     "ref": "#999999", "abl": "#CC79A7", "proj": "#E69F00"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, name), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("wrote", name)


def read_log(run):
    p = os.path.join(METRICS, "train", run, "log.csv")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["mse_uncond"]]


def fig1_training():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    runs = [("main", "main (6,000 steps)", C["flow"], "-"), ("ref_s0", "reference seed 0", C["ref"], "-"),
            ("ref_s1", "reference seed 1", C["ref"], "--"), ("abl_notime", "no time input", C["ar"], "-"),
            ("abl_auxce", "+ token loss", C["mdlm"], "-"), ("abl_tpow2", "more small-t training", C["abl"], "-"),
            ("abl_len32", "length 32", C["proj"], "-")]
    for run, lab, col, ls in runs:
        rows = read_log(run)
        if not rows:
            continue
        s = [int(r["step"]) * 128 * 64 / 1e6 for r in rows]
        axes[0].plot(s, [float(r["mse_uncond"]) for r in rows], color=col, ls=ls, marker="o", ms=3, lw=1.6, label=lab)
        axes[1].plot(s, [float(r["acc_t0.1"]) for r in rows], color=col, ls=ls, marker="o", ms=3, lw=1.6)
    axes[0].set(xlabel="training tokens (millions)", ylabel="validation velocity MSE (no prompt)",
                title="Flow-matching loss")
    axes[1].set(xlabel="training tokens (millions)", ylabel="token accuracy of implied x1 at t = 0.1",
                title="How well the model guesses tokens from heavy noise")
    axes[0].legend(fontsize=8, frameon=False)
    for a in axes:
        a.grid(color="#eeeeee")
    save(fig, "fig1_training.png")


def fig2_decodability():
    d = load_json(METRICS, "diagnostics.json")["decodability"]
    t = [r["t"] for r in d]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(t, [r["acc_nearest"] for r in d], color=C["flow"], marker="o", lw=2, label="model, nearest-embedding decoder")
    ax.plot(t, [r["acc_projection"] for r in d], color=C["proj"], marker="s", lw=2, label="model, projection decoder")
    ax.plot(t, [r["acc_rescaled_xt"] for r in d], color=C["ref"], ls="--", marker=".", lw=1.5,
            label="no model: nearest token to x_t / t")
    ax.set(xlabel="time t (0 = pure noise, 1 = clean embeddings)", ylabel="share of true tokens recovered",
           title="Decoding the model's guess of the clean sequence", ylim=(0, 1.02))
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(color="#eeeeee")
    save(fig, "fig2_decodability.png")


def fig3_steps():
    runs = load_json(METRICS, "runs.json")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for solver, mk in (("euler", "o"), ("heun", "s")):
        for dec, col in (("nearest", C["flow"]), ("projection", C["proj"])):
            pts = sorted((v["nfe"], v["scorer_ppl"]) for k, v in runs.items()
                         if k.startswith(f"test_flow_{solver}") and k.endswith(dec))
            if pts:
                ax.plot(*zip(*pts), color=col, marker=mk, lw=1.8, ls="-" if solver == "euler" else "--",
                        label=f"flow, {solver}, {dec}")
    refs = [("test_ar5_topp0.95_s0", "AR 5.3M", C["ar"]), ("test_mdlm5_small-S32_s0", "MDLM 5.3M", C["mdlm"]),
            ("test_real", "real continuation", C["real"]), ("test_real_shuffled", "real words shuffled", C["ref"]),
            ("test_unigram", "random frequent words", C["abl"])]
    for k, lab, col in refs:
        if k in runs:
            ax.axhline(runs[k]["scorer_ppl"], color=col, lw=1, ls=":")
            ax.text(80, runs[k]["scorer_ppl"], lab, color=col, va="center", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xlim(6.5, 260)
    ax.set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])
    ax.set(xlabel="network calls per sample (NFE)", ylabel="scorer perplexity (log scale, lower = better)",
           title="Quality vs number of ODE steps (300 test prompts)")
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(0, 0.62))
    ax.grid(color="#eeeeee", which="both")
    save(fig, "fig3_steps.png")


def fig4_comparison():
    runs = load_json(METRICS, "runs.json")
    ch = load_json(METRICS, "flow_chosen.json")
    rows = [("real continuation", "test_real", C["real"]), ("AR 13M (teacher)", "test_ar13_topp0.95_s0", C["ar"]),
            ("AR 5.3M", "test_ar5_topp0.95_s0", C["ar"]), ("MDLM 13M", "test_mdlm13_rand-S32_s0", C["mdlm"]),
            ("MDLM 5.3M", "test_mdlm5_small-S32_s0", C["mdlm"]),
            (f"flow 5.2M ({ch['solver']} {ch['steps']})", f"test_flow_{ch['solver']}{ch['steps']}_{ch['decoder']}",
             C["flow"]),
            ("real words, shuffled", "test_real_shuffled", C["ref"]), ("random frequent words", "test_unigram", C["ref"])]
    rows = [r for r in rows if r[1] in runs]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    v = [runs[k]["scorer_ppl"] for _, k, _ in rows]
    ax.barh(range(len(rows)), v, color=[c for *_, c in rows], height=0.62)
    for i, x in enumerate(v):
        ax.text(x * 1.15, i, f"{x:,.0f}" if x >= 10 else f"{x:.1f}", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(right=max(v) * 8)
    ax.set(xlabel="scorer perplexity (log scale, lower = better)", title="Continuations of 300 test prompts")
    ax.grid(axis="x", color="#eeeeee")
    save(fig, "fig4_comparison.png")


def fig5_speed():
    b = load_json(METRICS, "benchmark.json")["systems"]
    runs = load_json(METRICS, "runs.json")
    match = {"flow_euler8": "test_flow_euler8_nearest", "flow_euler16": "test_flow_euler16_nearest",
             "flow_euler32": "test_flow_euler32_nearest", "flow_euler64": "test_flow_euler64_nearest",
             "flow_heun16": "test_flow_heun16_nearest", "flow_heun32": "test_flow_heun32_nearest",
             "ar5_topp0.95": "test_ar5_topp0.95_s0", "ar13_topp0.95": "test_ar13_topp0.95_s0",
             "mdlm5_small-S32": "test_mdlm5_small-S32_s0", "mdlm5_small-S8": "test_mdlm5_small-S8_s0",
             "mdlm13_rand-S32": "test_mdlm13_rand-S32_s0"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    names = [k for k in match if k in b]
    col = [C["flow"] if k.startswith("flow") else C["ar"] if k.startswith("ar") else C["mdlm"] for k in names]
    lat = [b[k]["latency_median_s"] * 1000 for k in names]
    axes[0].barh(range(len(names)), lat, color=col, height=0.6)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].invert_yaxis()
    for i, x in enumerate(lat):
        axes[0].text(x, i, f" {x:.0f}", va="center", fontsize=8)
    axes[0].set(xlabel="milliseconds per continuation (batch 1, median)", title="Latency, 48 new tokens, CPU")
    for k, c in zip(names, col):
        if match[k] in runs:
            axes[1].scatter(b[k]["latency_median_s"] * 1000, runs[match[k]]["scorer_ppl"], color=c, s=40, zorder=3)
            if k.startswith("flow_heun"):
                continue        # same latency as the Euler run with equal NFE; labelled as a group below
            lab = k.replace("flow_euler", "flow NFE ").replace("_topp0.95", "").replace("_small-", " ").replace("_rand-", " ")
            axes[1].annotate(lab, (b[k]["latency_median_s"] * 1000, runs[match[k]]["scorer_ppl"]), fontsize=7,
                             xytext=(3, -10 if k.startswith("flow") else 3), textcoords="offset points")
    axes[1].set_yscale("log")
    axes[1].set_xscale("log")
    axes[1].set(xlabel="latency, ms (log)", ylabel="scorer perplexity (log)", title="Quality vs speed (lower-left = better)")
    for a in axes:
        a.grid(color="#eeeeee")
    save(fig, "fig5_speed.png")


def fig6_ablations():
    comp = load_json(METRICS, "comparisons.json")["ablations"]
    if not comp:
        return
    lab = {"abl_ref_s1_test": "reference, other seed", "abl_abl_notime_test": "no time input",
           "abl_abl_auxce_test": "+ token cross-entropy", "abl_abl_pca128_test": "128-d PCA space",
           "abl_abl_tpow2_test": "more training at small t", "abl_main_auxce_test": "+ token CE, 6,000 steps",
           "len32 vs ref (8-token prompt, 24 tokens)": "length 32 (8+24 task)"}
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(comp) + 1.3))
    for i, c in enumerate(comp):
        d, ci = c["nll_diff"], c["nll_ci95"]
        ax.errorbar(d, i, xerr=[[d - ci[0]], [ci[1] - d]], fmt="o", capsize=3,
                    color=C["abl"] if c.get("counts") else "#555555")
        ax.text(ci[1] + 0.02, i, f"ppl {c['ppl_a']:,.0f}", va="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(comp)), [lab.get(c["label"], c["label"]) for c in comp])
    ax.invert_yaxis()
    ax.set(xlabel="scorer NLL difference vs reference (2,000 steps, seed 0); < 0 = better",
           title="Ablations (95% bootstrap intervals; coloured = counts by the pre-registered rule)")
    ax.grid(axis="x", color="#eeeeee")
    save(fig, "fig6_ablations.png")


def fig7_inside():
    d = load_json(METRICS, "diagnostics.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    dt = d["decision_time"]
    axes[0].plot([r["t"] for r in dt], [r["share_equal_final"] for r in dt], color=C["flow"], lw=2)
    axes[0].set(xlabel="ODE time t", ylabel="share of tokens already equal to the final token",
                title="When does each token get decided?", ylim=(0, 1.02))
    pi = d["prompt_influence"]
    labs = ["same noise:\nprompt vs no prompt", "same noise:\nprompt vs other prompt", "same prompt:\ndifferent noise"]
    vals = [pi["same_noise_prompt_vs_no_prompt"], pi["same_noise_prompt_vs_other_prompt"],
            pi["same_prompt_different_noise"]]
    axes[1].bar(range(3), vals, color=[C["flow"], C["flow2"], C["ref"]], width=0.6)
    for i, x in enumerate(vals):
        axes[1].text(i, x + 0.01, f"{x:.0%}", ha="center", fontsize=9)
    axes[1].set_xticks(range(3), labs, fontsize=8)
    axes[1].set(ylabel="share of identical continuation tokens", title="What decides the words: noise or prompt?",
                ylim=(0, 1))
    for a in axes:
        a.grid(axis="y", color="#eeeeee")
    save(fig, "fig7_inside.png")


def fig8_errors():
    e = load_json(METRICS, "errors_auto.json")
    order = [("real", "real continuation", C["real"]), ("ar13", "AR 13M", C["ar"]), ("ar5", "AR 5.3M", C["ar"]),
             ("mdlm13", "MDLM 13M", C["mdlm"]), ("mdlm5", "MDLM 5.3M", C["mdlm"]), ("flow", "flow 5.2M", C["flow"]),
             ("real_shuffled", "real words shuffled", C["ref"]), ("unigram", "random frequent words", C["ref"])]
    order = [o for o in order if o[0] in e]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    v = [e[k]["bigram_seen"] for k, *_ in order]
    ax.barh(range(len(order)), v, color=[c for *_, c in order], height=0.6)
    for i, x in enumerate(v):
        ax.text(x + 0.01, i, f"{x:.0%}", va="center", fontsize=8)
    ax.set_yticks(range(len(order)), [o[1] for o in order])
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.1), xlabel="share of adjacent token pairs ever seen in 20M training tokens",
           title="Local syntax: do neighbouring tokens fit together?")
    ax.grid(axis="x", color="#eeeeee")
    save(fig, "fig8_errors.png")


def main():
    for f in (fig1_training, fig2_decodability, fig3_steps, fig4_comparison, fig5_speed, fig6_ablations, fig7_inside,
              fig8_errors):
        try:
            f()
        except FileNotFoundError as ex:
            print("skip", f.__name__, ex)


if __name__ == "__main__":
    main()
