# Project 10 — Flow-Matching Language Model: phase-by-phase instructions

Source plan: `docs/Project_10_Flow_Matching_Language_Model_Guide.docx` (kept locally, not in the repository).

**Goal:** build a language model that does not pick tokens one by one (autoregressive, Projects 1–2) and does not
unmask tokens step by step (masked diffusion, Projects 3–6). Instead it works in a **continuous** space: every token
is a point (its embedding), a whole sequence is a cloud of 64 points, and a neural network learns a **velocity field**
that moves pure noise to sequences of token embeddings along straight lines (**flow matching**). An ODE solver follows
that field, and the end points are rounded to the nearest token. Then compare it with AR and diffusion models of the
same size and training budget.

**Research question:** can a small continuous flow-matching model write coherent text, and where exactly does the
continuous-to-discrete step succeed or fail compared with AR and masked diffusion LMs?

```
TinyStories openings (64 tokens) ──► frozen Project 1 token embeddings (384-d, standardised) = x1
noise x0 ~ N(0, I)  ──►  x_t = (1 - t) x0 + t x1   ──►  Transformer learns v(x_t, t) ≈ x1 - x0   (masked MSE)
sampling: x0 ──(Euler / Heun ODE, 8-64 network calls)──► x1_hat ──(nearest embedding)──► tokens
prompt: the first 16 positions are held at the prompt's embeddings during the whole ODE ──► 48 new tokens
compare on 300 test prompts with AR-5.3M, MDLM-5.3M (same size, same 49M tokens) and the 13.9M AR / MDLM models
```

## How to use this file
1. Work through the phases in order. Each has **Learn**, **Files**, **Run**, **Check** and **Result**.
2. Run everything from the `project10` folder with `.venv` active (Windows: `.venv\Scripts\activate`).
3. Status changes from ⬜ to ✅ when the phase is done and checked.

## Progress

| # | Phase | Status |
|---|---|---|
| 1 | Prerequisites and setup | ✅ |
| 2 | Dataset and baselines | ✅ |
| 3 | Choose a continuous representation | ✅ |
| 4 | Define the flow-matching path | ✅ |
| 5 | Implement the velocity Transformer | ✅ |
| 6 | Implement the flow-matching loss | ✅ |
| 7 | Train the flow model | ✅ |
| 8 | Build the ODE sampler | ✅ |
| 9 | Decode continuous samples into tokens | ✅ |
| 10 | Diagnose the continuous–discrete gap | ✅ |
| 11 | Add prompt-conditioned continuation | ✅ |
| 12 | Compare ODE solvers and step counts | ✅ |
| 13 | Compare with AR and diffusion LMs | ✅ |
| 14 | Ablations and error analysis | ✅ |
| 15 | Repository structure | ✅ |
| 16 | Final report and deliverables | ✅ |

## The answer in five lines
- **No.** At 5.2M parameters and 49M training tokens the flow LM writes **word salad**: scorer perplexity
  **117,771** versus 313 for MDLM-5.3M and 22.7 for AR-5.3M; blind AI review coherence **0.00 / 2** for all 20 samples.
- Its text sits **between shuffled real words (60,756) and random frequent words (155,007)**: it picks story words
  that fit the topic (better than random words) but puts them in no order (adjacent-pair test: 46%, same as random).
- The failure is located: about **75% of tokens are fixed by t = 0.2**, when the model can only guess 30–80% of
  tokens and each position decides almost on its own; the **starting noise decides the words, not the prompt**
  (74% of tokens identical with or without the prompt).
- Training 3× longer lowers the flow loss (0.696 → 0.643) but **not** the text quality (112k–126k → 115k–121k).
- It is **fast**: 68 ms per 48-token continuation at 32 Euler steps, faster than AR-5.3M (96 ms) and MDLM-5.3M (143 ms).

## Decisions for this PC and this project (senior-researcher choices, with reasons)

- **CPU only** (no GPU on this PC), fp32, 24 threads, as in Projects 1–9.
- **Continuous space = frozen Project 1 `small` token embeddings** (384-d), standardised with a frequency-weighted
  mean and one scalar scale so the data has mean 0 and variance 1, like the noise. Frozen embeddings cannot collapse
  (a learned space can shrink all tokens together to make the loss small). Project 1's `baseline` embeddings were
  rejected: they were trained on only 100k stories, and decoding by dot product recovered only 12% of tokens.
- **Same budget as the 5.3M baselines:** the flow model has 5.18M parameters and sees 6,000 × 128 × 64 = 49.2M
  tokens, the same as AR-5.3M (Project 2 `student_scratch`) and MDLM-5.3M (Project 3 `mdlm_small`).
- **Fixed 64-token windows = story openings.** Flow matching needs a fixed shape; the first 64 tokens of each story
  (99.98% of windows are full) give one clean, comparable task: continue an opening.
- **Straight-line (rectified) path, uniform t, masked MSE** — the simplest flow-matching recipe. Extras (token
  cross-entropy, other t sampling, PCA space, shorter windows, no time input) are ablations, not part of the main model.
- **Hand-written Euler and Heun solvers** instead of `torchdiffeq`: 30 lines, and every network call (NFE) is counted.
- **Independent scorer:** TinyStories-33M (Project 4's pinned download) scores every continuation. It is not one of
  the compared models, and all systems are scored on exactly the same 48-token slot after the same 16-token prompt.
- **Decoding settings chosen on validation (dev) prompts only,** baselines use the settings Projects 1/3/4 chose on
  their validation data. Hypotheses were written before the main model was trained (`configs/hypotheses.md`).

---
## Phase 1 — Prerequisites and setup

**Learn:** a *flow* is a rule that moves points over time. If you know the velocity v(x, t) at every point and time,
you can start from a random point at t = 0 and follow the arrows until t = 1 — that is solving an **ODE** (ordinary
differential equation), dx/dt = v(x, t). **Flow matching** teaches a network this velocity field so that following it
turns noise into data. It is the continuous cousin of diffusion: no masks, no token-by-token choices, every position
moves a little at every step.

**Files:** `requirements.txt`, `requirements.lock.txt`, `pytest.ini`, `flow/__init__.py` (paths, token ids,
JSON/JSONL helpers; makes Projects 1, 3 and 4 importable read-only), `scripts/env_report.py`, `flow/toy2d.py`

**Run:**
```
python -m venv .venv
.venv\Scripts\pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt
python scripts/env_report.py      # -> results/metrics/env.json
python -m flow.toy2d              # -> results/metrics/toy2d.json, results/figures/fig0_toy2d.png
```

**Check:** the toy problem must work before any text: a small MLP learns to move 2-D Gaussian noise onto 8 Gaussian
blobs on a circle (4,000 steps, 7.5 s). Quality = sliced Wasserstein distance (how far apart two point clouds are,
averaged over random 1-D projections; lower is better).

**Result:** Python 3.12.10, torch 2.14.1+cpu, transformers 5.22.2, 24 threads, 63.7 GB RAM, no CUDA. Hashes of every
file read from other projects are in `env.json` (embeddings `b4a9986a…611361`, scorer `41316d3b…dad302`, tokenizer
`158ade55…c0f530`, baselines ar5 `666d9813…`, mdlm5 `a551ab6e…`, ar13 `90b1de01…`, mdlm13 `29f3eb9d…`).

| toy sampler | network calls | sliced W1 | samples near a blob |
|---|---|---|---|
| noise itself | 0 | 1.759 | — |
| Euler 1 step | 1 | 2.437 | 0% (everything lands in the centre = the average of all blobs) |
| Euler 4 | 4 | 0.272 | 76% |
| Euler 8 | 8 | 0.175 | 88% |
| Euler 32 | 32 | 0.110 | 94% |
| Heun 4 | 8 | **0.087** | 91% |
| two real samples compared | — | 0.110 | — |

At 32 steps generated points are as close to the data as two real samples are to each other. **Keep the Euler-1
result in mind:** one big step lands on the *average* of all possible targets — the same averaging problem returns
for text in Phase 10.

---
## Phase 2 — Dataset and baselines

**Learn:** the data is a fixed-length window so every training example has the same shape (64 positions × 384
numbers). Splits are by story, taken from Project 1's tokenized data, so the test stories were never trained on.

**Files:** `flow/dataset.py` → `data/{train,val,dev,test,test_prompt}.npy`, `data_manifests/data.json`

**Run:** `python -m flow.dataset`

**Result:**

| split | openings | use |
|---|---|---|
| train | 1,772,187 (113.4M valid tokens) | flow training |
| val | 2,000 | validation MSE during training |
| dev | 100 (stories > 64 tokens) | choosing the decoding setting |
| test | 2,000 | diagnostics |
| test_prompt | 300 (stories > 64 tokens) | the comparison: first 16 tokens = prompt, 48 new tokens |

99.98% of training windows are full 64 tokens (an `<EOS>` almost never falls inside a window, so no system is asked to
end a story). Identical 64-token openings across splits: 0, except one train–test pair (counted, not removed).

**Baselines** (all read-only, loaded through Project 4's `bench`): **AR-5.3M** (Project 2 student from scratch, 49M
tokens), **MDLM-5.3M** (Project 3 `mdlm_small`, 49M tokens), and as stronger references **AR-13.9M** (Project 1
teacher) and **MDLM-13.9M** (Project 3 `mdlm_base`). Decoding: AR top-p 0.95; MDLM-5.3M 32 steps, linear schedule,
random order, top-k 50; MDLM-13.9M 32 steps, confidence order, top-k 20 (Projects 3/4 choices).

---
## Phase 3 — Choose a continuous representation

**Learn:** flow matching needs numbers, not token ids. Each token gets a fixed vector (its embedding); to turn a
vector back into a token, take the **nearest** embedding (L2 distance) or use a **projection** (multiply by all
embeddings and take the largest score, as a language model's output layer does). The space matters: tokens must be
far apart so that rounding is reliable, and the data should have the same scale as the noise.

**Files:** `flow/embeddings.py` (`TokenSpace`: encode / nearest / project / decode) → `data/embedding_stats.pt`,
`results/metrics/embeddings.json`, `tests/test_embeddings.py`

**Run:** `python -m flow.embeddings`

**Result:** raw embedding scale 0.050 → standardised with σ = 0.0407 and a frequency-weighted mean; the data then has
std **0.996**.

| check | 384-d (main) | 128-d PCA (ablation) |
|---|---|---|
| round trip over the 8,192-token vocabulary, nearest | 100% | 100% |
| round trip, projection | 77.1% | — |
| clean validation tokens, nearest / projection | 100% / 99.72% | 100% |
| clean + N(0, 1) noise (as large as the data!), nearest | 99.991% | 99.73% |
| median distance to the nearest *other* token (data-weighted) | **19.3** | 10.0 |

Tokens are so far apart in 384 dimensions that even noise as large as the signal hardly ever changes the nearest
token. **So rounding itself is not the problem**; any failure must come from where the model's samples end up.

---
## Phase 4 — Define the flow-matching path

**Learn:** pick a noise sample x0 and a data sample x1 and connect them with a straight line:
`x_t = (1 − t) x0 + t x1`. The velocity along that line is constant: `u = x1 − x0`. The network sees x_t and t and
must predict u. It cannot know which x0/x1 pair produced x_t, so the best it can do is the **average** velocity of all
pairs that pass through x_t — which, followed from noise, still carries the noise distribution to the data
distribution. From a velocity guess you can read off the model's current guess of the clean data:
`x1_hat = x_t + (1 − t) v`.

**Files:** `flow/flow_path.py` (`sample_t`, `interpolate`, `target_velocity`, `implied_x1`, `masked_mse`,
`build_xt`), `tests/test_flow_path.py`

**Result:** 4 unit tests: the path hits x0 at t = 0 and x1 at t = 1, a perfect velocity recovers x1 exactly, padded
positions are ignored by the loss, prompt positions are replaced by their clean embeddings.

---
## Phase 5 — Implement the velocity Transformer

**Learn:** the network is a **bidirectional** Transformer (every position sees every other, like MDLM, unlike AR).
Input: 64 noisy 384-d vectors, the time t, and a flag per position saying "this is a clean prompt token". The time
enters through **adaLN-Zero** (as in DiT): t is turned into sinusoidal features, and these set a scale, shift and
gate for every layer norm; the gates start at zero, so each block starts as the identity and training is stable.

**Files:** `flow/model.py` (`FlowConfig`, `Block`, `VelocityTransformer`), `tests/test_shapes.py`

**Result:** 4 layers, 4 heads, width 256, input/output projections 384 ↔ 256, **5,178,752 parameters** (AR-5.3M:
5.3M, MDLM-5.3M: 5.3M). 5 tests: output shape and the output reacts to t (after randomising the zero-initialised
weights), the no-time ablation ignores t, a fresh model predicts zero velocity (zero-init), padded keys are masked,
and the valid-token mask stops at `<EOS>`.

---
## Phase 6 — Implement the flow-matching loss

**Learn:** the loss is the mean squared error between predicted and true velocity, over valid (non-padding)
positions. To make one model do both free generation and continuation, half of each batch gets a **clean prompt
prefix** of 8–24 tokens: those positions are set to the true embeddings, flagged, and left out of the loss.

**Files:** `flow/train.py` (`loss_fn`, `aux_ce` for the ablation, `validate`)

**Check:** `python -m flow.train --overfit` drives the loss on one batch close to zero (done during development,
checkpoint deleted).

**Validation** (fixed noise seed, 8 fixed times): MSE without prompt and with a 16-token prompt, plus the token
accuracy of `x1_hat` at t = 0.05 / 0.1 / 0.2 / 0.3. (Accuracy at t ≥ 0.5 is near 100% even for an *untrained* model,
because x_t is already close to x1 there — so it was moved to small t, where it is informative.)

---
## Phase 7 — Train the flow model

**Learn:** a short **learning-rate pilot** on validation data, then one main run with the full budget.

**Files:** `configs/flow.yaml`, `scripts/lr_pilot.sh`, `scripts/train_all.sh`, `results/metrics/lr_pilot.json`,
`results/metrics/train/<run>/`

**Run:** `bash scripts/lr_pilot.sh` (3e-4, 1e-3, 3e-3; 6e-3 and 1.2e-2 were then run the same way because the best
value was at the edge), then `bash scripts/train_all.sh` (main + 7 ablation runs, about 2.5 hours).

**Learning-rate pilot** (1,500 steps, seed 100, validation MSE):

| peak lr | 3e-4 | 1e-3 | 3e-3 | **6e-3** | 1.2e-2 |
|---|---|---|---|---|---|
| val MSE | 0.949 | 0.818 | 0.744 | **0.733** | 1.704 (unstable) |

**Setup:** AdamW (betas 0.9/0.99, weight decay 0.01, none on embeddings and biases), 300 warm-up steps, cosine to
0.1×, gradient clip 1.0, batch 128 × 64 tokens, 6,000 steps.

**Result:** 32.4 minutes, 25,295 tokens/s, peak memory 1.8 GB, stable throughout; validation MSE **0.643** (with a
16-token prompt 0.666), still slowly falling at the end; accuracy of `x1_hat` 11.6% at t = 0.05, 30.5% at t = 0.1,
79.7% at t = 0.2, 92.0% at t = 0.3 ([figure 1](results/figures/fig1_training.png)).

---
## Phase 8 — Build the ODE sampler

**Learn:** to generate, start from noise at t = 0 and take steps of size h = 1/steps.
**Euler:** `x ← x + h·v(x, t)` (1 network call per step). **Heun:** take an Euler step, measure the velocity there
too, and use the average of the two (2 calls per step, more accurate per step). For prompts, the prompt positions are
reset to their clean embeddings after every step (and inside Heun's trial step).

**Files:** `flow/ode_sampler.py` (`integrate`), `tests/test_ode.py`

**Result:** 3 tests: for a constant field both solvers are exact and report the right NFE; for a time-dependent field
Heun's error shrinks like h² (second order) and is far below Euler's; clamped positions never move. NFE (number of function evaluations = network calls) is
returned by every call.

---
## Phase 9 — Decode continuous samples into tokens

**Learn:** the ODE ends at continuous vectors; each one is rounded to a token by the nearest embedding or the
projection head. The decoding setting (solver, steps, decoder) was chosen on the 100 dev prompts only.

**Files:** `flow/generate.py` (`flow_sample`), `flow/evaluate.py`, `flow/run_eval.py --stage dev`,
`results/metrics/flow_chosen.json`

**Run:** `python -m flow.run_eval --stage dev`

**Result** (dev, scorer perplexity, lower is better):

| | Euler 8 | 16 | 32 | 64 | Heun 4 | 8 | 16 | **32** |
|---|---|---|---|---|---|---|---|---|
| NFE | 8 | 16 | 32 | 64 | 8 | 16 | 32 | 64 |
| nearest | 153,769 | 134,872 | 115,823 | 111,114 | 272,992 | 118,148 | 110,523 | **104,901** |
| projection | 521,884 | 423,270 | 397,810 | 381,344 | 996,179 | 411,492 | 363,363 | 344,225 |

**Chosen: Heun, 32 steps (64 calls), nearest embedding.** The projection head is about 3.4× worse and produces 24–42%
words that never occur in the training vocabulary list (nearest: 2%), because it scores every token by dot product,
which can prefer tokens with long embedding vectors over the closest one (it recovers only 77% of the
vocabulary even from clean embeddings).

A sample (dev prompt → 48 flow tokens):
> *Once upon a time, there was a little girl named Lily. She loved to* → *day kitchen. bear things it round Mae saw One
> Mar mom dinosaur to. of not lo blinking swam the to She laughed so. Simon around. …*

Story words, almost no sentence structure.

---
## Phase 10 — Diagnose the continuous–discrete gap

**Learn:** the model has a low loss but writes word salad. Where does it go wrong? Four measurements.

**Files:** `flow/diagnose.py` → `results/metrics/diagnostics.json`

**Run:** `python -m flow.diagnose`

**1 Decodability over time** (500 test openings; does the model's guess `x1_hat` decode to the true token?) —
[figure 2](results/figures/fig2_decodability.png)

| t | 0.02 | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.4 | 0.5+ |
|---|---|---|---|---|---|---|---|---|
| model, nearest | 5.5% | 11.2% | 30.2% | 59.1% | 79.9% | 92.4% | 98.1% | 100% |
| no model: nearest token to x_t / t | 0.4% | 2.1% | 15.7% | 51.4% | 85.8% | 99.8% | 100% | 100% |

Below t = 0.15 the model knows more than the noisy input alone; from t = 0.2 to 0.4 its guess is even *worse* than
simply rescaling x_t (it blends neighbouring tokens), and its MSE stays at 0.47 for all t ≥ 0.3.

**2 Decision time** (100 test prompts, along the chosen Heun-32 path; share of continuation tokens whose current
decoded guess already equals the final token): 1% at t = 0, **61% at t = 0.12, 80% at t = 0.25**, 88% at t = 0.5
([figure 7, left](results/figures/fig7_inside.png)). Most tokens are fixed in the first fifth of the path, when the
model can only guess 30–80% of tokens correctly even when the *true* sequence is behind the noise.

**3 Prompt influence** (same starting noise, 200 prompts; share of identical continuation tokens):

| comparison | identical tokens |
|---|---|
| same noise, with the prompt vs without any prompt | **74%** |
| same noise, with the prompt vs with a *different* prompt | **73%** |
| same prompt, different noise | 2% |

The **noise decides the words**; the prompt changes only about a quarter of them ([figure 7, right](results/figures/fig7_inside.png)).

**4 End-point geometry** (distance from each generated vector to its nearest token embedding, and the margin to the
second nearest):

| vectors | distance to nearest token | margin |
|---|---|---|
| clean embeddings | 0.00 | 18.7 |
| clean + N(0, 1) noise | 19.6 | 7.4 |
| flow, Euler 8 → 64 steps | 14.8 → 13.6 | 5.5 → 7.6 |
| flow, Heun 32 (chosen) | 13.5 | 7.8 |

Generated vectors end **far from every token** (13.5, while tokens are 19 apart), i.e. in the empty space *between*
tokens; more steps move them only slightly closer.

**5 Frequent-token overuse:** share of tokens among the 100 most frequent training tokens — real 63.3%, flow 60.6%,
AR-5.3M 65.0%, MDLM-5.3M 70.0%, random frequent words 61.3%. Distinct tokens over 300 continuations: real 1,731, flow
1,315, AR 1,568, MDLM 847. **The flow model does not over-use frequent tokens** (MDLM does more).

**Diagnosis:** a velocity trained with MSE points toward the **average** of all sequences consistent with the current
x_t. Early on (small t) many sequences are possible, so every position is pulled toward a blend of plausible tokens,
and the positions are only weakly coordinated. By the time the picture is clear, most tokens are already fixed by
where the noise started. The model learned *which words occur in stories* (and partly which go with the prompt) but
not *which word goes next to which*: the joint structure of a sentence is lost in the averaging — the text version of
the toy Euler-1 collapse to the centre.

---
## Phase 11 — Add prompt-conditioned continuation

**Learn:** a prompt is added by holding its positions at the clean embeddings during the whole ODE (inpainting). The
same model was trained for this: half of its training sequences had a clean prefix.

**Files:** `flow/generate.py` (clamping), `flow/train.py` (`p_prompt` 0.5, prefix 8–24 tokens)

**Result:**
- **The prompt is preserved exactly in 100% of samples** (asserted in code for the nearest decoder; the projection
  decoder misreads 0.3% of clean embeddings, so for it the given prompt ids are written back).
- Validation MSE with a 16-token prompt (0.666) is *higher* than without (0.643): the prompt does not make the
  remaining tokens easier to predict for this model — a first sign it barely uses the prompt.
- Prompt relevance is low (Phase 10.3 and Phase 14): only 46% of continuations re-use a name introduced in the prompt
  (AR-5.3M 78%, real 95%, random words 17%).
- Unconditional samples (64 tokens from noise) score perplexity 51,985 (Heun 32), still word salad; AR-5.3M from
  `<BOS>` scores 13.4.

---
## Phase 12 — Compare ODE solvers and step counts

**Files:** `flow/run_eval.py --stage test`, `flow/benchmark.py` → `results/metrics/benchmark.json`,
`flow/compare.py` → `results/metrics/comparisons.json`

**Run:** `python -m flow.run_eval --stage test`, `python -m flow.benchmark`, `python -m flow.compare`

**Result** (300 test prompts, seed 0, nearest decoder; latency at batch 1, median) —
[figure 3](results/figures/fig3_steps.png), [figure 5](results/figures/fig5_speed.png)

| setting | NFE | scorer perplexity | vs chosen: NLL difference [95% CI] | latency |
|---|---|---|---|---|
| Euler 8 | 8 | 166,705 | +0.371 [+0.299, +0.445] | 18 ms |
| Heun 4 | 8 | 280,246 | +0.890 [+0.801, +0.983] | — |
| Euler 16 | 16 | 139,417 | +0.192 [+0.133, +0.253] | 33 ms |
| Heun 8 | 16 | 132,617 | +0.142 [+0.089, +0.196] | — |
| Euler 32 | 32 | 126,769 | +0.097 [+0.051, +0.142] | 68 ms |
| Heun 16 | 32 | 119,490 | +0.038 [+0.009, +0.067] | 68 ms |
| Euler 64 | 64 | 121,000 | +0.050 [+0.020, +0.080] | 134 ms |
| **Heun 32 (chosen)** | 64 | **115,068** | — | 133 ms |

Projection decoding is worse at every setting (+1.09 to +2.11 nats). Latency grows exactly linearly with NFE
(2.1 ms per network call); Heun costs the same per call as Euler.

- **H1 mostly holds:** quality improves from 8 to 32 Euler steps (167k → 127k) and only a little from 32 to 64
  (127k → 121k, a small but real difference). But even 64 steps leave the text a factor of ~370 behind MDLM.
- **H2 partly holds:** at the same NFE Heun beats Euler from 16 calls on, but at 8 calls (4 Heun steps) it is much
  worse — 4 steps are too few for the predictor–corrector to help when the path bends.

---
## Phase 13 — Compare with AR and diffusion LMs

**Result** (300 test prompts; flow and the 5.3M baselines pooled over 3 seeds) — [figure 4](results/figures/fig4_comparison.png)

| system | params | scorer perplexity | distinct-2 | blind coherence (0–2) | blind relevance (0–2) | latency | batch-64 tokens/s | NFE |
|---|---|---|---|---|---|---|---|---|
| real continuation | — | 4.0 | 0.554 | 2.00 | 2.00 | — | — | — |
| AR-13.9M | 13.9M | 7.2 | 0.505 | — | — | 143 ms | 3,588 | 48 |
| **AR-5.3M** | 5.3M | **22.7** | 0.523 | 1.05 | 1.70 | 96 ms | 4,965 | 48 |
| MDLM-13.9M | 13.9M | 44.6 | 0.473 | — | — | 146 ms | 432 | 32 |
| **MDLM-5.3M** | 5.3M | **313.4** | 0.458 | 1.00 | 1.20 | 143 ms | 490 | 32 |
| MDLM-5.3M, 8 steps | 5.3M | 543.4 | 0.479 | — | — | 35 ms | 1,784 | 8 |
| **flow 5.2M (Heun 32)** | 5.2M | **117,771** | 0.750 | **0.00** | **0.45** | 133 ms | 1,049 | 64 |
| flow 5.2M (Euler 8) | 5.2M | 166,705 | 0.761 | — | — | 18 ms | 7,605 | 8 |
| real words, shuffled | — | 60,756 | 0.774 | — | — | — | — | — |
| random frequent words | — | 155,007 | 0.818 | — | — | — | — | — |

**Paired differences in scorer NLL** (A − B, nats per token, 95% bootstrap over prompts):

| comparison | difference |
|---|---|
| flow − AR-5.3M | +8.56 [+8.48, +8.63] |
| flow − MDLM-5.3M | +5.93 [+5.86, +6.00] |
| MDLM-5.3M − AR-5.3M | +2.63 [+2.55, +2.71] |
| flow − real words shuffled | +0.66 [+0.56, +0.76] |
| flow − random frequent words | **−0.28 [−0.37, −0.18]** |

Peak extra memory at batch 64: flow ~300 MB, AR ~100–190 MB, MDLM ~600–700 MB.

- **H5 holds, far more strongly than expected:** the flow LM is not a weaker language model, it is not yet a language
  model of sentences at all. Its high distinct-2 is not "diversity": random words have even higher distinct-2.
- **H7 is refuted:** the flow model is *faster* at batch 1 than AR-5.3M (68 ms at 32 Euler steps vs 96 ms) and about
  2× faster per step than MDLM-5.3M (2.1 vs 4.5 ms per call): it has no 8,192-word output layer and no sampling, just
  384 numbers per position. AR wins throughput at batch 64 except against Euler 8. Speed is not the flow model's problem.

---
## Phase 14 — Ablations and error analysis

**Learn:** each ablation changes one thing and trains for 2,000 steps (16.4M tokens) against a reference with the main
settings trained twice (seeds 0 and 1). It **counts** only if its 95% interval against reference seed 0 excludes 0
**and** its perplexity is outside the range of the two reference seeds (112,012 – 126,154).

**Files:** `scripts/train_all.sh`, `flow/run_eval.py --stage ablations`, `flow/compare.py`, `flow/errors.py` →
`results/metrics/errors_auto.json`, `results/error_review/`, `results/metrics/errors_review.json`

**Run:** `python -m flow.run_eval --stage ablations`, `python -m flow.compare`,
`python -m flow.errors --stage auto`, `--stage sample`, label `blind.jsonl`, `--stage count`

**Ablations** ([figure 6](results/figures/fig6_ablations.png)):

| ablation (2,000 steps, seed 0) | val MSE | scorer ppl | vs reference [95% CI] | counts? | verdict |
|---|---|---|---|---|---|
| reference, seed 1 (seed noise) | 0.697 | 126,154 | +0.119 [+0.032, +0.205] | — | seed variation alone is ±0.12 nats |
| no time input (H8) | 0.754 | 105,058 | −0.064 [−0.160, +0.029] | no | **H8 refuted**: higher MSE, no effect on text |
| + token cross-entropy λ 0.05 (H9) | 0.796 | 178,489 | **+0.466** [+0.363, +0.569] | yes, **worse** | **H9 refuted**; follow-up full run not triggered |
| 128-d PCA space (H10) | 0.357* | 95,195 | **−0.163** [−0.275, −0.052] | yes, better | **H10 refuted** (opposite): smaller space slightly better |
| t = u² (more small-t training, H12) | 0.689 | 113,903 | +0.017 [−0.067, +0.101] | no | **H12 not supported** |
| length 32 vs 64, 8-token prompt + 24 tokens (H11) | 0.662 | 95,285 vs 137,824 | **−0.369** [−0.535, −0.208] | yes, better | **H11 refuted**: shorter windows help |

\*different space, not comparable. The **main model at 6,000 steps (115,068, seeds 117k/121k) is no better than the
2,000-step references (112k/126k)**: more training improved the flow loss but not the text.

All effects are small compared with the gap to MDLM (≈ 6 nats): every variant still writes word salad. The token
cross-entropy made the model sharper at t = 0.3 (96.6% vs 89.8% accuracy) but the text worse and its name re-use fell
to 15% — it pushes single positions to commit early, which is exactly the failure diagnosed in Phase 10.

**Automatic error measures** (300 test continuations; [figure 8](results/figures/fig8_errors.png)):

| system | adjacent token pairs seen in training | re-uses the prompt's name | words shared with prompt | same word twice in a row |
|---|---|---|---|---|
| real | 98.7% | 94.8% | 14.6% | 0% |
| AR-13.9M | 98.9% | 89.6% | 14.7% | 0.7% |
| AR-5.3M | 97.9% | 78.3% | 13.6% | 2.0% |
| MDLM-13.9M | 97.5% | 89.6% | 16.5% | 14.7% |
| MDLM-5.3M | 95.1% | 67.0% | 13.1% | 24.3% |
| **flow** | **46.3%** | **46.1%** | 11.5% | 54.7% |
| real words, shuffled | 50.6% | 93.9% | 14.3% | 51.7% |
| random frequent words | 45.8% | 16.5% | 9.5% | 45.0% |

**Blind review** (AI annotator — a Claude subagent, not a person — 20 test prompts × 4 systems, shuffled, systems
hidden; coherence and relevance 0–2):

| system | coherence | relevance | issue tags (count of 20) |
|---|---|---|---|
| real | 2.00 | 2.00 | broken word 1, grammar 1 |
| AR-5.3M | 1.05 | 1.70 | contradiction 13, grammar 13, repetition 4, names 2 |
| MDLM-5.3M | 1.00 | 1.20 | grammar 20, repetition 9, names 8, contradiction 4, off-topic 4 |
| flow | **0.00** | **0.45** | **word salad 20**, names 19, broken word 8 |

**Failure taxonomy** (from the 80 reviewed samples and the automatic measures):
1. **No local syntax** (all flow samples): neighbouring tokens are as unrelated as random words (46% seen pairs).
2. **Topic without order:** the words fit children's stories and partly the prompt (better than random words by 0.28
   nats; name re-use 46% vs 17%), but they are a bag.
3. **Name soup:** names from all over the training data appear together (Mae, Simon, Patty, Bessie…) — each position
   picks "some name" independently.
4. **Broken words:** sub-word pieces without their partner ("lo", "Mar", "Look\""), because neighbouring positions do
   not agree on how a word continues.
5. **Never ends:** no `<EOS>` (also because the training windows almost never contained one).

By contrast the baselines fail at the *story* level (AR: contradictions; MDLM: grammar slips and repetition), not at
the word-order level.

---
## Phase 15 — Repository structure

```
project10/
  configs/        flow.yaml (main settings), hypotheses.md (pre-registered)
  flow/           __init__ (paths), dataset, embeddings, flow_path, model, train, ode_sampler, generate, evaluate,
                  run_eval, diagnose, compare, benchmark, errors, plots, toy2d
  scripts/        env_report.py, lr_pilot.sh, train_all.sh
  tests/          test_flow_path, test_ode, test_shapes, test_embeddings (15 tests)
  data_manifests/ data.json (split rules, sizes, hashes, overlaps)
  results/
    metrics/      env, toy2d, embeddings, lr_pilot, train/<run>/, runs.json (every generation run), flow_chosen,
                  comparisons, diagnostics, benchmark, errors_auto, errors_review, logs/
    generations/  <run>.jsonl (every prompt, continuation and score)
    error_review/ blind.jsonl, key.jsonl, labels_ai.json
    figures/      fig0–fig8
  INSTRUCTIONS.md, README.md, project10.html (tutorial)
  (not in git: .venv, docs/, data/ 0.23 GB, checkpoints/ 0.32 GB)
```

---
## Phase 16 — Final report and deliverables

**Files:** `README.md`, `project10.html` (tutorial with the figures and interactive charts), this file.

**Hypothesis verdicts:**

| # | hypothesis | verdict |
|---|---|---|
| H1 | steps help 8→32, flat 32→64 | **mostly holds** (167k → 127k → 121k) |
| H2 | Heun ≥ Euler at equal NFE | **partly**: from 16 NFE yes, at 8 NFE no |
| H3 | nearest > projection | **holds** (3×; projection makes 24% unknown words) |
| H4 | samples end far from tokens; frequent-token overuse; low MSE ≠ good text | **gap and MSE part hold** (distance 13.5 vs 0; MSE fell, text did not); **overuse refuted** |
| H5 | flow clearly worse than AR-5.3M and MDLM-5.3M | **holds, strongly** (+8.6 / +5.9 nats; coherence 0) |
| H6 | prompts preserved, but less relevant | **holds** (100% preserved; relevance 0.45 vs 1.2–1.7; 74% of tokens unchanged without the prompt) |
| H7 | flow slower than AR, ≈ MDLM | **refuted**: faster than both at batch 1 |
| H8 | time input needed | **refuted** for text (no effect), holds for MSE |
| H9 | token CE is the best ablation | **refuted**: it is the only harmful one |
| H10 | 128-d PCA worse | **refuted**: slightly better |
| H11 | 32 tokens not better | **refuted**: better on the 8+24 task |
| H12 | more small-t training helps | **not supported** |

**Cost:** main model 32 min; 7 ablations 10–25 min each (token CE 2.3× slower); all training ≈ 2.5 CPU-hours (+ 0.7
for the pilot); dev sweep ≈ 2 min, test evaluation of all systems ≈ 8 min, ablation evaluation ≈ 3 min,
benchmark ≈ 4 min.

**Reproduce everything (CPU, about 3.5 hours):**
```
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.lock.txt
python scripts/env_report.py && python -m flow.toy2d
python -m flow.dataset && python -m flow.embeddings
bash scripts/lr_pilot.sh            # + --lr 6e-3 and 1.2e-2 the same way (see Phase 7)
bash scripts/train_all.sh           # main + 7 ablations, about 2.5 h
python -m flow.run_eval --stage dev
python -m flow.run_eval --stage test
python -m flow.run_eval --stage ablations
python -m flow.compare && python -m flow.diagnose && python -m flow.benchmark
python -m flow.errors --stage auto && python -m flow.errors --stage sample
python -m flow.errors --stage count # after labelling results/error_review/blind.jsonl
python -m flow.plots
pytest                              # 15 tests
```
Requires Project 1 (`data/tokenizer.json`, `data/full/`, `checkpoints/small/model_best.pt`), Projects 2–3 (baseline
checkpoints) and Project 4 (`bench/`, `checkpoints/scorer_tinystories33m`) next to this folder, read-only; hashes in
`results/metrics/env.json`. Every random choice has a fixed seed.

## What would plausibly fix it (not tested here)
- **Self-conditioning / more steps where it matters:** let each step see the previous `x1_hat`, or spend many more
  steps at t < 0.25 where 80% of tokens are decided.
- **A learned, lower-dimensional space** trained jointly with a decoder (as in CDCD / Plaid), with a rounding loss
  that keeps it from collapsing; the 128-d PCA result hints that a tighter space helps.
- **Much more compute:** published continuous text-diffusion models need roughly 10–100× more training than AR to
  catch up; 49M tokens is a tiny budget for a model that must learn a whole joint distribution at once.
- **Discrete flow matching** (flows over token probabilities instead of embeddings), which keeps the MDLM advantage.

## Limitations
- One small model size (5.2M), one 49M-token budget, one fixed window of 64 tokens; ablations with one seed each.
- The scorer (TinyStories-33M) and the bigram test measure fluency, not story quality; at perplexities above 10⁵ small
  differences between flow variants mean little for a reader.
- The blind review was done by an AI annotator (Claude), not by people, on 20 prompts per system.
- H12 was added after seeing the pilot's text (before the main run), and is marked as such.
