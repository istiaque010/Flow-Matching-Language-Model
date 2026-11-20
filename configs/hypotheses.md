# Pre-registered hypotheses and controls (written 2026-11-20, before the main flow model was trained)

Written during the learning-rate pilot; no flow model had generated text yet and no test data had been used.
Known at this point: the toy 2D result, the embedding statistics (Phase 3) and the first pilot losses.

## Fixed conditions

| item | value |
|---|---|
| data | first 64 tokens of every TinyStories training story (1.77M openings), Project 1 tokenizer and splits |
| representation | Project 1 `small` token embeddings (384-d), frozen, standardised (frequency-weighted mean 0, variance 1) |
| flow model | bidirectional Transformer, 4 layers, 4 heads, width 256, adaLN time conditioning, 5.18M parameters |
| training | straight-line flow matching, t ~ U(0,1), masked MSE; 50% of sequences with a clean prompt prefix of 8-24 tokens; 6,000 steps x 128 x 64 = 49.2M tokens (= the budget of the two 5.3M baselines) |
| comparison task | 300 held-out test openings: first 16 tokens = prompt, 48 new tokens; same prompt ids for every system |
| baselines | AR-5.3M (Project 2 student from scratch, 49M tokens), MDLM-5.3M (Project 3 mdlm_small, 49M tokens), and the 13.9M AR / MDLM models as references; decoding settings = the ones Projects 1/3/4 chose on validation data (AR top-p 0.95; MDLM-5.3M 32 steps top-k 50; MDLM-13.9M 32 steps random order top-k 20) |
| flow decoding choice | steps, solver and decoder chosen on 100 validation ("dev") prompts only, by scorer perplexity (ties: fewer function evaluations) |
| metrics | scorer perplexity (TinyStories-33M, independent), distinct-1/2, rep4, adjacent duplicates, unknown words, EOS share, NFE, latency (median / p95 at batch 1), throughput (batch 64), peak memory; blind AI review of coherence and prompt relevance (disclosed as AI) |

Training losses of different model families are never compared (AR perplexity, MDLM bound and flow MSE measure
different things).

## Hypotheses

| # | hypothesis |
|---|---|
| H1 | **Steps.** Text quality improves from 8 to 32 Euler steps and is nearly flat from 32 to 64; latency grows about linearly with NFE. |
| H2 | **Solver.** At equal NFE, Heun is at least as good as Euler (as in the toy 2D problem). |
| H3 | **Decoder.** Nearest-embedding decoding gives better text than the tied projection head. |
| H4 | **Continuous-discrete gap.** Generated vectors end far from every token embedding compared with clean embeddings, and the decoded text over-uses frequent tokens (lower distinct-2 and more repetition than real text). Low flow MSE does not imply good text. |
| H5 | **Main comparison.** At the same size and token budget, the flow LM writes clearly worse continuations than AR-5.3M and MDLM-5.3M (higher scorer perplexity, lower coherence in the blind review). |
| H6 | **Prompting.** Prompts are preserved exactly in 100% of samples; flow continuations are less related to the prompt than the baselines'. |
| H7 | **Speed.** At batch 1, the flow LM with 32 steps is slower than AR-5.3M with a KV cache and about as fast as MDLM-5.3M with 32 steps (same network size and window). |
| H8 | **Time conditioning.** Removing t from the network raises validation MSE and worsens text. |
| H9 | **Auxiliary token loss.** Adding a token cross-entropy on the implied clean sample improves text (scorer perplexity) more than any other ablation. |
| H10 | **Embedding dimension.** A 128-d PCA version of the embeddings gives worse text (tokens are closer together: median neighbour distance 10 instead of 19). |
| H11 | **Length.** A 32-token model is not better per token than the 64-token model. |

| H12 | **Time sampling (added after seeing the pilot's text, before the main run).** Drawing t with more weight on early times (t = u^2, so half of the training samples have t < 0.25) improves text. Reason: the pilot decodes 88% of tokens correctly from the implied clean sample already at t = 0.3, so the decisions that matter happen early, where uniform t spends only 30% of training. |

**Ablations** (H8-H12) are trained for 2,000 steps (16.4M tokens) against a reference with the main settings trained
twice (seeds 0 and 1) for 2,000 steps. An ablation counts only if (a) its paired 95% bootstrap interval (over the 300
test prompts) against reference seed 0 excludes zero and (b) its value lies outside the range of the two reference
seeds. Otherwise: "no effect beyond seed variation".

**Follow-up rule (decided now):** if the auxiliary-loss ablation counts as better on scorer perplexity, a full-budget
(6,000-step) model with that loss is trained and reported next to the main model as a second flow system.
