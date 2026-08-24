# About this fork

This is a **personal integration branch**, not a release and not a proposal to upstream.
It exists to run one model on one machine well:

| | |
|---|---|
| GPU | RTX 5090 **Laptop** (GB203, 82 SMs — not the desktop's 170), 24 GiB |
| OS | Windows 11, MSVC + CUDA 13.x + Ninja |
| Model | Qwen3.8-27B NVFP4, single artifact, 262,144-token context |
| Serving | `rk2v4-e8` KV, `--prefill-chunk 1024`, MTP `k=6`, vision overlay, WebUI |

Upstream is [Neroued/ninfer](https://github.com/Neroued/ninfer). Nothing here is intended to
replace it.

## What it combines

Work from several open pull requests and personal forks, none of it merged upstream at the time
of writing. Every commit was cherry-picked with `-x` and **retains its original author**:

- **Compressed-KV (E8 lattice/root)** — Daniel Parker, from PR #35. Two bits per key dimension is
  what puts a 262k context on a 24 GiB card at all.
- **Windows support** — pelebel, from PR #59, plus local fixes for artifacts above 4 GiB and short
  direct reads at EOF.
- **Vision residency overlay** — Valeriy Selitskiy. Keeps the vision tower host-pinned and borrows
  device memory from evicted weights, returning its footprint to KV capacity.
- **MTP and CUDA work** — Mirko Covizzi: adaptive verification widths, the wide-MTP decode fix, and
  deriving cooperative-launch limits from the active GPU (without which the Laptop's 82 SMs die at
  `--prefill-chunk 1024`).
- **Serving fixes** — jpf (content-part arrays in tool messages), Gideon Zenz (typing tool-call
  parameters by their declared schema), mr-september (stray `</think>` leaking into content),
  natpate (in-process llama.cpp WebUI).

If you are one of these authors and would rather your work were not carried here, say so and it
comes out.

## How changes get in

Six rules, each of them learned the expensive way.

**1. Measure on this machine, or do not carry it.** Every patch is A/B'd on the same binary, the
same prompt, with speculative-decode acceptance as a control. Upstream's MoE-prefetch PR claimed
+3.7%; measured here it was inside run-to-run variance, so it was dropped. A turboquant LUT hoist
claimed nothing in particular and was worth +2% at depth, so it stayed. Benchmarks published
against a 35B-A3B do not transfer to a 27B dense model.

**2. Carry no unmeasured divergence.** Fork divergence is a standing cost paid at every rebase.
A commit that cannot be shown to help on this configuration is not kept "just in case".

**3. Commit subjects are not a reliable guide — read the dispatch.** A patch titled *E8* may
dispatch only on `e8_lattice` and never touch our `e8_root` path. A patch titled *INT8* may reach
us precisely because E8 rides the `DType::I8` route. Both mistakes happened, in opposite
directions: one nearly discarded a 5x decode win, the other nearly added dead weight.

**4. A feature's repair commits are not named after the feature.** Cherry-picking the five commits
whose subject said *webui* produced a tree that failed to compile and then aborted at runtime; the
two commits that fixed it were called *build the test suite under MSVC* and *keep the
reasoning-effort conflict in prompt resolution*. Always build and run the affected suites — a
textually clean merge proves nothing.

**5. Synthetic single-prompt comparisons mislead; the real loop decides.** Prefill chunking and
speculative draft width both change the greedy trajectory, so two configurations emit *different
tokens* — measured here, not assumed. Comparing them on one prompt measures the trajectory, not
the implementation. Judge on many real agentic sessions, and prefer `tokens/round`, which is
invariant to context depth, over raw tokens per second.

**6. Prefill is what you wait for.** In agentic sessions the user waits on time-to-first-token,
not on decode. The two largest wins here are both prefill wins: raising the prefill chunk to 1024
(2001 → 2817 tok/s) and keeping reasoning between turns so the prefix cache stops rewinding
(−67% TTFT per request, −53% of all prefill work).

## Layout

`e8-overlay-adaptive` is the daily driver. `e8-win` and `e8-win-overlay` are deliberately older
known-good fallbacks — they are *not* kept in sync, so that a bad day on the driver has somewhere
to fall back to. The `pr*` branches hold A/B endpoints, including patches that were measured and
rejected. Tags named `pre-*` are rollback points taken before each round of changes.

Licensed under Apache 2.0, like upstream.
