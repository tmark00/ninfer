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

Ten rules, each of them learned the expensive way.

**1. Measure on this machine, or do not carry it.** Every patch is A/B'd on the same binary, the
same prompt, with speculative-decode acceptance as a control. Upstream's MoE-prefetch PR claimed
+3.7%; measured here it was inside run-to-run variance, so it was dropped. A turboquant LUT hoist
claimed nothing in particular and was worth +2% at depth, so it stayed. Benchmarks published
against a 35B-A3B do not transfer to a 27B dense model. How many runs a claim needs is
proportional to what it decides: five interleaved A/B/A/B passes when the answer picks a
direction, one or two when it is curiosity.

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
(−67% TTFT per request, −53% of all prefill work). Two kernel patches taken from upstream in
September 2026 — reading the NVFP4 activation scales tile-contiguous, and walking that kernel's
grid token-fastest so the weight matrix is streamed once instead of once per tile — added a
further 13.7%, from 4827 to 5488 tok/s.

**7. Measure through a warm server, not repeated launches of the CLI.** A fresh process spends
about twelve seconds loading at idle clocks and then measures a 1.4-second window while the clock
is still climbing. The result is not noisy, it is *bimodal*: one run in three lands 6–14% low, and
averaging them hides it. The same work driven at an already-warm server has a coefficient of
variation near 0.5% and no outliers at all, which is the difference between being able to resolve
a 1% effect and not. Record the laptop's power mode with every number — Balanced, Performance and
Hyperboost are three different machines, and a comparison that spans two of them is measuring the
fans. Interleaving alone does not save you either: a plain A/B/A/B against a clock that sinks
monotonically through the session puts every B later than its A, and the drift lands on the arm
difference. One such run read +0.00% on pooled medians while B lost all three pairs. Counterbalance
the order — ABBA — so each pair holds one A before and one after.

**8. Verify a port twice: once by the diff, once by the output.** *By the diff*, because on Windows
a clean-looking patch can be an entire file rewritten in CRLF — and `git show HEAD:file` hides the
carriage returns, so the only honest witnesses are `git cat-file blob` and what `git diff --cached
--stat` says after staging. Twenty-four changed lines and four hundred and thirty-one look
identical in the terminal. *By the output*, because a kernel patch that is meant to be a pure
scheduling change should emit token-for-token what it replaced; that check is cheap and it is the
only inexpensive way to tell a correct port from a subtly wrong one. An approximation has to outbid
it. Upstream's approximate `silu` was worth a real +0.87% here (t = 4.96) and was still reverted,
because a percent does not buy back the ability to verify.

**9. A wrong constant is not a finding; what it does at your operating point is.** Upstream writes
its SM count as a literal 170, the desktop part, and that literal is wrong twice over in this tree
on an 82-SM laptop. Fixing it in the sparse-MoE prefill was worth +1.92 pp with t = 19.3 and was
taken; fixing it in the GDN chunked output measured −0.35 pp with t = −1.5 and was thrown away. The
difference is not the constant, it is where our fixed prefill chunk lands against it. The MoE
kernels are persistent, and 510 uniform blocks over 246 slots retire in three waves with the last
one carrying eighteen blocks; the GDN launcher at our one operating point merely reshuffled a
single wave. Compute where you actually sit before you believe a diff.

**10. Calibrate a probe in the units the system measures, and distrust one nothing fails.** A
long-context probe sized in bytes came out at 182k and 266k tokens against a 240k and 350k target,
because the corpus ran 3.52 bytes to the token and not the assumed 2.6 — which left the deep arm
3673 tokens past the ceiling with its deepest needle still inside it. Three clean 6/6 results, and
the one that was supposed to prove something proved nothing. Read back the count the server itself
reports. And when every arm scores full marks, say so plainly: a saturated test shows an effect is
absent, never how large it is, and ranking arms needs a task the baseline does not already ace.

## Layout

`e8-overlay-adaptive` is the daily driver. `e8-win` and `e8-win-overlay` are deliberately older
known-good fallbacks — they are *not* kept in sync, so that a bad day on the driver has somewhere
to fall back to. The `pr*` branches hold A/B endpoints, including patches that were measured and
rejected. Tags named `pre-*` are rollback points taken before each round of changes.

`yarn-393k` is the one branch that serves past the model's native context. The 27B weights are
trained to 262,144; it adds YaRN and raises the ceiling to 393,216 at scale 1.5, which the probe in
`tools/long_context_probe` reads and reasons over at 381k tokens for about 1% of prefill. It stays
a branch rather than a merge because the rotation changes, so the same prompt takes a different
greedy path even well inside the native range — a separate serving mode, not a larger ceiling on
the daily driver.

Licensed under Apache 2.0, like upstream.
