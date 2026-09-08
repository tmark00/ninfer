# Long-context probe

Two questions, one harness: at a given depth, does the model still **read**, and does it still
**reason**? Both are scored programmatically against values the filler cannot imply.

```
python probe.py --exe ../../build-win/apps/ninfer-serve.exe \
                --model /path/qwen3_8_27b_nvfp4.ninfer --mode reasoning
```

Arms are `CONTEXT:ROPE_SCALE:HAYSTACK_BYTES`. The default three compare an unscaled baseline
against YaRN on a **byte-identical** haystack, then push past the native ceiling. That first pair
is the whole point: it is the only comparison in which scaling is the sole variable.

## What the two modes measure

`needle` asks for six planted values verbatim. It tests positional recall, which is exactly what
rope scaling perturbs, and nothing else.

`reasoning` asks six questions that each need several values combined — a sum spanning the first
and last percent of the corpus, an ordering over all six, a parity count, and one multi-hop item
where a rule planted at 88% names three values held at 82%, 92% and 5%. Getting that last one
right means the model crossed the whole context, not that it looked something up.

## Results, 27B NVFP4 on a 5090 Laptop, 2026-09-08

| mode | context | scale | tokens | score | prefill |
|---|---:|---:|---:|:---:|---:|
| needle | 262144 | 1.0 | 181773 | 6/6 | 1252.0 tok/s |
| needle | 393216 | 1.5 | 181773 | 6/6 | 1240.4 tok/s |
| needle | 393216 | 1.5 | 381351 | 6/6 | 644.0 tok/s |
| reasoning | 262144 | 1.0 | 181835 | 6/6 | 1257.9 tok/s |
| reasoning | 393216 | 1.5 | 181835 | 6/6 | 1245.8 tok/s |
| reasoning | 393216 | 1.5 | 381413 | 6/6 | 652.3 tok/s |

On the identical haystack YaRN costs about 1% of prefill in both modes and nothing in score. In
the deep arm four of six values, and the rule itself, sit beyond the 262144 the weights were
trained for.

## Two traps this harness was built around

**Size the haystack in tokens, not bytes.** The first run of this probe targeted 240k and 350k
tokens and produced 182k and 266k, because this Polish filler runs about 3.52 bytes to the token
rather than the 2.6 that was assumed. The deep arm ended 3673 tokens past the ceiling with its
deepest value still inside it, so a clean 6/6 proved nothing at all. Read the `prompt=` field the
server reports and correct the byte target before trusting a deep arm.

**A test everything passes cannot rank anything.** Every arm above scores 6/6, so these runs show
that scaling does not break reading or reasoning — they do not measure what it costs. Detecting a
small regression needs a task the unscaled baseline itself does not saturate.

## Why one request per arm

The engine keeps a single rewrite checkpoint per lane, so a second question against the same
haystack diverges before that checkpoint and replays the entire prefill. Six questions would mean
six full prefills; at 380k tokens that is an hour of prefill to answer six short questions. Asking
all of them in one request costs one prefill and still scores per item.
