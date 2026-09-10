# FSDP loss aggregation

The current FSDP actor and critic normalize over **one global optimizer
minibatch**, not independently per microbatch. This applies to PPO/Agentic OPD
policy loss, entropy regularization, KL regularization and clipped value loss.
The default remains `token-mean`; reward and advantage definitions are unchanged.

Let `B` be the number of rows with at least one trainable token, `L_i` the
trainable token count in row i, and `S_i` its masked token-loss sum:

| loss_agg_mode | Global optimizer-minibatch objective |
| --- | --- |
| token-mean | sum(S_i) / sum(L_i) |
| seq-mean-token-sum | sum(S_i) / B |
| seq-mean-token-mean | sum(S_i / L_i) / B |
| seq-mean-token-sum-norm | sum(S_i) / (B * L_ref) |

`L_ref=actor_rollout_ref.actor.loss_normalizer_length` defaults to
`data.max_response_length`. The critic inherits it. It is a fixed configured
budget, never the current padded tensor width. GSPO explicitly requires
`seq-mean-token-mean`; incompatible settings fail rather than being ignored.

## Invariants

- Counts exclude explicit padding, estimator skip rows and masked tokens.
  A zero-advantage row is not automatically a masked row.
- Static and token-balanced dynamic microbatches contribute additive losses.
  Do not additionally divide by microbatch count or multiply by row fractions.
- Counts are synchronized once per minibatch across FSDP ranks. Ulysses SP
  copies are counted once; its existing gather-backward scaling is preserved.
- A locally empty rank still forwards/backwards to participate in collectives.
  A globally empty minibatch skips optimizer updates, including AdamW decay
  and momentum. An entirely skipped update also skips the LR scheduler.
- Small final minibatches are retained, including multimodal batches.
- Loss and clip/KL/value statistics sum microbatch contributions before the
  existing rank reduction. Across multiple optimizer steps, logged metrics
  are step means, not means of arbitrarily partitioned microbatches.

The signature plumbing in Megatron accepts the fixed normalizer, but its
pipeline/distributed accumulation has **not** been upgraded or validated by
this FSDP change. Do not claim Megatron partition invariance from these tests.

## Resume and experiment comparison

`loss_normalization: global-minibatch-v1` is recorded in resolved run config
and FSDP checkpoint extra-state metadata, together with mode and fixed divisor.
Loading a checkpoint with missing/different metadata warns explicitly. Model,
optimizer and task-schedule restoration remains possible; the subsequent
objective has changed, so this is **not numerically lossless continuation** of
the former training algorithm. Prefer a fresh run for comparisons and retain
old results under their original protocol.

In particular, fixing token-mean can change gradient directions as well as
scale when response lengths differ. Do not compare old and new grad_norm or
loss curves as if this were only a logging correction.

## Offline verification

```bash
python -m pytest tests/test_loss_aggregation.py -q
```

Tests compare independent full-batch formulas, gradients and optimizer updates
across all four modes, static/dynamic/multimodal partitions, small tails, explicit
padding and empty ranks. They run the actual FSDP actor/critic update methods
with tiny CPU models, plus real Gloo/DDP and Ulysses-gather backward on 2/4
processes and logical 2/8-rank SP layouts. This does not replace an eventual GPU
FSDP end-to-end smoke on a free machine.
