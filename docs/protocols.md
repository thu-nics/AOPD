# Scientific protocols

| Recipe | Data/collection | Training steps |
|---|---|---:|
| main | 59 AWM + 5 EnvScaler tasks per step | 100 |
| tau | 8 airline + 8 retail + 8 telecom tasks per step | 50 |

N=4 student candidates share one state; K=3 teacher votes retain duplicates.
Positive reward is `1 + lambda * (count - 1)/(K - 1)` with lambda=0.5;
legal nonmatches receive 0, invalid actions -1. Partial teacher sets still use
configured K as denominator. Equal-reward groups are masked. Only one candidate
advances the real environment. Terminal success is measured, not added to the
semantic loss. Missing supervision masks that group, not earlier healthy groups.

Both recipes commit one highest-reward candidate with randomized tie-breaking;
the main recipe also applies its progress safeguards. Native schema validation
and Tau's unexecuted-transfer-notice penalty remain enabled. Teacher and matcher
failures mask only the affected state group, preserving earlier healthy groups.

Main: 32,000 total, 27,904 prompt, 4,096 response tokens; up to 20 AWM and 40
EnvScaler decisions. Tau: 32,768 total, 24,576 student prompt, 4,096 response;
20 agent decisions.

Teacher and student see the same public tools/history by default; teacher-only
guidance requests one action. No official test-split task is used for training.
Official train counts: airline 30, retail 74, telecom 74;
test counts: 20, 40, 40.

Student decoding: thinking enabled, T=0.6, top-p=.95, top-k=20. AdamW, LR=1e-6,
constant schedule, zero warmup, one PPO epoch, minibatch 32; group centering then
batch whitening, active-token mean, PPO clip .2, dual clip 3; no KL penalty or
entropy bonus. Current loss normalization includes the microbatch aggregation
fix. Hydra details remain in `verl/trainer/config/`.

The separate Tau outcome-GRPO entry point uses vanilla trajectory rollout,
terminal DB/communicate reward and trajectory-level GRPO. It does not construct
teacher state groups or query a teacher/matcher. Its executable defaults are in
`verl/trainer/config/tau_outcome.yaml` and `examples/tau_bench/train/run.sh`;
see the [launch command](runtime.md#tau-outcome-grpo).

Benchmark evaluation is external. Optional periodic validation reuses the
training rollout engine; it is not a replacement for final benchmark evaluation.
Tau training and validation use DB/communicate outcome checks without NL assertions.
