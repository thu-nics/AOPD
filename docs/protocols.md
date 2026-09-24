# Scientific protocols

| Recipe | Data/collection | Training steps |
|---|---|---:|
| main | 59 AWM + 5 EnvScaler tasks per step | 100 |
| tau-full | 8 airline + 8 retail + 8 telecom tasks per step | 50 |
| tau-a1 | Full collection; validity-only optimization reward | 50 |
| tau-a4 | Full with binary teacher-hit reward | 50 |
| tau-a5 | Full reward; uniform random candidate commit | 50 |
| tau-s1 | Current student weights as teacher; public context plus teacher guidance | 50 |
| tau-s2 | S1 plus reviewed private customer facts and reference goals | 50 |

N=4 student candidates share one state; K=3 teacher votes retain duplicates.
Positive reward is `1 + lambda * (count - 1)/(K - 1)` with lambda=0.5;
legal nonmatches receive 0, invalid actions -1. Partial teacher sets still use
configured K as denominator. Equal-reward groups are masked. Only one candidate
advances the real environment. Terminal success is measured, not added to the
semantic loss. Missing supervision masks that group, not earlier healthy groups.

Full uses uniform argmax commit with the current progress safeguards. A1 keeps
semantic commit but learns validity; A4 removes frequency bonus; A5 removes
selection by reward. Native schema validation and the unexecuted-transfer-notice
penalty remain enabled. See Tau `ablations.py` for exact overrides.

Main: 32,000 total, 27,904 prompt, 4,096 response tokens; up to 20 AWM and 40
EnvScaler decisions. Tau: 32,768 total, 24,576 student prompt, 4,096 response;
20 agent decisions. Self teacher has 4,096 additional private-prompt tokens;
the shared public history is not separately shortened to fit its private input.

Teacher and student see the same public tools/history by default. S1 changes
teacher guidance only. S2 uses training-only reviewed briefs and answer evidence;
it does not expose user-side executable tools as agent tools. No official
test-split task is used for training. Official train counts: airline 30, retail 74, telecom 74;
test counts: 20, 40, 40.

Student decoding: thinking enabled, T=0.6, top-p=.95, top-k=20. AdamW, LR=1e-6,
constant schedule, zero warmup, one PPO epoch, minibatch 32; group centering then
batch whitening, active-token mean, PPO clip .2, dual clip 3; no KL penalty or
entropy bonus. Current loss normalization includes the microbatch aggregation
fix. Hydra details remain in `verl/trainer/config/`.

Native Tau eval uses the full official base split by default, four trials,
greedy agent, output 4,096, total context 40,960 and 200 native transitions.
NL assertions are disabled; DB/communicate outcome checks are retained.
Report domains separately and separate train/test when assessing in-domain Tau
training. pass^k means all k attempts succeed, not at-least-one pass@k. Incomplete
planned trials must remain visible and count as failures in conservative tables.
