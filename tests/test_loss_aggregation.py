"""CPU objective/gradient/update checks; no model checkpoints or GPUs required."""

from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, compute_value_loss
from verl.trainer.ppo.loss_aggregation import LOSS_AGG_MODES, LossNormalization, loss_normalization_metadata, validate_loss_config
from verl.workers.ppo_batch import split_ppo_batch, training_response_mask

MODES = sorted(LOSS_AGG_MODES)


def reference(loss, mask, mode, length=16):
    valid = mask.bool()
    lengths = mask.sum(-1)
    sums = torch.where(valid, loss, 0.0).sum(-1)
    if mode == "token-mean":
        return sums.sum() / lengths.sum().clamp(min=1)
    if mode == "seq-mean-token-mean":
        sums = sums / lengths.clamp(min=1)
    result = sums.sum() / (lengths > 0).sum().clamp(min=1)
    return result / length if mode == "seq-mean-token-sum-norm" else result


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("partitions", [[[0, 1, 2, 3]], [[0], [1, 2], [3]], [[3, 1], [2], [0]]])
def test_loss_and_gradient_are_partition_invariant(mode, partitions):
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 0, 0]])
    full = torch.arange(20, dtype=torch.float64).reshape(4, 5).requires_grad_()
    target = reference(full, mask, mode)
    target.backward()
    split = full.detach().clone().requires_grad_()
    norm = LossNormalization.from_mask(mask, loss_normalizer_length=16)
    actual = sum(agg_loss(split[idx], mask[idx], mode, normalization=norm) for idx in partitions)
    actual.backward()
    torch.testing.assert_close(actual, target)
    torch.testing.assert_close(split.grad, full.grad)
    # Extra tensor-width padding and a wholly masked row have no influence.
    padded = torch.nn.functional.pad(split.detach(), (0, 6, 0, 1), value=float("nan")).requires_grad_()
    padded_mask = torch.nn.functional.pad(mask, (0, 6, 0, 1))
    padded_loss = agg_loss(padded, padded_mask, mode, loss_normalizer_length=16)
    padded_loss.backward()
    torch.testing.assert_close(padded_loss, target)
    assert torch.isfinite(padded.grad).all()
    torch.testing.assert_close(padded.grad[:4, :5], full.grad)
    assert not padded.grad[4].any()


@pytest.mark.parametrize("mode", MODES)
def test_empty_losses_are_finite_zero_with_zero_gradient(mode):
    values = torch.full((2, 3), float("nan"), dtype=torch.float64, requires_grad=True)
    mask = torch.zeros_like(values)
    loss = agg_loss(values, mask, mode, loss_normalizer_length=16)
    loss.backward()
    assert loss.item() == 0 and not values.grad.any()
    for func in (compute_policy_loss, compute_value_loss):
        current = values.detach().clone().requires_grad_()
        if func is compute_policy_loss:
            results = func(current.detach(), current, current.detach(), mask, cliprange=0.2, loss_agg_mode=mode, loss_normalizer_length=16)
        else:
            results = func(current, current.detach(), current.detach(), mask, 0.2, loss_agg_mode=mode, loss_normalizer_length=16)
        results[0].backward()
        assert all(torch.isfinite(item).all() and not item.any() for item in results)
        assert not current.grad.any()


def test_config_and_gspo_contracts():
    for value in (None, 0, -1, True, 2.5):
        with pytest.raises(ValueError, match="loss_normalizer_length"):
            validate_loss_config(OmegaConf.create({"loss_agg_mode": "seq-mean-token-sum-norm", "loss_normalizer_length": value}))
    data = torch.ones(2, 3)
    with pytest.raises(ValueError, match="GSPO requires"):
        compute_policy_loss_gspo(data, data, data, data, cliprange=0.2, loss_agg_mode="token-mean")


@pytest.mark.parametrize("compatible", [False, True])
def test_checkpoint_resume_reports_loss_protocol_change(monkeypatch, compatible):
    from verl.utils.checkpoint import fsdp_checkpoint_manager as checkpoint

    manager = checkpoint.FSDPCheckpointManager.__new__(checkpoint.FSDPCheckpointManager)
    manager.model = torch.nn.Linear(1, 1)
    manager.optimizer = None
    manager.lr_scheduler = None
    manager.rank, manager.world_size = 0, 1
    config = OmegaConf.create({"loss_agg_mode": "token-mean", "loss_normalizer_length": 16})
    manager.training_metadata = loss_normalization_metadata(config)
    extra = {"lr_scheduler": None}
    if compatible:
        extra["training_metadata"] = dict(manager.training_metadata)
    monkeypatch.setattr(checkpoint, "copy_to_local", lambda path: path)
    monkeypatch.setattr(checkpoint, "get_fsdp_state_ctx", lambda *args: nullcontext())
    monkeypatch.setattr(checkpoint.torch, "load", lambda path, **kw: extra if "extra_state" in path else None if "optim_world" in path else manager.model.state_dict())
    if compatible:
        import warnings

        with warnings.catch_warnings(record=True) as emitted:
            manager.load_checkpoint("/unused-offline-test")
        assert not emitted
    else:
        with pytest.warns(UserWarning, match="not a lossless training resume"):
            manager.load_checkpoint("/unused-offline-test")


def test_yaml_default_normalizer_is_fixed_response_budget():
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../verl/trainer/config"):
        config = compose(config_name="ppo_trainer", overrides=["data.max_response_length=4096"])
    assert config.actor_rollout_ref.actor.loss_normalizer_length == 4096
    assert config.critic.loss_normalizer_length == 4096
    assert config.actor_rollout_ref.actor.loss_normalization == "global-minibatch-v1"


def batch_fixture(multimodal=False):
    n, length = 7, 6
    responses = torch.arange(n * length).reshape(n, length) % 9
    attention = torch.ones(n, length + 2, dtype=torch.long)
    mask = torch.tensor([[1] * 6, [1, 0, 0, 0, 0, 0], [0] * 6, [1] * 4 + [0] * 2, [1] * 3 + [0] * 3, [1] * 5 + [0], [1, 1, 0, 0, 0, 0]])
    # Attention includes one extra token; training must use the explicit mask.
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.cat((torch.ones(n, 2, dtype=torch.long), responses), dim=1),
            "responses": responses,
            "attention_mask": attention,
            "position_ids": torch.arange(length + 2).expand(n, -1),
            "old_log_probs": torch.full((n, length), -0.6, dtype=torch.float64),
            "ref_log_prob": torch.full((n, length), -0.7, dtype=torch.float64),
            "advantages": torch.linspace(-2, 2, n * length, dtype=torch.float64).reshape(n, length),
            "response_mask": mask,
            "values": torch.full((n, length), -0.5, dtype=torch.float64),
            "returns": torch.linspace(-1, 1, n * length, dtype=torch.float64).reshape(n, length),
        },
        non_tensors={"is_padding": np.array([False] * 6 + [True])},
        meta_info={"temperature": 0.6, "multi_turn": False},
    )
    if multimodal:
        data.non_tensor_batch["multi_modal_inputs"] = np.array([{"pixel_values": torch.tensor([i])} for i in range(n)], dtype=object)
    return data


def toy_prediction(model, data):
    x = data["input_ids"][:, -data["responses"].shape[1] :].double() / 10
    return model(torch.stack((x, x.square()), dim=-1)).squeeze(-1) - 0.7


def run_worker(monkeypatch, role, mode, micro_size, dynamic, multimodal, mini_size=7, loss_mode="vanilla", data=None):
    from verl.utils.debug import performance
    from verl.workers.actor import dp_actor
    from verl.workers.critic import dp_critic

    monkeypatch.setattr(performance, "_get_current_mem_info", lambda: ("0", "0", "0", "0"))
    module = dp_actor if role == "actor" else dp_critic
    monkeypatch.setattr(module, "get_torch_device", lambda: SimpleNamespace(current_device=lambda: "cpu"))
    cls = module.DataParallelPPOActor if role == "actor" else module.DataParallelPPOCritic
    worker = cls.__new__(cls)
    model = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.17, -0.08]]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    worker.config = OmegaConf.create(
        {
            "loss_agg_mode": mode,
            "loss_normalizer_length": 16,
            "ppo_mini_batch_size": mini_size,
            "ppo_micro_batch_size_per_gpu": micro_size,
            "use_dynamic_bsz": dynamic,
            "ppo_max_token_len_per_gpu": 16,
            "ppo_epochs": 2,
            "use_kl_loss": True,
            "kl_loss_type": "k2",
            "kl_loss_coef": 0.11,
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.12,
            "clip_ratio_high": 0.25,
            "cliprange_value": 0.2,
            "entropy_coeff": 0.07,
            "policy_loss": {"loss_mode": loss_mode},
        }
    )
    setattr(worker, f"{role}_module", model)
    setattr(worker, f"{role}_optimizer", optimizer)
    worker.ulysses_sequence_parallel_size = 1
    if role == "actor":
        worker._forward_micro_batch = lambda micro_batch, **kw: (2 - toy_prediction(model, micro_batch).square() / 2, toy_prediction(model, micro_batch))
    else:
        worker._forward_micro_batch = lambda micro_batch: toy_prediction(model, micro_batch)
    gradients = []

    def step():
        gradients.append(model.weight.grad.clone())
        norm = torch.linalg.vector_norm(model.weight.grad)
        optimizer.step()
        return norm

    worker._optimizer_step = step
    batch = batch_fixture(multimodal) if data is None else data
    result = worker.update_policy(batch) if role == "actor" else worker.update_critic(batch)
    return model.weight.detach(), gradients, result, optimizer.state_dict(), worker


@pytest.mark.parametrize("role", ["actor", "critic"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("micro_size,dynamic,multimodal", [(1, False, False), (3, False, False), (2, True, False), (3, False, True), (3, True, True)])
def test_real_worker_update_is_microbatch_invariant(monkeypatch, role, mode, micro_size, dynamic, multimodal):
    expected = run_worker(monkeypatch, role, mode, 7, False, False)
    actual = run_worker(monkeypatch, role, mode, micro_size, dynamic, multimodal)
    torch.testing.assert_close(actual[0], expected[0], atol=1e-12, rtol=1e-10)
    for got, target in zip(actual[1], expected[1], strict=True):
        torch.testing.assert_close(got, target, atol=1e-12, rtol=1e-10)
    assert actual[2].keys() == expected[2].keys()
    for key in actual[2]:
        np.testing.assert_allclose(actual[2][key], expected[2][key], atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_tail_minibatches_and_empty_optimizer_steps(monkeypatch, role):
    expected = run_worker(monkeypatch, role, "token-mean", 4, False, False, mini_size=4)
    actual = run_worker(monkeypatch, role, "token-mean", 2, False, True, mini_size=4)
    torch.testing.assert_close(actual[0], expected[0])
    assert len(actual[1]) == 4  # two minibatches, two epochs (including small tail)
    worker = actual[-1]
    optimizer = getattr(worker, f"{role}_optimizer")
    before = deepcopy(optimizer.state_dict())
    weights = getattr(worker, f"{role}_module").weight.detach().clone()
    empty = batch_fixture(True)
    empty.batch["response_mask"].zero_()
    metrics = worker.update_policy(empty) if role == "actor" else worker.update_critic(empty)
    torch.testing.assert_close(getattr(worker, f"{role}_module").weight, weights)
    assert metrics[f"{role}/optimizer_step_skipped"] == [1.0] * 4
    for key, state in before["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(optimizer.state_dict()["state"][key][field], value)


def test_gspo_worker_partition_and_mask_invariants(monkeypatch):
    expected = run_worker(monkeypatch, "actor", "seq-mean-token-mean", 7, False, False, loss_mode="gspo")
    actual = run_worker(monkeypatch, "actor", "seq-mean-token-mean", 1, True, False, loss_mode="gspo")
    torch.testing.assert_close(actual[0], expected[0])
    for got, target in zip(actual[1], expected[1], strict=True):
        torch.testing.assert_close(got, target)


def test_explicit_masks_padding_and_multimodal_linkage():
    data = batch_fixture(True)
    data.non_tensor_batch["state_group_skip_loss"] = np.array([False, True] + [False] * 5)
    mask = training_response_mask(data)
    assert not mask[1].any() and not mask[6].any()
    assert not data.batch["response_mask"][1].eq(0).all()  # original data not mutated
    chunks = split_ppo_batch(data, 4)
    assert [len(chunk) for chunk in chunks] == [4, 3]
    assert chunks[1].non_tensor_batch["multi_modal_inputs"][0]["pixel_values"].item() == 4
    data.batch["loss_mask"] = data.batch["attention_mask"].clone()
    data.batch["loss_mask"][0, -3:] = 0
    turn_mask = training_response_mask(data, multi_turn=True)
    assert turn_mask[0].sum() == 3
    assert not turn_mask[1].any() and not turn_mask[6].any()
    # Without a multi-turn loss mask, critic fallback uses response positions,
    # not the shifted prediction positions (which would include an extra token).
    single = batch_fixture()
    del single.batch["response_mask"]
    single.batch["attention_mask"][0, -3:] = 0
    assert training_response_mask(single)[0].sum() == 3


@pytest.mark.parametrize("world,sp", [(2, 1), (2, 2), (8, 1), (8, 2), (8, 4), (8, 8)])
@pytest.mark.parametrize("mode", MODES)
def test_logical_dp_sp_layouts(world, sp, mode):
    dp = world // sp
    matrices = [torch.arange(12, dtype=torch.float64).reshape(3, 4) + rank for rank in range(dp)]
    masks = [torch.tensor([[1, 1, 1, 1], [int(rank % 2 == 0), 0, 0, 0], [0, 0, 0, 0]]) for rank in range(dp)]
    full = torch.cat(matrices).requires_grad_()
    mask = torch.cat(masks)
    target = reference(full, mask, mode)
    target.backward()
    norm = LossNormalization(mask.sum().item(), (mask.sum(-1) > 0).sum().item(), dp, 16)
    replicas, losses = [], []
    for rank in range(world):
        local = matrices[rank // sp].clone().requires_grad_()
        loss = sum(agg_loss(local[i : i + 1], masks[rank // sp][i : i + 1], mode, normalization=norm) for i in range(3))
        loss.backward()
        replicas.append(local.grad)
        losses.append(loss.item())
    assert np.mean(losses) == pytest.approx(target.item())
    for rank in range(dp):
        torch.testing.assert_close(sum(replicas[rank * sp : (rank + 1) * sp]) / world, full.grad[rank * 3 : (rank + 1) * 3])


def distributed_check(rank, world, sp, init_file):
    """True Gloo all-reduce + DDP + Ulysses Gather backward; no GPU allocation."""
    from verl.utils.ulysses import Gather

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world, timeout=timedelta(seconds=90))
    try:
        groups = [dist.new_group(list(range(start, start + sp))) for start in range(0, world, sp)]
        sp_group = groups[rank // sp]
        for mode in MODES:
            for empty_rank in (False, True):
                model = torch.nn.Linear(1, 1, bias=False).double()
                model.weight.data.fill_(0.2)
                ddp = torch.nn.parallel.DistributedDataParallel(model)
                dp = world // sp
                full_inputs = torch.arange(1, dp * 8 + 1, dtype=torch.float64).reshape(dp * 2, 4, 1) / 10
                full_mask = torch.ones(dp * 2, 4, dtype=torch.float64)
                full_mask[::2, 1:] = 0
                if empty_rank:
                    full_mask[-2:] = 0
                local_mask = full_mask[(rank // sp) * 2 : (rank // sp + 1) * 2]
                norm = LossNormalization.from_mask(local_mask, distributed=True, sequence_parallel_size=sp, loss_normalizer_length=16)
                assert norm.sequence_count == (full_mask.sum(-1) > 0).sum().item()
                optimizer = torch.optim.SGD(ddp.parameters(), lr=0.1)
                local_input = full_inputs[(rank // sp) * 2 : (rank // sp + 1) * 2]
                # Two microbatches, including a rank with no trainable tokens.
                total = 0.0
                for i in range(2):
                    token_input = local_input[i : i + 1].chunk(sp, dim=1)[rank % sp]
                    local_prediction = ddp(token_input).squeeze(-1)
                    prediction = Gather.apply(sp_group, local_prediction, 1, True)
                    loss = agg_loss(prediction.square(), local_mask[i : i + 1], mode, normalization=norm)
                    total += loss.detach()
                    loss.backward()
                expected_model = torch.nn.Linear(1, 1, bias=False).double()
                expected_model.weight.data.fill_(0.2)
                expected_loss = reference(expected_model(full_inputs).squeeze(-1).square(), full_mask, mode)
                expected_loss.backward()
                torch.testing.assert_close(model.weight.grad, expected_model.weight.grad)
                dist.all_reduce(total)
                torch.testing.assert_close(total / world, expected_loss)
                optimizer.step()
                expected_optimizer = torch.optim.SGD(expected_model.parameters(), lr=0.1)
                expected_optimizer.step()
                torch.testing.assert_close(model.weight, expected_model.weight)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,sp", [(2, 1), (4, 2)])
def test_real_distributed_gradients(tmp_path, world, sp):
    torch.multiprocessing.spawn(distributed_check, args=(world, sp, str(tmp_path / "gloo-init")), nprocs=world, join=True)
