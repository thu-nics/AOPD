import pytest

from aopd.runtime import resolve_roles, validate_runtime


def runtime():
    return {
        "model": "/models/student",
        "student": {"gpus": [2, 5], "tp": 1, "sp": 1},
        "services": {
            "shared": {"mode": "local", "model": "qwen", "model_path": "/models/qwen", "gpus": [0, 3], "tp": 2, "port": 8180},
            "teacher": {"mode": "api", "provider": "openai-compatible", "model": "teacher", "base_url": "https://example.org/v1", "api_key_env": "TEACHER_KEY"},
        },
        "roles": {"user": {"service": "shared"}, "matcher": {"service": "shared"}, "teacher": {"service": "teacher"}},
    }


def test_noncontiguous_devices_and_shared_roles():
    r = runtime()
    validate_runtime(r)
    roles = resolve_roles(r)
    assert roles["user"]["base_url"] == "http://127.0.0.1:8180/v1"
    assert roles["matcher"]["model"] == roles["user"]["model"]
    assert roles["teacher"]["base_url"] == "https://example.org/v1"


def test_overlapping_independent_service_rejected():
    r = runtime()
    r["services"]["shared"]["gpus"] = [0, 2]
    with pytest.raises(ValueError, match="overlap"):
        validate_runtime(r)


def test_six_student_gpus_are_not_rejected_as_non_power_of_two():
    r = runtime()
    r["student"] = {"gpus": [2, 3, 4, 5, 6, 7], "tp": 2, "sp": 2}
    r["services"]["shared"]["gpus"] = [0, 1]
    validate_runtime(r)


def test_parallelism_must_divide_gpu_count():
    r = runtime()
    r["student"]["sp"] = 3
    with pytest.raises(ValueError, match="divid"):
        validate_runtime(r)


def test_remote_services_need_no_gpu_and_no_literal_secret():
    r = runtime()
    r["services"]["teacher"]["api_key"] = "do-not-store-secrets"
    with pytest.raises(ValueError, match="api_key_env"):
        validate_runtime(r)


def test_missing_service_is_not_silently_replaced():
    r = runtime()
    r["roles"]["teacher"]["service"] = "missing"
    with pytest.raises(ValueError, match="missing"):
        validate_runtime(r)
