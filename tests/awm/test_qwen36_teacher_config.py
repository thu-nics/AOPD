from pathlib import Path

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).parents[2]
CONFIG_DIR = ROOT / "verl" / "trainer" / "config"


def _compose(config_name, overrides=None):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        return compose(config_name=config_name, overrides=overrides or [])


def test_awm_oracle_provider_defaults_remain_deepseek():
    for config_name in ("awm_agentic_opd", "awm_outcome"):
        config = _compose(config_name)
        oracle = config.env.awm.oracle
        assert oracle.provider == "deepseek"
        assert oracle.api_base == "https://api.deepseek.com"
        assert oracle.enable_thinking is True
        assert oracle.reasoning_effort == "max"
        assert oracle.thinking_budget is None
        assert oracle.temperature is None
        assert oracle.top_p is None
        assert oracle.matcher_provider == "deepseek"
        assert oracle.matcher_model == "deepseek-v4-flash"
        assert oracle.matcher_api_key_env == "DEEPSEEK_API_KEY"


def test_qwen36_flash_overrides_compose_without_changing_judges():
    config = _compose(
        "awm_envscaler_agentic_opd",
        [
            "env.awm.oracle.provider=dashscope",
            "env.awm.oracle.model=qwen3.6-flash",
            "env.awm.oracle.api_base=https://dashscope.aliyuncs.com/compatible-mode/v1",
            "env.awm.oracle.api_key_env=DASHSCOPE_API_KEY",
            "env.awm.oracle.reasoning_effort=null",
            "env.awm.oracle.thinking_budget=4096",
            "env.awm.oracle.temperature=0.6",
            "env.awm.oracle.top_p=0.95",
            "env.awm.oracle.max_tokens=8192",
        ],
    )
    oracle = config.env.awm.oracle
    assert (oracle.provider, oracle.model, oracle.reasoning_effort) == ("dashscope", "qwen3.6-flash", None)
    assert (oracle.thinking_budget, oracle.temperature, oracle.top_p, oracle.max_tokens) == (4096, 0.6, 0.95, 8192)
    assert (oracle.matcher_provider, config.env.awm.runtime_failures.judge.provider) == ("deepseek", "deepseek")


def test_qwen36_flash_teacher_wrapper_uses_dashscope_without_changing_judges():
    wrapper = (ROOT / "examples" / "agentic_opd" / "run_mixed_qwen36_flash_teacher.sh").read_text(encoding="utf-8")
    launcher = (ROOT / "examples" / "awm" / "train" / "run_training.sh").read_text(encoding="utf-8")

    assert 'ORACLE_PROVIDER="${ORACLE_PROVIDER:-dashscope}"' in wrapper
    assert 'ORACLE_MODEL="${ORACLE_MODEL:-qwen3.6-flash}"' in wrapper
    assert 'ORACLE_API_KEY_ENV="${ORACLE_API_KEY_ENV:-DASHSCOPE_API_KEY}"' in wrapper
    assert 'ORACLE_ENABLE_THINKING="${ORACLE_ENABLE_THINKING:-true}"' in wrapper
    assert 'ORACLE_REASONING_EFFORT="${ORACLE_REASONING_EFFORT:-null}"' in wrapper
    assert 'ORACLE_THINKING_BUDGET="${ORACLE_THINKING_BUDGET:-4096}"' in wrapper
    assert 'ORACLE_TEMPERATURE="${ORACLE_TEMPERATURE:-0.6}"' in wrapper
    assert 'ORACLE_TOP_P="${ORACLE_TOP_P:-0.95}"' in wrapper
    assert 'ORACLE_MAX_TOKENS="${ORACLE_MAX_TOKENS:-8192}"' in wrapper
    assert 'MATCHER_PROVIDER="${MATCHER_PROVIDER:-deepseek}"' in launcher
    assert 'RUNTIME_JUDGE_PROVIDER="${RUNTIME_JUDGE_PROVIDER:-deepseek}"' in launcher
    assert 'env.awm.oracle.provider="$ORACLE_PROVIDER"' in launcher
    assert '"env.awm.runtime_failures.judge.provider=$RUNTIME_JUDGE_PROVIDER"' in launcher
