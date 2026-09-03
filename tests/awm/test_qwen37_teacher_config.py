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


def test_qwen37_flash_overrides_compose_without_changing_judges():
    config = _compose(
        "awm_envscaler_agentic_opd",
        [
            "env.awm.oracle.provider=dashscope",
            "env.awm.oracle.model=qwen3.7-flash",
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
    assert (oracle.provider, oracle.model, oracle.reasoning_effort) == ("dashscope", "qwen3.7-flash", None)
    assert (oracle.thinking_budget, oracle.temperature, oracle.top_p, oracle.max_tokens) == (4096, 0.6, 0.95, 8192)
    assert (oracle.matcher_provider, config.env.awm.runtime_failures.judge.provider) == ("deepseek", "deepseek")
