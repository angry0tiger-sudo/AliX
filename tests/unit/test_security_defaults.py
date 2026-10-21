import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_config_is_dry_run_and_has_no_enabled_live_trading() -> None:
    config = json.loads((ROOT / "config.json").read_text())
    assert config["dry_run"] is True
    assert config["max_open_trades"] == 0
    assert config["exchange"]["key"] == ""
    assert config["exchange"]["secret"] == ""
    assert config["exchange"]["pair_whitelist"] == []


def test_env_example_contains_no_real_secret_values() -> None:
    content = (ROOT / ".env.example").read_text()
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if any(token in key for token in ("TOKEN", "SECRET", "API_KEY")):
            assert value == "", f"secret-like placeholder must be empty: {key}"
    assert "BALE_CHAT_ID=832112407" in content
    assert "ALIx_BALE_CHAT_ALLOWLIST=832112407" in content
