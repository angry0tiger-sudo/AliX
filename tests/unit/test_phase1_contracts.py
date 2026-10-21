import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_freqtrade_configuration_is_stopped_dry_run_and_market_closed() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    assert config["dry_run"] is True
    assert config["initial_state"] == "stopped"
    assert config["max_open_trades"] == 0
    assert config["stake_amount"] == 0
    assert config["exchange"]["key"] == ""
    assert config["exchange"]["secret"] == ""
    assert config["exchange"]["pair_whitelist"] == []
    assert config["exchange"]["pair_blacklist"] == [".*"]


def test_freqtrade_version_matches_pinned_novaryn_integration_contract() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "freqtrade==2026.7" in requirements
    assert "baleio @ git+https://github.com/ehsndvr/baleio.git@66be3000af6e62332e1764f227aed64847a0d9ba" in requirements


def test_bale_defaults_are_changeable_and_reporting_is_disabled() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "BALE_CHAT_ID=832112407" in env_example
    assert "ALIx_BALE_CHAT_ALLOWLIST=832112407" in env_example
    assert "ALIx_REPORTING_ENABLED=false" in env_example
    assert "BALE_BOT_TOKEN=" in env_example


def test_phase_one_has_no_concrete_network_transport_or_order_write() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    assert config["dry_run"] is True
    assert config["max_open_trades"] == 0
    assert config["force_entry_enable"] is False
