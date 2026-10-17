"""Shared fixtures for Chulengo tests."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path so `import chulengo` works during collection.
sys.path.insert(0, str(Path(__file__).parent.parent))

import chulengo


@pytest.fixture(autouse=True)
def isolate_chulengo_config(tmp_path_factory, monkeypatch):
    """Redirect the config path so tests never write to the real user config.

    Several commands (``create``/``update``/first-run seeding in
    ``load_settings``) write through ``save_settings``. If a test forgets to
    mock ``save_settings``, an unredirected run would clobber
    ``~/.config/chulengo/models.yaml``. Point both the module global and the
    ``CHULENGO_CONFIG`` env var at a throwaway path for every test.
    """
    target = tmp_path_factory.mktemp("chulengo-config") / "models.yaml"
    monkeypatch.setattr(chulengo, "SETTINGS_FILE", target)
    monkeypatch.setenv("CHULENGO_CONFIG", str(target))
    return target


@pytest.fixture
def default_settings_content():
    """Return default models.yaml content for testing."""
    return """
defaults:
  generic:
    ctx_size: 4096
    flash_attn: off
    reasoning: off
  qwen:
    ctx_size: 8192
    flash_attn: on
    reasoning: off
    jinja: on
model_overrides: {}
"""


@pytest.fixture
def mock_filesystem(mocker, tmp_path):
    """Mock filesystem operations for testing."""
    # Create a fake models.yaml file
    settings_file = tmp_path / "models.yaml"
    settings_file.write_text(
        "defaults:\n  generic:\n    ctx_size: 4096\nmodel_overrides: {}"
    )

    # Mock Path.exists to check for specific test scenarios
    def mock_exists(self):
        # Check if this is the user settings file
        if str(self).endswith("models.yaml"):
            return True
        return Path(self).exists()

    mocker.patch("pathlib.Path.exists", mock_exists)

    return tmp_path


@pytest.fixture
def sample_model_names():
    """Return sample model names for testing."""
    return [
        "Qwen/Qwen3-8B",
        "meta-llama/Llama-3-8B",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-2-9b-it",
        "codestral/codestral-embeddings",
        "Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF:Q4_K_M",
    ]


@pytest.fixture
def sample_model_families():
    """Return mapping of model names to their expected name-detection
    families (pure name matching, no GGUF). Dots separate a minor version in
    name detection, so 'Llama-3.1-8B' resolves to llama3, not llama3_1; the
    narrower blocks (llama3_1, qwancoder, qwopus) only apply when a cached
    GGUF reports them."""
    return {
        "Qwen/Qwen3-8B": "qwen3",
        "Qwen/Qwen2.5-Coder": "qwen",  # qwancoder not reachable by this name
        "meta-llama/Llama-3-8B": "llama3",
        "meta-llama/Llama-3.1-8B": "llama3",  # name-only: dots keep minor version
        "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
        "mistralai/Mixtral-8x7B-Instruct-v7.1": "mixtral",
        "google/gemma-2-9b-it": "gemma2",
        "google/gemma-3-27B-it": "gemma3",
        "codestral/codestral-embeddings": "codestral",
        "Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF": "qwopus",
    }
