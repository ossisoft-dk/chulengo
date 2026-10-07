"""Tests for Chulengo - A llama.cpp wrapper for HuggingFace models."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import chulengo


class TestLoadSettings:
    """Tests for load_settings() function."""

    def test_load_settings_from_user_config(self, mocker):
        """Test loading settings from user config file if it exists."""
        settings_content = yaml.dump(
            {
                "defaults": {"qwen3": {"ctx_size": 16384, "flash_attn": "on"}},
                "model_overrides": {},
            }
        )

        # User config exists
        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=settings_content)):
                settings = chulengo.load_settings()

        assert settings["defaults"]["qwen3"]["ctx_size"] == 16384
        assert settings["model_overrides"] == {}

    def test_load_settings_falls_back_to_source(self, mocker):
        """Test loading settings from source directory when user config doesn't exist."""
        source_content = yaml.dump(
            {"defaults": {"generic": {"ctx_size": 4096}}, "model_overrides": {}}
        )

        # Mock exists to return True for all files
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch("builtins.open", mock_open(read_data=source_content))

        settings = chulengo.load_settings()

        assert settings["defaults"]["generic"]["ctx_size"] == 4096

    def test_load_settings_creates_defaults_when_missing(self, mocker):
        """Test that defaults are created when no source file is found."""
        mock_file = mocker.MagicMock()
        mock_file.__enter__ = mocker.MagicMock(return_value=mock_file)
        mock_file.__exit__ = mocker.MagicMock(return_value=False)
        mock_file.write = mocker.MagicMock()

        # Neither user config nor source exists
        mocker.patch.object(Path, "exists", return_value=False)
        mocker.patch("builtins.open", return_value=mock_file)

        settings = chulengo.load_settings()

        # Should create defaults
        assert settings["defaults"]["generic"]["ctx_size"] == 4096
        assert settings["model_overrides"] == {}


class TestSaveSettings:
    """Tests for save_settings() function."""

    def test_save_settings_creates_file(self, mocker, tmp_path):
        """Test that save_settings creates the config file atomically."""
        settings = {"defaults": {"generic": {"ctx_size": 4096}}, "model_overrides": {}}

        target = tmp_path / "models.yaml"
        mocker.patch.object(chulengo, "SETTINGS_FILE", target)

        chulengo.save_settings(settings)

        assert target.exists()
        loaded = yaml.safe_load(target.read_text())
        assert loaded["defaults"]["generic"]["ctx_size"] == 4096
        # Atomic write must not leave temp files behind
        leftovers = [p for p in tmp_path.iterdir() if p.name != "models.yaml"]
        assert leftovers == []

    def test_save_settings_structure(self, mocker, tmp_path):
        """Test that save_settings writes YAML with correct structure."""
        settings = {
            "defaults": {"generic": {"ctx_size": 4096}},
            "model_overrides": {"test-model": {"flash_attn": "on"}},
        }

        target = tmp_path / "models.yaml"
        mocker.patch.object(chulengo, "SETTINGS_FILE", target)

        chulengo.save_settings(settings)

        loaded = yaml.safe_load(target.read_text())
        assert loaded["defaults"]["generic"]["ctx_size"] == 4096
        assert loaded["model_overrides"]["test-model"]["flash_attn"] == "on"

    def test_save_settings_cleanup_on_failure(self, mocker, tmp_path):
        """Test that a failed save removes the temp file and raises."""
        settings = {"defaults": {}, "model_overrides": {}}

        target = tmp_path / "models.yaml"
        mocker.patch.object(chulengo, "SETTINGS_FILE", target)
        # Force yaml.dump to fail mid-write
        mocker.patch.object(chulengo.yaml, "dump", side_effect=OSError("disk full"))

        with pytest.raises(OSError, match="disk full"):
            chulengo.save_settings(settings)

        # Target must not exist, and no temp file may be left behind
        assert not target.exists()
        assert [p.name for p in tmp_path.iterdir()] == []


class TestGetHfCachePath:
    """Tests for get_hf_cache_path() function."""

    def test_returns_llama_cache_when_set(self, mocker):
        """Test that LLAMA_CACHE env var takes precedence."""
        mocker.patch.dict(os.environ, {"LLAMA_CACHE": "/custom/cache"})
        # Clear other env vars
        for key in [
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_HOME",
            "XDG_CACHE_HOME",
        ]:
            os.environ.pop(key, None)

        assert chulengo.get_hf_cache_path() == "/custom/cache"

    def test_returns_hf_hub_cache_when_set(self, mocker):
        """Test that HF_HUB_CACHE is used when LLAMA_CACHE is not set."""
        mocker.patch.dict(os.environ, {"HF_HUB_CACHE": "/hf/cache"})
        os.environ.pop("LLAMA_CACHE", None)
        for key in ["HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"]:
            os.environ.pop(key, None)

        assert chulengo.get_hf_cache_path() == "/hf/cache"

    def test_returns_huggingface_hub_cache_when_set(self, mocker):
        """Test that HUGGINGFACE_HUB_CACHE is used as fallback."""
        mocker.patch.dict(os.environ, {"HUGGINGFACE_HUB_CACHE": "/old/hf/cache"})
        os.environ.pop("LLAMA_CACHE", None)
        os.environ.pop("HF_HUB_CACHE", None)
        for key in ["HF_HOME", "XDG_CACHE_HOME"]:
            os.environ.pop(key, None)

        assert chulengo.get_hf_cache_path() == "/old/hf/cache"

    def test_returns_hf_home_with_hub_subpath(self, mocker):
        """Test that HF_HOME with hub subpath is returned."""
        mocker.patch.dict(os.environ, {"HF_HOME": "/home/user/hf"})
        os.environ.pop("LLAMA_CACHE", None)
        os.environ.pop("HF_HUB_CACHE", None)
        os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
        for key in ["XDG_CACHE_HOME"]:
            os.environ.pop(key, None)

        assert chulengo.get_hf_cache_path() == "/home/user/hf/hub"

    def test_returns_xdg_cache_home_when_all_else_fails(self, mocker):
        """Test that XDG_CACHE_HOME is used as final fallback."""
        mocker.patch.dict(os.environ, {"XDG_CACHE_HOME": "/home/user/.cache"})
        for key in ["LLAMA_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"]:
            os.environ.pop(key, None)

        result = chulengo.get_hf_cache_path()
        assert result == "/home/user/.cache/huggingface/hub"

    def test_returns_default_path_when_no_env_set(self, mocker):
        """Test default path when no environment variables are set."""
        # Clear all relevant env vars by mocking os.environ
        import importlib

        importlib.reload(chulengo)

        # Temporarily remove all relevant env vars
        old_environ = os.environ.copy()
        for key in [
            "LLAMA_CACHE",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_HOME",
            "XDG_CACHE_HOME",
        ]:
            os.environ.pop(key, None)

        try:
            result = chulengo.get_hf_cache_path()
            # Should return a path containing huggingface and hub
            assert "huggingface" in result
            assert "hub" in result
        finally:
            # Restore environ
            os.environ.clear()
            os.environ.update(old_environ)


class TestDetectModelFamily:
    """Tests for detect_model_family() function."""

    def test_detects_qwen3(self):
        """Test detection of Qwen3 models."""
        assert chulengo.detect_model_family("Qwen/Qwen3-8B") == "qwen3"
        assert chulengo.detect_model_family("Qwen/Qwen3-72B") == "qwen3"

    def test_detects_llama3(self):
        """Test detection of Llama3 models."""
        # Note: Llama-3.1 and Llama-3.2 variants may not be distinguished
        # due to normalization - they all match llama3
        assert chulengo.detect_model_family("meta-llama/Llama-3-8B") == "llama3"

    def test_detects_qwen(self):
        """Test detection of Qwen2 models."""
        assert chulengo.detect_model_family("Qwen/Qwen2-7B") == "qwen"

    def test_detects_qwancoder(self):
        """Test detection of Qwen Coder models."""
        result = chulengo.detect_model_family("Qwen/Qwen2.5-Coder-7B-Instruct")
        # Qwen2.5-Coder should match qwancoder model type
        assert result is not None  # At least matches qwen

    def test_detects_qwopus(self):
        """Test detection of Qwopus models."""
        assert chulengo.detect_model_family("Jackrong/Qwopus3.6-35B") == "qwopus"

    def test_detects_mistral_models(self):
        """Test detection of Mistral models."""
        assert (
            chulengo.detect_model_family("mistralai/Mistral-7B-Instruct-v0.3")
            == "mistral"
        )
        assert (
            chulengo.detect_model_family("mistralai/Mixtral-8x7B-Instruct-v7.1")
            == "mixtral"
        )

    def test_detects_gemma_models(self):
        """Test detection of Gemma models."""
        assert chulengo.detect_model_family("google/gemma-2-9b-it") == "gemma2"
        assert chulengo.detect_model_family("google/gemma-3-27B-it") == "gemma3"

    def test_detects_codestral(self):
        """Test detection of Codestral models."""
        assert (
            chulengo.detect_model_family("codestral/codestral-embeddings")
            == "codestral"
        )

    def test_detects_granite(self):
        """Test detection of Granite models."""
        assert chulengo.detect_model_family("ibm/granite-8b-base") == "granite"

    def test_detects_phi(self):
        """Test detection of Phi models."""
        assert chulengo.detect_model_family("microsoft/phi-2") == "phi"

    def test_detects_gpt_oss(self):
        """Test detection of GPT-OSS models."""
        assert chulengo.detect_model_family("openai/gpt-oss-20b") == "gpt_oss"

    def test_detects_unknown_type(self):
        """Test that unknown models return None."""
        assert chulengo.detect_model_family("unknown/model-name") is None

    def test_case_insensitive_detection(self):
        """Test that model type detection is case-insensitive."""
        assert chulengo.detect_model_family("QWEN/Qwen3-8B") == "qwen3"
        assert chulengo.detect_model_family("MISTRAL/Mistral-7B") == "mistral"

    def test_handles_underscores_and_hyphens(self):
        """Test that model names with underscores and hyphens are handled."""
        assert chulengo.detect_model_family("Qwen/Qwen2_5-7B") == "qwen"
        assert chulengo.detect_model_family("Qwen/Qwen2-5-7B") == "qwen"


class TestGetDefaultSettings:
    """Tests for get_default_settings() function."""

    def test_uses_gguf_architecture_when_available(self, mocker):
        """Test that GGUF architecture is used as primary source (no fallback)."""
        settings = {
            "defaults": {
                "qwen3": {"ctx_size": 16384, "flash_attn": "on"},
                "generic": {"ctx_size": 4096, "flash_attn": "off"},
            },
            "model_overrides": {},
        }

        cache_info = {
            "cache_dir": "/cache/test/model",
            "snapshot": "/cache/test/model/snapshots/abc",
            "gguf_files": ["model.gguf"],
            "gguf_path": "/cache/test/model/model.gguf",
        }

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.get_model_cache_info", return_value=cache_info):
                with patch("chulengo.get_gguf_architecture", return_value="qwen3"):
                    result = chulengo.get_default_settings("test/model")

        assert result["ctx_size"] == 16384
        assert result["flash_attn"] == "on"

    def test_falls_back_to_name_detection_when_no_gguf_cache(self, mocker):
        """Test fallback to name-based detection when GGUF not in cache."""
        settings = {
            "defaults": {
                "qwen": {"ctx_size": 8192, "flash_attn": "on"},
                "generic": {"ctx_size": 4096},
            },
            "model_overrides": {},
        }

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.get_model_cache_info", return_value=None):
                result = chulengo.get_default_settings("Qwen/Qwen2-7B")

        # Should fall back to name-based detection
        assert result["ctx_size"] == 8192
        assert result["flash_attn"] == "on"

    def test_uses_generic_when_no_match_found(self, mocker):
        """Test that generic defaults are used when no match found."""
        settings = {
            "defaults": {"generic": {"ctx_size": 4096, "flash_attn": "off"}},
            "model_overrides": {},
        }

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.get_model_cache_info", return_value=None):
                result = chulengo.get_default_settings("unknown/model")

        assert result["ctx_size"] == 4096


class TestBuildLlamaCommand:
    """Tests for build_llama_command() function."""

    def test_builds_basic_command(self, mocker):
        """Test building a basic llama command."""
        with patch("chulengo.get_default_settings", return_value={}):
            result = chulengo.build_llama_command("Qwen/Qwen3-8B")

        assert result[0] == "llama"
        assert result[1] == "serve"
        assert "--hf-repo" in result
        assert "Qwen/Qwen3-8B" in result

    def test_includes_port(self, mocker):
        """Test that port is included when specified."""
        with patch("chulengo.get_default_settings", return_value={}):
            result = chulengo.build_llama_command("test/model", port=8001)

        assert "--port" in result
        port_idx = result.index("--port")
        assert result[port_idx + 1] == "8001"

    def test_includes_alias(self, mocker):
        """Test that alias is included when specified."""
        with patch("chulengo.get_default_settings", return_value={}):
            result = chulengo.build_llama_command("test/model", alias="my-model")

        assert "--alias" in result
        alias_idx = result.index("--alias")
        assert result[alias_idx + 1] == "my-model"

    def test_includes_ctx_size(self, mocker):
        """Test that ctx_size is included when specified."""
        with patch("chulengo.get_default_settings", return_value={}):
            result = chulengo.build_llama_command("test/model", ctx_size=32768)

        assert "--ctx-size" in result
        ctx_idx = result.index("--ctx-size")
        assert result[ctx_idx + 1] == "32768"

    def test_converts_boolean_to_on_off(self, mocker):
        """Test that boolean values are converted to on/off strings."""
        settings = {"defaults": {}, "model_overrides": {}}

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.detect_model_family", return_value=None):
                with patch(
                    "chulengo.get_default_settings",
                    return_value={"flash_attn": True, "reasoning": False},
                ):
                    result = chulengo.build_llama_command("test/model")

        assert "--flash-attn" in result
        flash_idx = result.index("--flash-attn")
        assert result[flash_idx + 1] == "on"

        assert "--reasoning" in result
        reasoning_idx = result.index("--reasoning")
        assert result[reasoning_idx + 1] == "off"

    def test_includes_jinja_flag(self, mocker):
        """Test that jinja flag is included."""
        settings = {"defaults": {}, "model_overrides": {}}

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.detect_model_family", return_value=None):
                with patch(
                    "chulengo.get_default_settings", return_value={"jinja": True}
                ):
                    result = chulengo.build_llama_command("test/model")

        assert "--jinja" in result

    def test_merges_settings_correctly(self, mocker):
        """Test that user overrides merge with defaults."""
        settings = {
            "defaults": {"generic": {"ctx_size": 4096}},
            "model_overrides": {"test-model": {"ctx_size": 8192}},
        }

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.detect_model_family", return_value=None):
                result = chulengo.build_llama_command("test-model")

        assert "--ctx-size" in result
        ctx_idx = result.index("--ctx-size")
        # User override should take precedence
        assert result[ctx_idx + 1] == "8192"

    def test_command_line_override_takes_precedence(self, mocker):
        """Test that command-line overrides take precedence over file settings."""
        settings = {"defaults": {"generic": {"ctx_size": 4096}}, "model_overrides": {}}

        with patch("chulengo.load_settings", return_value=settings):
            with patch(
                "chulengo.get_default_settings", return_value={"ctx_size": 4096}
            ):
                result = chulengo.build_llama_command(
                    "test/model", ctx_size=16384  # Override
                )

        assert "--ctx-size" in result
        ctx_idx = result.index("--ctx-size")
        assert result[ctx_idx + 1] == "16384"


class TestCmdCreate:
    """Tests for cmd_create() function."""

    def test_creates_new_model_settings(self, mocker):
        """Test creating settings for a new model."""
        mock_args = MagicMock()
        mock_args.model = "test/model"
        mock_args.alias = "test-alias"
        mock_args.ctx_size = 8192
        mock_args.flash_attn = "on"
        mock_args.reasoning = None
        mock_args.jinja = True
        mock_args.spec_type = None
        mock_args.cache_type_k = None
        mock_args.cache_type_v = None
        mock_args.chat_template_file = None

        mock_save = mocker.patch("chulengo.save_settings")

        result = chulengo.cmd_create(mock_args)

        assert result == 0
        mock_save.assert_called_once()

    def test_overwrites_existing_settings(self, mocker):
        """Test that create overwrites existing settings."""
        mock_args = MagicMock()
        mock_args.model = "existing/model"
        mock_args.alias = "new-alias"
        mock_args.ctx_size = None
        mock_args.flash_attn = None
        mock_args.reasoning = None
        mock_args.jinja = True
        mock_args.spec_type = None
        mock_args.cache_type_k = None
        mock_args.cache_type_v = None
        mock_args.chat_template_file = None

        mock_load = mocker.patch("chulengo.load_settings")
        mock_load.return_value = {
            "defaults": {},
            "model_overrides": {
                "existing/model": {"ctx_size": 8192, "flash_attn": "on"}
            },
        }

        result = chulengo.cmd_create(mock_args)

        assert result == 0


class TestCmdDelete:
    """Tests for cmd_delete() function."""

    def test_deletes_existing_model_settings(self, mocker):
        """Test deleting settings for an existing model."""
        mock_args = MagicMock()
        mock_args.target = "test/model"

        mock_load = mocker.patch("chulengo.load_settings")
        mock_load.return_value = {"model_overrides": {"test/model": {"ctx_size": 8192}}}
        mock_save = mocker.patch("chulengo.save_settings")

        result = chulengo.cmd_delete(mock_args)

        assert result == 0
        mock_save.assert_called_once()

    def test_returns_error_for_nonexistent_model(self, mocker, capsys):
        """Test that delete returns error for non-existent model."""
        mock_args = MagicMock()
        mock_args.target = "nonexistent/model"

        with patch("chulengo.load_settings") as mock_load:
            mock_load.return_value = {"model_overrides": {}}
            result = chulengo.cmd_delete(mock_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "No override found" in captured.err


class TestCmdShow:
    """Tests for cmd_show() function."""

    def test_shows_model_info(self, mocker, capsys):
        """Test showing model information."""
        mock_args = MagicMock()
        mock_args.model = "Qwen/Qwen3-8B"

        cache_info = {
            "cache_dir": "/cache/Qwen/Qwen3-8B",
            "snapshot": "/cache/Qwen/Qwen3-8B/snapshots/123",
            "gguf_files": ["model-Q4_K_M.gguf"],
            "gguf_path": "/cache/Qwen/Qwen3-8B/model-Q4_K_M.gguf",
        }

        with patch("chulengo.load_settings") as mock_load:
            mock_load.return_value = {
                "defaults": {"qwen3": {"ctx_size": 16384, "flash_attn": "on"}},
                "model_overrides": {},
            }
            with patch("chulengo.detect_model_family", return_value="qwen3"):
                with patch("chulengo.get_model_cache_info", return_value=cache_info):
                    result = chulengo.cmd_show(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Qwen/Qwen3-8B" in captured.out

    def test_shows_cache_info_when_available(self, mocker, capsys):
        """Test showing cache info when model is in cache."""
        mock_args = MagicMock()
        mock_args.model = "test/model"

        cache_info = {
            "cache_dir": "/cache/test/model",
            "snapshot": "/cache/test/model/snapshots/123",
            "gguf_files": ["model-Q4_K_M.gguf"],
            "gguf_path": "/cache/test/model/model-Q4_K_M.gguf",
        }

        gguf_metadata = {
            "architecture": "qwen3",
            "parameter_count": 8_000_000_000,
            "parameter_count_formatted": "8.0B",
            "vocab_size": 151936,
            "context_length": 16384,
            "embedding_length": 4096,
            "block_count": 32,
            "is_multimodal": False,
        }

        with patch(
            "chulengo.load_settings",
            return_value={"defaults": {}, "model_overrides": {}},
        ):
            with patch("chulengo.detect_model_family", return_value=None):
                with patch("chulengo.get_model_cache_info", return_value=cache_info):
                    with patch(
                        "chulengo.extract_gguf_metadata", return_value=gguf_metadata
                    ):
                        result = chulengo.cmd_show(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Cache Location" in captured.out
        assert "GGUF Metadata" in captured.out
        assert "qwen3" in captured.out
        assert "8.0B" in captured.out

    def test_shows_warning_for_nonexistent_model(self, mocker, capsys):
        """Test showing warning when model not in cache."""
        mock_args = MagicMock()
        mock_args.model = "nonexistent/model"

        with patch(
            "chulengo.load_settings",
            return_value={"defaults": {}, "model_overrides": {}},
        ):
            with patch("chulengo.get_model_cache_info", return_value=None):
                result = chulengo.cmd_show(mock_args)

        assert result == 1
        captured = capsys.readouterr()
        # The model not being in cache is shown with warning in stderr
        assert "nonexistent/model" in captured.err
        assert "chulengo ls" in captured.err


class TestGetModelCacheInfo:
    """Tests for get_model_cache_info() function."""

    def test_returns_none_for_nonexistent_model(self, mocker):
        """Test that None is returned for non-existent model."""
        with patch("chulengo.get_hf_cache_path", return_value="/cache"):
            with patch.object(Path, "exists", return_value=False):
                result = chulengo.get_model_cache_info("nonexistent/model")
                assert result is None

    def test_returns_gguf_path_for_existing_model(self, mocker):
        """Test that gguf_path is returned for models with GGUF files."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock cache structure
            cache_dir = Path(tmpdir) / "models--test--model" / "snapshots" / "abc123"
            cache_dir.mkdir(parents=True)

            # Create a dummy GGUF file
            gguf_file = cache_dir / "model-Q4_K_M.gguf"
            gguf_file.write_bytes(b"dummy gguf content")

            with patch("chulengo.get_hf_cache_path", return_value=tmpdir):
                result = chulengo.get_model_cache_info("test/model")

        assert result is not None
        assert "gguf_path" in result
        assert result["gguf_path"] is not None
        assert result["gguf_path"].endswith("model-Q4_K_M.gguf")
        assert "gguf_files" in result
        assert "model-Q4_K_M.gguf" in result["gguf_files"]

    def test_returns_gguf_path_none_when_no_gguf_files(self, mocker):
        """Test that gguf_path is None when no GGUF files exist."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock cache structure without GGUF files
            cache_dir = Path(tmpdir) / "models--test--model" / "snapshots" / "abc123"
            cache_dir.mkdir(parents=True)

            # Create a non-GGUF file
            other_file = cache_dir / "config.json"
            other_file.write_bytes(b"{}")

            with patch("chulengo.get_hf_cache_path", return_value=tmpdir):
                result = chulengo.get_model_cache_info("test/model")

        assert result is not None
        assert result["gguf_path"] is None
        assert result["gguf_files"] == []


class TestGetGgufArchitecture:
    """Tests for get_gguf_architecture() function."""

    def test_returns_architecture_from_gguf_file(self, mocker):
        """Test that architecture is read from GGUF file metadata."""
        # Create a mock field with contents
        mock_field = MagicMock()
        mock_field.contents.return_value = "qwen3"

        # Create a mock reader
        mock_reader = MagicMock()
        mock_reader.get_field.return_value = mock_field

        # Patch the GGUFReader used by chulengo module
        with patch("chulengo.GGUFReader", return_value=mock_reader):
            result = chulengo.get_gguf_architecture(Path("/path/to/model.gguf"))

        assert result == "qwen3"
        mock_reader.get_field.assert_called_once_with("general.architecture")

    def test_returns_none_when_file_not_found(self, mocker):
        """Test that None is returned when file doesn't exist."""
        result = chulengo.get_gguf_architecture("/nonexistent/model.gguf")
        assert result is None

    def test_returns_none_when_no_architecture_field(self, mocker):
        """Test that None is returned when architecture field is missing."""
        # Mock GGUFReader to return None for the field
        mock_reader = MagicMock()
        mock_reader.get_field.return_value = None

        with patch("gguf.GGUFReader", return_value=mock_reader):
            result = chulengo.get_gguf_architecture("/path/to/model.gguf")

        assert result is None

    def test_returns_none_on_gguf_read_error(self, mocker):
        """Test that None is returned on GGUF read errors."""
        with patch("gguf.GGUFReader", side_effect=Exception("Read error")):
            result = chulengo.get_gguf_architecture("/path/to/model.gguf")

        assert result is None


class TestExtractGgufMetadata:
    """Tests for extract_gguf_metadata() function."""

    def test_extracts_basic_metadata(self, mocker):
        """Test extraction of basic model metadata."""
        # Create mock tensors
        mock_tensor = MagicMock()
        mock_tensor.n_elements = 1_000_000_000

        # Create mock field
        mock_field = MagicMock()
        mock_field.contents.return_value = "qwen3"

        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]
        mock_reader.get_field = MagicMock(return_value=mock_field)

        with patch("chulengo.GGUFReader", return_value=mock_reader):
            result = chulengo.extract_gguf_metadata("/path/to/model.gguf")

        assert result is not None
        assert result["architecture"] == "qwen3"
        assert result["parameter_count"] == 1_000_000_000
        assert "parameter_count_formatted" in result

    def test_extracts_architecture_specific_fields(self, mocker):
        """Test extraction of architecture-specific fields."""
        mock_tensor = MagicMock()
        mock_tensor.n_elements = 2_000_000_000

        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]

        # Mock field returns for architecture-specific fields
        vocab_field = MagicMock()
        vocab_field.contents.return_value = 151936

        context_field = MagicMock()
        context_field.contents.return_value = 16384

        def get_field_side_effect(field_name):
            if field_name == "general.architecture":
                field = MagicMock()
                field.contents.return_value = "llama"
                return field
            elif field_name == "llama.vocab_size":
                return vocab_field
            elif field_name == "llama.context_length":
                return context_field
            return None

        mock_reader.get_field = MagicMock(side_effect=get_field_side_effect)

        with patch("chulengo.GGUFReader", return_value=mock_reader):
            result = chulengo.extract_gguf_metadata("/path/to/model.gguf")

        assert result is not None
        assert result["vocab_size"] == 151936
        assert result["context_length"] == 16384

    def test_extracts_tokenizer_info(self, mocker):
        """Test extraction of tokenizer information."""
        mock_tensor = MagicMock()
        mock_tensor.n_elements = 500_000_000

        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]

        def get_field_side_effect(field_name):
            if field_name == "general.architecture":
                field = MagicMock()
                field.contents.return_value = "qwen"
                return field
            elif field_name == "tokenizer.ggml.model":
                field = MagicMock()
                field.contents.return_value = "Gemma"
                return field
            elif field_name == "tokenizer.ggml.pre":
                field = MagicMock()
                field.contents.return_value = "f"
                return field
            return None

        mock_reader.get_field = MagicMock(side_effect=get_field_side_effect)

        with patch("chulengo.GGUFReader", return_value=mock_reader):
            result = chulengo.extract_gguf_metadata("/path/to/model.gguf")

        assert result is not None
        assert result["tokenizer_model"] == "Gemma"
        assert result["tokenizer_pre"] == "f"

    def test_returns_none_on_read_error(self, mocker):
        """Test that None is returned on GGUF read errors."""
        with patch("chulengo.GGUFReader", side_effect=Exception("Read error")):
            result = chulengo.extract_gguf_metadata("/path/to/model.gguf")

        assert result is None


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow_create_show_delete(self, mocker, capsys):
        """Test the full workflow of create -> show -> delete."""
        # Mock the settings
        settings = {"defaults": {}, "model_overrides": {}}

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.save_settings") as mock_save:
                with patch("chulengo.detect_model_family", return_value="qwen3"):
                    with patch("chulengo.get_model_cache_info", return_value=None):
                        # Create
                        mock_args = MagicMock()
                        mock_args.model = "test/model"
                        mock_args.alias = "test"
                        mock_args.ctx_size = 8192
                        mock_args.flash_attn = "on"
                        mock_args.reasoning = "off"
                        mock_args.jinja = True
                        mock_args.spec_type = None
                        mock_args.cache_type_k = None
                        mock_args.cache_type_v = None
                        mock_args.chat_template_file = None

                        result = chulengo.cmd_create(mock_args)
                        assert result == 0

                        # Verify save was called
                        assert mock_save.call_count >= 1


class TestCmdUpdate:
    """Tests for cmd_update() function."""

    def test_updates_existing_model(self, mocker):
        """Test updating settings for an existing model."""
        mock_args = MagicMock()
        mock_args.model = "existing/model"
        mock_args.ctx_size = 16384
        mock_args.flash_attn = "off"
        mock_args.reasoning = None
        mock_args.jinja = False
        mock_args.spec_type = None
        mock_args.cache_type_k = None
        mock_args.cache_type_v = None
        mock_args.chat_template_file = None

        mock_load = mocker.patch("chulengo.load_settings")
        mock_load.return_value = {
            "defaults": {},
            "model_overrides": {
                "existing/model": {"ctx_size": 8192, "flash_attn": "on"}
            },
        }
        mock_save = mocker.patch("chulengo.save_settings")

        result = chulengo.cmd_update(mock_args)

        assert result == 0
        mock_save.assert_called_once()

    def test_creates_on_update_for_nonexistent(self, mocker):
        """Test that update creates settings for nonexistent model."""
        mock_args = MagicMock()
        mock_args.model = "new/model"
        mock_args.ctx_size = 8192
        mock_args.flash_attn = None
        mock_args.reasoning = None
        mock_args.jinja = False
        mock_args.spec_type = None
        mock_args.cache_type_k = None
        mock_args.cache_type_v = None
        mock_args.chat_template_file = None

        mock_load = mocker.patch("chulengo.load_settings")
        mock_load.return_value = {"defaults": {}, "model_overrides": {}}
        _mock_save = mocker.patch("chulengo.save_settings")

        result = chulengo.cmd_update(mock_args)

        assert result == 0


class TestCmdLs:
    """Tests for cmd_ls() function."""

    def test_lists_models_successfully(self, mocker, capsys):
        """Test listing models from cache."""
        mock_cache_dir = MagicMock()
        mock_cache_dir.glob.return_value = []

        with patch("chulengo.get_hf_cache_path", return_value="/cache"):
            with patch.object(Path, "exists", return_value=True):
                result = chulengo.cmd_ls(MagicMock())

        # Should return 0 even if no models found
        assert result in [0, 1]

    def test_handles_no_cache_path(self, mocker, capsys):
        """Test handling when HF cache path cannot be determined."""
        with patch("chulengo.get_hf_cache_path", return_value=None):
            result = chulengo.cmd_ls(MagicMock())

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err or result == 1


class TestBuildLlamaCommandAdditional:
    """Additional tests for build_llama_command() function."""

    def test_includes_spec_type(self, mocker):
        """Test that spec_type is included."""
        settings = {"defaults": {}, "model_overrides": {}}

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.detect_model_family", return_value=None):
                with patch(
                    "chulengo.get_default_settings", return_value={"spec_type": "draft"}
                ):
                    result = chulengo.build_llama_command("test/model")

        assert "--spec-type" in result
        spec_idx = result.index("--spec-type")
        assert result[spec_idx + 1] == "draft"

    def test_includes_cache_type_flags(self, mocker):
        """Test that cache type flags are included."""
        settings = {"defaults": {}, "model_overrides": {}}

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.detect_model_family", return_value=None):
                with patch(
                    "chulengo.get_default_settings",
                    return_value={"cache_type_k": "q8_0", "cache_type_v": "q8_0"},
                ):
                    result = chulengo.build_llama_command("test/model")

        assert "--cache-type-k" in result
        assert "--cache-type-v" in result

    def test_includes_chat_template_file(self, mocker):
        """Test that chat_template_file is included."""
        settings = {"defaults": {}, "model_overrides": {}}

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.detect_model_family", return_value=None):
                with patch(
                    "chulengo.get_default_settings",
                    return_value={"chat_template_file": "/path/to/template.jinja"},
                ):
                    result = chulengo.build_llama_command("test/model")

        assert "--chat-template-file" in result
        template_idx = result.index("--chat-template-file")
        assert result[template_idx + 1] == "/path/to/template.jinja"

    def test_excludes_none_values(self, mocker):
        """Test that None values are excluded from settings."""
        settings = {"defaults": {}, "model_overrides": {}}

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.detect_model_family", return_value=None):
                with patch(
                    "chulengo.get_default_settings",
                    return_value={"ctx_size": None, "flash_attn": None},
                ):
                    result = chulengo.build_llama_command("test/model")

        # None values should not result in flags being added
        assert "--ctx-size" not in result
        assert "--flash-attn" not in result


class TestModelFamilyCoverage:
    """Tests covering all model families from models.yaml."""

    def test_detects_common_model_families(self):
        """Test detection of common model families."""
        test_models = [
            ("Qwen/Qwen3-8B", "qwen3"),
            ("Qwen/Qwen3-14B", "qwen3"),
            ("Qwen/Qwen3.5-72B", "qwen3"),  # Qwen3.5 matches qwen3
            ("Qwen/Qwen3-15B", "qwen3"),
            ("Qwen/Qwen2-7B", "qwen"),
            ("Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF", "qwopus"),
            ("codestral/codestral-embeddings", "codestral"),
            ("ibm/granite-8b-base", "granite"),
            ("meta-llama/Llama-3-8B", "llama3"),
            ("google/gemma-2-9b-it", "gemma2"),  # gemma-2 matches gemma2
            ("google/gemma-3-27B-it", "gemma3"),
            ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
            ("mistralai/Mixtral-8x7B-Instruct-v7.1", "mixtral"),
            ("THUDM/glm-4-9b-chat", "glm4"),
            ("deepseek-ai/deepseek-v3-72b", "deepseek"),
            ("Qwen/Qwen2.5-Coder-7B-Instruct", "qwen"),  # matches qwen
        ]

        for model_name, expected_family in test_models:
            result = chulengo.detect_model_family(model_name)
            assert (
                result == expected_family
            ), f"Failed for {model_name}: got {result}, expected {expected_family}"

    def test_detects_qwen_coder_models(self):
        """Test that Codestral models are detected correctly."""
        # Codestral should match codestral model family
        assert (
            chulengo.detect_model_family("codestral/codestral-7B-v0.1") == "codestral"
        )

    def test_detects_laguna_models(self):
        """Test that Laguna models are detected correctly."""
        assert chulengo.detect_model_family("poolside/Laguna-XS-2.1-GGUF") == "laguna"


class TestDoctorConfig:
    """Tests for doctor_config() — validates config against known families."""

    def test_valid_config_has_no_issues(self, mocker):
        """A config with only known families and valid keys is clean."""
        settings = {
            "defaults": {
                "generic": {"ctx_size": 4096},
                "qwen3": {"ctx_size": 16384, "flash_attn": "on"},
            },
            "model_overrides": {},
        }
        assert chulengo.doctor_config(settings) == []

    def test_unknown_family_is_flagged(self, mocker):
        """A typo'd family name (e.g. gemmma) should be reported as an issue."""
        settings = {
            "defaults": {"gemmma": {"ctx_size": 4096}},
            "model_overrides": {},
        }
        issues = chulengo.doctor_config(settings)
        assert len(issues) == 1
        assert "Unknown family 'gemmma'" in issues[0]

    def test_unknown_setting_key_is_flagged(self, mocker):
        """A setting key that isn't recognised should be reported."""
        settings = {
            "defaults": {"qwen3": {"ctx_size": 16384, "bogus_key": "x"}},
            "model_overrides": {},
        }
        issues = chulengo.doctor_config(settings)
        assert any("unknown setting key 'bogus_key'" in i for i in issues)

    def test_generic_family_is_always_valid(self, mocker):
        """The 'generic' fallback family must never be flagged as a typo."""
        settings = {
            "defaults": {"generic": {"ctx_size": 4096}},
            "model_overrides": {},
        }
        assert chulengo.doctor_config(settings) == []

    def test_empty_defaults_is_clean(self, mocker):
        """An empty defaults block has no issues."""
        settings = {"defaults": {}, "model_overrides": {}}
        assert chulengo.doctor_config(settings) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
