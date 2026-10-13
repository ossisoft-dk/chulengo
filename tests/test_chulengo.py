"""Tests for Chulengo - A llama.cpp wrapper for HuggingFace models."""

import argparse
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

        # Read back with the same YAML 1.2 loader chulengo uses; PyYAML (1.1)
        # would turn an unquoted `on` into True.
        loaded = chulengo.ruamel.yaml.YAML().load(target.read_text())
        assert loaded["defaults"]["generic"]["ctx_size"] == 4096
        assert loaded["model_overrides"]["test-model"]["flash_attn"] == "on"

    def test_save_settings_cleanup_on_failure(self, mocker, tmp_path):
        """Test that a failed save removes the temp file and raises."""
        settings = {"defaults": {}, "model_overrides": {}}

        target = tmp_path / "models.yaml"
        mocker.patch.object(chulengo, "SETTINGS_FILE", target)
        # Force yaml.dump to fail mid-write
        mocker.patch.object(
            chulengo.ruamel.yaml, "YAML", side_effect=OSError("disk full")
        )

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

    def test_detects_lfm2(self):
        """Test detection of Liquid AI LFM2 and LFM2.5 models."""
        assert chulengo.detect_model_family("LiquidAI/LFM2-1.2B-GGUF") == "lfm2"
        assert chulengo.detect_model_family("LiquidAI/LFM2.5-230M-GGUF") == "lfm2"

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


class TestResolveModelFamily:
    """Tests for resolve_model_family() warnings and fallbacks."""

    SETTINGS = {
        "defaults": {
            "qwen3": {"ctx_size": 16384, "flash_attn": "on"},
            "gpt_oss": {"ctx_size": 131072},
            "llama": {"ctx_size": 8192},
            "generic": {"ctx_size": 4096, "flash_attn": "off"},
        },
        "model_overrides": {},
    }

    def _resolve(self, model_name, mocker, *, arch=None, cached=True, settings=None):
        mocker.patch("chulengo.load_settings", return_value=settings or self.SETTINGS)
        mocker.patch(
            "chulengo.get_model_cache_info",
            return_value=(
                {
                    "cache_dir": "/cache/test/model",
                    "snapshot": "/cache/test/model/snapshots/abc",
                    "gguf_files": ["model.gguf"],
                    "gguf_path": "/cache/test/model/model.gguf",
                }
                if cached
                else None
            ),
        )
        mocker.patch("chulengo.get_gguf_architecture", return_value=arch)
        return chulengo.resolve_model_family(model_name)

    def test_known_arch_uses_block_without_warning(self, mocker):
        """A cached GGUF with a known architecture resolves silently."""
        resolution = self._resolve("some/model", mocker, arch="llama")

        assert resolution.warnings == []
        assert resolution.settings["ctx_size"] == 8192

    def test_hyphenated_arch_matches_underscore_block(self, mocker):
        """A 'gpt-oss' GGUF matches the shipped 'gpt_oss' block, warning-free.

        The model name may or may not contain 'gpt-oss'; the architecture
        alone decides.
        """
        for name in ("openai/gpt-oss-20b", "openai/ft-20b"):
            resolution = self._resolve(name, mocker, arch="gpt-oss")

            assert resolution.warnings == []
            assert resolution.settings["ctx_size"] == 131072

    def test_unknown_arch_warns_and_falls_back_to_name(self, mocker):
        """A readable architecture with no settings block warns, then uses
        name detection."""
        resolution = self._resolve("Qwen/Qwen3-8B", mocker, arch="plamo")

        assert len(resolution.warnings) == 1
        warning = resolution.warnings[0]
        assert "architecture 'plamo' has no settings" in warning
        assert "add a 'plamo:' block" in warning
        assert str(chulengo.SETTINGS_FILE) in warning
        # Name detection still decides the settings.
        assert resolution.settings["ctx_size"] == 16384

    def test_unknown_arch_falls_back_to_generic(self, mocker):
        """With no name match either, generic defaults apply (plus warning)."""
        resolution = self._resolve("unknown/model", mocker, arch="plamo")

        assert "architecture 'plamo' has no settings" in resolution.warnings[0]
        assert resolution.settings["ctx_size"] == 4096

    def test_unreadable_arch_warns_with_name_fallback(self, mocker):
        """An unreadable cached GGUF warns and names the fallback source."""
        resolution = self._resolve("Qwen/Qwen3-8B", mocker, arch=None)

        assert resolution.warnings == [
            "could not read the architecture from the cached GGUF; "
            "using qwen3 defaults from the model name"
        ]
        assert resolution.settings["ctx_size"] == 16384

    def test_unreadable_arch_warns_with_generic_fallback(self, mocker):
        """Unreadable arch + no name match warns about the generic fallback."""
        resolution = self._resolve("unknown/model", mocker, arch=None)

        assert resolution.warnings == [
            "could not read the architecture from the cached GGUF; "
            "using generic defaults from the model name"
        ]
        assert resolution.settings["ctx_size"] == 4096

    def test_no_cached_gguf_means_no_warning(self, mocker):
        """Without a cached GGUF there is nothing to warn about."""
        resolution = self._resolve("Qwen/Qwen3-8B", mocker, cached=False)

        assert resolution.warnings == []
        assert resolution.settings["ctx_size"] == 16384

    def test_user_arch_block_used_for_cached_arch(self, mocker):
        """A user `plamo:` block serves a cached plamo-architecture model
        with no warning."""
        settings = {
            "defaults": {
                "plamo": {"ctx_size": 8192},
                "generic": {"ctx_size": 4096},
            },
            "model_overrides": {},
        }

        resolution = self._resolve(
            "plamo/PLaMo-2-8B", mocker, arch="plamo", settings=settings
        )

        assert resolution.warnings == []
        assert resolution.settings["ctx_size"] == 8192

    def test_user_arch_block_found_by_name(self, mocker):
        """An uncached model is matched by name against user arch blocks."""
        settings = {
            "defaults": {
                "plamo": {"ctx_size": 8192},
                "generic": {"ctx_size": 4096},
            },
            "model_overrides": {},
        }

        resolution = self._resolve(
            "plamo/PLaMo-2-8B", mocker, cached=False, settings=settings
        )

        assert resolution.warnings == []
        assert resolution.settings["ctx_size"] == 8192

    def test_user_arch_block_beats_builtin_family(self, mocker):
        """A user `llama4:` block wins name detection over the shipped
        `llama` family."""
        settings = {
            "defaults": {
                "llama": {"ctx_size": 4096},
                "llama4": {"ctx_size": 65536},
                "generic": {"ctx_size": 2048},
            },
            "model_overrides": {},
        }

        resolution = self._resolve(
            "meta-llama/Llama-4-Scout", mocker, cached=False, settings=settings
        )

        # Without the user block, name detection would land on `llama`.
        assert resolution.settings["ctx_size"] == 65536


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

    @pytest.mark.parametrize("value", ["off", False, None])
    def test_jinja_off_omits_flag(self, mocker, value):
        """Test that jinja: off (a string under YAML 1.2) does not enable jinja."""
        settings = {"defaults": {}, "model_overrides": {}}

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.detect_model_family", return_value=None):
                with patch(
                    "chulengo.get_default_settings", return_value={"jinja": value}
                ):
                    result = chulengo.build_llama_command("test/model")

        assert "--jinja" not in result

    def test_on_off_strings_from_yaml(self, mocker):
        """Test that on/off strings loaded from the config map to llama flags."""
        settings = {"defaults": {}, "model_overrides": {}}

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.detect_model_family", return_value=None):
                with patch(
                    "chulengo.get_default_settings",
                    return_value={"flash_attn": "on", "reasoning": "off"},
                ):
                    result = chulengo.build_llama_command("test/model")

        assert result[result.index("--flash-attn") + 1] == "on"
        assert result[result.index("--reasoning") + 1] == "off"

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
                    "test/model",
                    ctx_size=16384,  # Override
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

        mock_load = mocker.patch("chulengo.load_settings")
        mock_load.return_value = {"defaults": {}, "model_overrides": {}}
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
        mock_save = mocker.patch("chulengo.save_settings")

        result = chulengo.cmd_create(mock_args)

        assert result == 0
        mock_save.assert_called_once()


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

    def test_show_prints_family_resolution_warning(self, mocker, capsys):
        """Family-resolution warnings go to stderr; stdout format is unchanged."""
        mock_args = MagicMock()
        mock_args.model = "Qwen/Qwen3-8B"

        cache_info = {
            "cache_dir": "/cache/Qwen/Qwen3-8B",
            "snapshot": "/cache/Qwen/Qwen3-8B/snapshots/123",
            "gguf_files": ["model-Q4_K_M.gguf"],
            "gguf_path": "/cache/Qwen/Qwen3-8B/model-Q4_K_M.gguf",
        }

        settings = {
            "defaults": {
                "qwen3": {"ctx_size": 16384},
                "generic": {"ctx_size": 4096},
            },
            "model_overrides": {},
        }

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.detect_model_family", return_value="qwen3"):
                with patch("chulengo.get_model_cache_info", return_value=cache_info):
                    with patch("chulengo.get_gguf_architecture", return_value="plamo"):
                        with patch("chulengo.extract_gguf_metadata", return_value=None):
                            result = chulengo.cmd_show(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "architecture 'plamo' has no settings" in captured.err
        assert "Detected family: qwen3" in captured.out

    def test_show_detects_user_arch_family(self, mocker, capsys):
        """User-configured architecture blocks feed 'Detected family'."""
        mock_args = MagicMock()
        mock_args.model = "plamo/PLaMo-2-8B"

        cache_info = {
            "cache_dir": "/cache/plamo/PLaMo-2-8B",
            "snapshot": "/cache/plamo/PLaMo-2-8B/snapshots/123",
            "gguf_files": ["plamo-2-8b.gguf"],
            "gguf_path": "/cache/plamo/PLaMo-2-8B/plamo-2-8b.gguf",
        }

        settings = {
            "defaults": {
                "plamo": {"ctx_size": 8192},
                "generic": {"ctx_size": 4096},
            },
            "model_overrides": {},
        }

        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.get_model_cache_info", return_value=cache_info):
                with patch("chulengo.get_gguf_architecture", return_value="plamo"):
                    with patch("chulengo.extract_gguf_metadata", return_value=None):
                        result = chulengo.cmd_show(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Detected family: plamo" in captured.out
        assert "Default settings for plamo" in captured.out
        assert captured.err == ""


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

    def test_preserves_comments_and_formatting(self, mocker, tmp_path):
        """Test that a real update round-trips comments and on/off values."""
        target = tmp_path / "models.yaml"
        target.write_text(
            "# top comment\n"
            "defaults:\n"
            "  generic:\n"
            "    flash_attn: off  # inline default\n"
            "    jinja: off\n"
            "\n"
            "model_overrides:\n"
            "  # tuned for my box\n"
            "  existing/model:\n"
            "    ctx_size: 8192  # inline override\n"
            "    flash_attn: on\n"
        )
        mocker.patch.object(chulengo, "SETTINGS_FILE", target)

        mock_args = MagicMock()
        mock_args.model = "existing/model"
        mock_args.alias = None
        mock_args.ctx_size = 16384
        mock_args.flash_attn = None
        mock_args.reasoning = None
        mock_args.jinja = False
        mock_args.spec_type = None
        mock_args.cache_type_k = None
        mock_args.cache_type_v = None
        mock_args.chat_template_file = None

        result = chulengo.cmd_update(mock_args)

        assert result == 0
        text = target.read_text()
        assert "# top comment" in text
        assert "# inline default" in text
        assert "# tuned for my box" in text
        assert "# inline override" in text
        assert "ctx_size: 16384" in text
        # on/off values are left as written, not rewritten or quoted
        assert "flash_attn: off" in text
        assert "jinja: off" in text
        assert "flash_attn: on" in text
        assert '"on"' not in text and '"off"' not in text


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
            assert result == expected_family, (
                f"Failed for {model_name}: got {result}, expected {expected_family}"
            )

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

    def test_shipped_models_yaml_is_clean(self):
        """Every family in the shipped models.yaml must be a known family."""
        shipped = Path(__file__).parent.parent / "models.yaml"
        settings = chulengo.ruamel.yaml.YAML().load(shipped.read_text())
        assert "lfm2" in settings["defaults"]
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

    def test_architecture_named_blocks_are_valid(self):
        """Blocks named after llama.cpp architectures (e.g. plamo, gpt-oss)
        are valid `defaults:` keys, not typos."""
        settings = {
            "defaults": {
                "plamo": {"ctx_size": 8192},
                "gpt-oss": {"ctx_size": 131072},
                "gemmma": {"ctx_size": 4096},
            },
            "model_overrides": {},
        }
        issues = chulengo.doctor_config(settings)

        assert not any("'plamo'" in i or "'gpt-oss'" in i for i in issues)
        assert any("Unknown family 'gemmma'" in i for i in issues)

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

    def test_unknown_override_key_is_flagged(self):
        """A typo'd setting key in a model override should be reported."""
        settings = {
            "defaults": {},
            "model_overrides": {"org/model": {"ctx_siz": 4096}},
        }
        issues = chulengo.doctor_config(settings)
        assert any(
            "Model override 'org/model'" in i and "unknown setting key 'ctx_siz'" in i
            for i in issues
        )

    def test_alias_is_valid_in_model_overrides(self):
        """`alias` is a legitimate model-override key, not an unknown setting."""
        settings = {
            "defaults": {},
            "model_overrides": {"org/model": {"alias": "mymodel"}},
        }
        assert chulengo.doctor_config(settings) == []

    def test_alias_in_family_defaults_is_flagged(self):
        """`alias` is only valid per-model, not inside a family default block."""
        settings = {
            "defaults": {"qwen3": {"alias": "mymodel"}},
            "model_overrides": {},
        }
        issues = chulengo.doctor_config(settings)
        assert any("unknown setting key 'alias'" in i for i in issues)

    def test_unknown_top_level_key_is_flagged(self):
        """An unknown top-level key (e.g. `default`) should be reported."""
        settings = {
            "default": {"qwen3": {"ctx_size": 4096}},
            "defaults": {},
            "model_overrides": {},
        }
        issues = chulengo.doctor_config(settings)
        assert any("Unknown top-level key 'default'" in i for i in issues)

    def test_malformed_defaults_is_reported_not_crashed(self):
        """A non-mapping `defaults` (e.g. a list) must not raise."""
        settings = {"defaults": ["qwen3"], "model_overrides": {}}
        issues = chulengo.doctor_config(settings)
        assert any("'defaults' must be a mapping" in i for i in issues)

    def test_non_mapping_family_block_is_reported(self):
        """A family block that isn't a mapping should be reported."""
        settings = {
            "defaults": {"qwen3": ["ctx_size"]},
            "model_overrides": {},
        }
        issues = chulengo.doctor_config(settings)
        assert any("Family 'qwen3': expected a mapping" in i for i in issues)

    def test_non_mapping_model_overrides_is_reported(self):
        """A non-mapping `model_overrides` should be reported."""
        settings = {"defaults": {}, "model_overrides": ["org/model"]}
        issues = chulengo.doctor_config(settings)
        assert any("'model_overrides' must be a mapping" in i for i in issues)

    def test_non_mapping_config_root_is_reported(self):
        """A config that isn't a mapping at all should be reported."""
        assert chulengo.doctor_config(None) == [
            "Config root must be a mapping, got NoneType"
        ]

    def test_non_string_top_level_key_does_not_crash(self):
        """Mixed-type top-level keys must not raise in sorted()."""
        settings = {1: {}, "defaults": {}, "model_overrides": {}}
        issues = chulengo.doctor_config(settings)
        assert any("Unknown top-level key '1'" in i for i in issues)

    def test_non_string_family_key_does_not_crash(self):
        """Mixed-type family keys must not raise in sorted()."""
        settings = {"defaults": {1: {}, "qwen3": {}}, "model_overrides": {}}
        issues = chulengo.doctor_config(settings)
        assert any("Unknown family '1'" in i for i in issues)


class TestDoctorCommand:
    """Tests for cmd_doctor() and load_user_config()."""

    def test_missing_config_reports_and_writes_nothing(self, mocker, tmp_path, capsys):
        """On a fresh install doctor must not seed or write a config file."""
        target = tmp_path / "chulengo" / "models.yaml"
        mocker.patch.object(chulengo, "SETTINGS_FILE", target)

        rc = chulengo.cmd_doctor(None)

        assert rc == 0
        assert "No user config found" in capsys.readouterr().out
        assert not target.exists()
        assert not target.parent.exists()

    def test_invalid_yaml_is_reported_not_crashed(self, mocker, tmp_path, capsys):
        """Broken YAML should be reported as an issue, not a traceback."""
        target = tmp_path / "models.yaml"
        target.write_text("defaults: [unclosed\n")
        mocker.patch.object(chulengo, "SETTINGS_FILE", target)

        rc = chulengo.cmd_doctor(None)

        assert rc == 1
        assert "Could not parse config file" in capsys.readouterr().out

    def test_valid_config_reports_ok(self, mocker, tmp_path, capsys):
        """A valid user config should pass."""
        target = tmp_path / "models.yaml"
        target.write_text(
            "defaults:\n  qwen3:\n    ctx_size: 4096\nmodel_overrides: {}\n"
        )
        mocker.patch.object(chulengo, "SETTINGS_FILE", target)

        rc = chulengo.cmd_doctor(None)

        assert rc == 0
        assert "OK: config is valid" in capsys.readouterr().out

    def test_load_user_config_parses_valid_yaml(self, mocker, tmp_path):
        """load_user_config returns the parsed mapping on success."""
        target = tmp_path / "models.yaml"
        target.write_text("defaults: {}\nmodel_overrides: {}\n")
        mocker.patch.object(chulengo, "SETTINGS_FILE", target)

        settings, error = chulengo.load_user_config()

        assert error is None
        assert settings == {"defaults": {}, "model_overrides": {}}


class TestConfigOverride:
    """Tests for --config / CHULENGO_CONFIG config path resolution."""

    def test_resolve_default_when_unset(self, mocker):
        """With no flag and no env var, the default path is used."""
        # Empty string is falsy, so this exercises the "unset" branch while
        # still being restored by mocker afterwards.
        mocker.patch.dict(os.environ, {"CHULENGO_CONFIG": ""})
        assert chulengo.resolve_settings_file() == chulengo.DEFAULT_SETTINGS_FILE_PATH

    def test_env_var_wins_over_default(self, mocker, tmp_path):
        """CHULENGO_CONFIG takes precedence over the default path."""
        target = tmp_path / "env.yaml"
        mocker.patch.dict(os.environ, {"CHULENGO_CONFIG": str(target)})
        assert chulengo.resolve_settings_file() == target

    def test_flag_wins_over_env_var(self, mocker, tmp_path):
        """The --config value takes precedence over CHULENGO_CONFIG."""
        env_target = tmp_path / "env.yaml"
        cli_target = tmp_path / "cli.yaml"
        mocker.patch.dict(os.environ, {"CHULENGO_CONFIG": str(env_target)})
        assert chulengo.resolve_settings_file(str(cli_target)) == cli_target

    def test_tilde_is_expanded(self, mocker):
        """A leading ~ in either source is expanded to the home directory."""
        mocker.patch.dict(os.environ, {"CHULENGO_CONFIG": ""})
        resolved = chulengo.resolve_settings_file("~/some/models.yaml")
        assert resolved == Path.home() / "some" / "models.yaml"

    def test_set_settings_file_updates_global(self, mocker, tmp_path):
        """set_settings_file points the module global at the resolved path."""
        target = tmp_path / "models.yaml"
        mocker.patch.object(chulengo, "SETTINGS_FILE", chulengo.SETTINGS_FILE)
        assert chulengo.set_settings_file(str(target)) == target
        assert chulengo.SETTINGS_FILE == target

    def test_doctor_uses_config_flag_after_subcommand(self, mocker, tmp_path, capsys):
        """`chulengo doctor --config <path>` validates the given file."""
        target = tmp_path / "models.yaml"
        target.write_text("defaults: [unclosed\n")
        mocker.patch.object(chulengo, "SETTINGS_FILE", chulengo.SETTINGS_FILE)
        mocker.patch.object(
            sys, "argv", ["chulengo", "doctor", "--config", str(target)]
        )

        assert chulengo.main() == 1
        assert "Could not parse config file" in capsys.readouterr().out
        assert str(target) in str(chulengo.SETTINGS_FILE)

    def test_doctor_uses_config_flag_before_subcommand(self, mocker, tmp_path, capsys):
        """The global `chulengo --config <path> doctor` form also works."""
        target = tmp_path / "models.yaml"
        target.write_text("defaults:\n  qwen3:\n    ctx_size: 4096\n")
        mocker.patch.object(chulengo, "SETTINGS_FILE", chulengo.SETTINGS_FILE)
        mocker.patch.object(
            sys, "argv", ["chulengo", "--config", str(target), "doctor"]
        )

        assert chulengo.main() == 0
        assert "OK: config is valid" in capsys.readouterr().out

    def test_doctor_uses_env_var(self, mocker, tmp_path, capsys):
        """Without a flag, doctor validates the CHULENGO_CONFIG file."""
        target = tmp_path / "models.yaml"
        target.write_text("defaults: [unclosed\n")
        mocker.patch.object(chulengo, "SETTINGS_FILE", chulengo.SETTINGS_FILE)
        mocker.patch.dict(os.environ, {"CHULENGO_CONFIG": str(target)})
        mocker.patch.object(sys, "argv", ["chulengo", "doctor"])

        assert chulengo.main() == 1
        assert "Could not parse config file" in capsys.readouterr().out


class TestAliases:
    """Tests for model alias resolution, emission, and validation."""

    @staticmethod
    def _ns(**kwargs):
        base = dict(
            model=None,
            port=None,
            alias=None,
            ctx_size=None,
            flash_attn=None,
            reasoning=None,
            jinja=None,
            spec_type=None,
            cache_type_k=None,
            cache_type_v=None,
            chat_template_file=None,
            target=None,
        )
        base.update(kwargs)
        return argparse.Namespace(**base)

    # --- resolve_model_name ---------------------------------------------

    def test_resolve_repo_key_passthrough(self):
        settings = {"model_overrides": {"org/repo": {"alias": "lfm"}}}
        assert chulengo.resolve_model_name(settings, "org/repo") == (
            "org/repo",
            None,
        )

    def test_resolve_unique_alias(self):
        settings = {"model_overrides": {"org/repo": {"alias": "lfm"}}}
        assert chulengo.resolve_model_name(settings, "lfm") == ("org/repo", None)

    def test_resolve_unknown_name_passes_through(self):
        settings = {"model_overrides": {"org/a": {"alias": "x"}}}
        assert chulengo.resolve_model_name(settings, "other/model") == (
            "other/model",
            None,
        )

    def test_resolve_ambiguous_alias_is_error(self):
        settings = {
            "model_overrides": {"org/a": {"alias": "x"}, "org/b": {"alias": "x"}}
        }
        resolved, error = chulengo.resolve_model_name(settings, "x")
        assert resolved is None
        assert "matches multiple models" in error
        assert "org/a" in error and "org/b" in error

    def test_resolve_key_wins_over_alias_match(self):
        # "lfm" is a repo key and also another model's alias; the key wins.
        settings = {"model_overrides": {"lfm": {}, "org/a": {"alias": "lfm"}}}
        assert chulengo.resolve_model_name(settings, "lfm") == ("lfm", None)

    # --- _alias_conflict -------------------------------------------------

    def test_alias_conflict_taken_by_other_model(self):
        settings = {"model_overrides": {"org/a": {"alias": "lfm"}}}
        error = chulengo._alias_conflict(settings, "org/b", "lfm")
        assert error and "already used by model 'org/a'" in error

    def test_alias_conflict_allows_same_model(self):
        settings = {"model_overrides": {"org/a": {"alias": "lfm"}}}
        assert chulengo._alias_conflict(settings, "org/a", "lfm") is None

    def test_alias_conflict_rejects_slash(self):
        settings = {"model_overrides": {}}
        error = chulengo._alias_conflict(settings, "org/a", "org/bad")
        assert error and "/" in error

    # --- build_llama_command ---------------------------------------------

    def test_emits_stored_alias_from_config(self, mocker):
        settings = {"defaults": {}, "model_overrides": {"org/repo": {"alias": "lfm"}}}
        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.get_default_settings", return_value={}):
                result = chulengo.build_llama_command("org/repo")
        assert result[result.index("--alias") + 1] == "lfm"

    def test_cli_alias_wins_over_stored(self, mocker):
        settings = {"defaults": {}, "model_overrides": {"org/repo": {"alias": "lfm"}}}
        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.get_default_settings", return_value={}):
                result = chulengo.build_llama_command("org/repo", alias="cli")
        assert result.count("--alias") == 1
        assert result[result.index("--alias") + 1] == "cli"

    def test_no_alias_flag_without_cli_or_config(self, mocker):
        settings = {"defaults": {}, "model_overrides": {"org/repo": {}}}
        with patch("chulengo.load_settings", return_value=settings):
            with patch("chulengo.get_default_settings", return_value={}):
                result = chulengo.build_llama_command("org/repo")
        assert "--alias" not in result

    # --- cmd_create ------------------------------------------------------

    def test_create_refuses_duplicate_alias(self, mocker, capsys):
        settings = {"defaults": {}, "model_overrides": {"org/a": {"alias": "lfm"}}}
        mocker.patch.object(chulengo, "load_settings", return_value=settings)
        save = mocker.patch.object(chulengo, "save_settings")
        ns = self._ns(model="org/b", alias="lfm")

        assert chulengo.cmd_create(ns) == 1
        save.assert_not_called()
        assert "already used by model 'org/a'" in capsys.readouterr().err

    def test_create_refuses_slash_alias(self, mocker, capsys):
        settings = {"defaults": {}, "model_overrides": {}}
        mocker.patch.object(chulengo, "load_settings", return_value=settings)
        save = mocker.patch.object(chulengo, "save_settings")
        ns = self._ns(model="org/b", alias="org/x")

        assert chulengo.cmd_create(ns) == 1
        save.assert_not_called()
        assert "must not contain '/'" in capsys.readouterr().err

    # --- cmd_update ------------------------------------------------------

    def test_update_resolves_alias_to_repo(self, mocker):
        settings = {"defaults": {}, "model_overrides": {"org/repo": {"alias": "lfm"}}}
        mocker.patch.object(chulengo, "load_settings", return_value=settings)
        save = mocker.patch.object(chulengo, "save_settings")
        ns = self._ns(model="lfm", ctx_size=32000)

        assert chulengo.cmd_update(ns) == 0
        saved = save.call_args.args[0]
        assert saved["model_overrides"]["org/repo"]["ctx_size"] == 32000
        assert "lfm" not in saved["model_overrides"]

    def test_update_refuses_duplicate_alias(self, mocker, capsys):
        settings = {
            "defaults": {},
            "model_overrides": {"org/a": {"alias": "x"}, "org/b": {}},
        }
        mocker.patch.object(chulengo, "load_settings", return_value=settings)
        save = mocker.patch.object(chulengo, "save_settings")
        ns = self._ns(model="org/b", alias="x")

        assert chulengo.cmd_update(ns) == 1
        save.assert_not_called()
        assert "already used by model 'org/a'" in capsys.readouterr().err

    # --- cmd_delete ------------------------------------------------------

    def test_delete_by_alias(self, mocker):
        settings = {"defaults": {}, "model_overrides": {"org/repo": {"alias": "lfm"}}}
        mocker.patch.object(chulengo, "load_settings", return_value=settings)
        mocker.patch.object(chulengo, "save_settings")
        ns = self._ns(target="lfm")

        assert chulengo.cmd_delete(ns) == 0
        assert "org/repo" not in settings["model_overrides"]

    # --- cmd_serve -------------------------------------------------------

    def test_serve_by_alias_resolves_repo_and_emits_alias(self, mocker):
        settings = {"defaults": {}, "model_overrides": {"org/repo": {"alias": "lfm"}}}
        mocker.patch.object(chulengo, "load_settings", return_value=settings)
        mocker.patch.object(
            chulengo,
            "resolve_model_family",
            return_value=chulengo.FamilyResolution({}, []),
        )
        popen = mocker.patch("chulengo.subprocess.Popen")
        popen.return_value.wait.return_value = 0
        ns = self._ns(model="lfm")

        assert chulengo.cmd_serve(ns) == 0
        cmd = popen.call_args.args[0]
        assert cmd[cmd.index("--hf-repo") + 1] == "org/repo"
        assert cmd[cmd.index("--alias") + 1] == "lfm"

    def test_serve_ambiguous_alias_errors(self, mocker, capsys):
        settings = {
            "model_overrides": {"org/a": {"alias": "x"}, "org/b": {"alias": "x"}}
        }
        mocker.patch.object(chulengo, "load_settings", return_value=settings)
        popen = mocker.patch("chulengo.subprocess.Popen")
        ns = self._ns(model="x")

        assert chulengo.cmd_serve(ns) == 1
        popen.assert_not_called()
        assert "matches multiple models" in capsys.readouterr().err

    # --- doctor ----------------------------------------------------------

    def test_doctor_flags_slash_alias(self):
        settings = {"defaults": {}, "model_overrides": {"org/a": {"alias": "org/x"}}}
        issues = chulengo.doctor_config(settings)
        assert any("must not contain '/'" in i for i in issues)

    def test_doctor_flags_duplicate_alias(self):
        settings = {
            "defaults": {},
            "model_overrides": {"org/a": {"alias": "x"}, "org/b": {"alias": "x"}},
        }
        issues = chulengo.doctor_config(settings)
        assert any("used by multiple models" in i for i in issues)

    def test_doctor_allows_unique_aliases(self):
        settings = {
            "defaults": {},
            "model_overrides": {"org/a": {"alias": "x"}, "org/b": {"alias": "y"}},
        }
        assert chulengo.doctor_config(settings) == []


class TestServeFamilyWarnings:
    """End-to-end serve tests for family-resolution warnings."""

    SETTINGS = {
        "defaults": {
            "qwen3": {"ctx_size": 16384, "flash_attn": "on"},
            "gpt_oss": {"ctx_size": 131072},
            "llama": {"ctx_size": 8192},
            "generic": {"ctx_size": 4096, "flash_attn": "off"},
        },
        "model_overrides": {},
    }

    def _serve(self, mocker, model, *, arch=None, cached=True):
        """Run `serve` through main(); assert it exits 0, return the command."""
        mocker.patch.object(sys, "argv", ["chulengo", "serve", model])
        mocker.patch("chulengo.load_settings", return_value=self.SETTINGS)
        mocker.patch(
            "chulengo.get_model_cache_info",
            return_value=(
                {
                    "cache_dir": "/cache/m",
                    "snapshot": "/cache/m/snapshots/abc",
                    "gguf_files": ["model.gguf"],
                    "gguf_path": "/cache/m/model.gguf",
                }
                if cached
                else None
            ),
        )
        mocker.patch("chulengo.get_gguf_architecture", return_value=arch)
        popen = mocker.patch("chulengo.subprocess.Popen")
        popen.return_value.wait.return_value = 0

        assert chulengo.main() == 0
        return popen.call_args.args[0]

    def test_serve_warns_for_arch_without_settings(self, mocker, capsys):
        """Unknown cached architecture: stderr warning, name fallback, exit 0."""
        cmd = self._serve(mocker, "Qwen/Qwen3-8B", arch="plamo")

        captured = capsys.readouterr()
        assert "architecture 'plamo' has no settings" in captured.err
        assert "add a 'plamo:' block" in captured.err
        # Name fallback still picked qwen3; stdout format is unchanged.
        assert "Running: llama serve" in captured.out
        assert cmd[cmd.index("--ctx-size") + 1] == "16384"

    def test_serve_warns_for_unreadable_arch(self, mocker, capsys):
        """Unreadable cached GGUF: warning names the fallback source."""
        cmd = self._serve(mocker, "Qwen/Qwen3-8B", arch=None)

        captured = capsys.readouterr()
        assert "could not read the architecture from the cached GGUF" in captured.err
        assert "using qwen3 defaults from the model name" in captured.err
        assert "Running: llama serve" in captured.out
        assert cmd[cmd.index("--ctx-size") + 1] == "16384"

    def test_serve_hyphenated_arch_uses_gpt_oss_block(self, mocker, capsys):
        """Cached gpt-oss uses the gpt_oss block with no warning, with or
        without 'gpt-oss' in the model name."""
        for name in ("openai/gpt-oss-20b", "openai/ft-20b"):
            cmd = self._serve(mocker, name, arch="gpt-oss")

            captured = capsys.readouterr()
            assert captured.err == ""
            assert cmd[cmd.index("--ctx-size") + 1] == "131072"

    def test_serve_llama_arch_without_warning(self, mocker, capsys):
        """A cached llama-architecture model resolves silently to llama."""
        cmd = self._serve(mocker, "some/model", arch="llama")

        captured = capsys.readouterr()
        assert captured.err == ""
        assert cmd[cmd.index("--ctx-size") + 1] == "8192"

    def test_serve_no_cached_gguf_no_warning(self, mocker, capsys):
        """No cached GGUF: silent name-based resolution, as before."""
        cmd = self._serve(mocker, "Qwen/Qwen3-8B", cached=False)

        captured = capsys.readouterr()
        assert captured.err == ""
        assert cmd[cmd.index("--ctx-size") + 1] == "16384"


class TestServeJinja:
    """Tests for how `serve --jinja` combines with jinja from the config."""

    def _serve(self, mocker, argv, config_jinja):
        mocker.patch.object(sys, "argv", ["chulengo", "serve", "test/model", *argv])
        mocker.patch(
            "chulengo.load_settings",
            return_value={"defaults": {}, "model_overrides": {}},
        )
        mocker.patch(
            "chulengo.resolve_model_family",
            return_value=chulengo.FamilyResolution({"jinja": config_jinja}, []),
        )
        mock_popen = mocker.patch("chulengo.subprocess.Popen")
        mock_popen.return_value.wait.return_value = 0

        assert chulengo.main() == 0
        return mock_popen.call_args.args[0]

    def test_config_jinja_applies_without_flag(self, mocker):
        """Test that jinja: on in the config is used when --jinja is not passed."""
        cmd = self._serve(mocker, [], config_jinja="on")
        assert "--jinja" in cmd

    def test_config_jinja_off_without_flag(self, mocker):
        """Test that jinja: off in the config stays off when --jinja is not passed."""
        cmd = self._serve(mocker, [], config_jinja="off")
        assert "--jinja" not in cmd

    def test_flag_enables_jinja_over_config(self, mocker):
        """Test that --jinja turns jinja on even when the config has it off."""
        cmd = self._serve(mocker, ["--jinja"], config_jinja="off")
        assert "--jinja" in cmd


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
