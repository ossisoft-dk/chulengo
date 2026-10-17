#!/usr/bin/env python3
"""
Chulengo - A small and wild llama.cpp wrapper

A lightweight wrapper that simplifies serving HuggingFace models with llama.cpp
by applying sensible defaults based on model family.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

import ruamel.yaml
from gguf import GGUFReader
from gguf.constants import MODEL_ARCH_NAMES
from gguf.utility import model_weight_count_rounded_notation

# Default settings file path. The active path can be overridden per invocation
# with `--config` or the CHULENGO_CONFIG environment variable; see
# resolve_settings_file(). SETTINGS_FILE is a module global so existing callers
# keep using it, and main() points it at the resolved path before dispatch.
DEFAULT_SETTINGS_FILE_PATH = Path.home() / ".config" / "chulengo" / "models.yaml"
SETTINGS_FILE = DEFAULT_SETTINGS_FILE_PATH
DEFAULT_SETTINGS_FILE = Path(os.path.realpath(__file__)).parent / "models.yaml"


# Known model families. Kept as a module-level constant so family detection and
# config validation (chulengo doctor) share a single source of truth.
#
# Ordered most specific first so a shorter prefix never shadows a longer family
# name: mistral3/mistral4 before mistral, granitemoeshared before granitemoe
# before granite, qwencoder before qwen, llama3_2/llama3_1/llama3 before llama,
# qwen35moe before qwen35, ... Keep the `defaults:` blocks in models.yaml in the
# same order so the two are trivial to diff (see AGENTS.md "Family list").
KNOWN_FAMILIES = [
    "qwen35moe",
    "qwen35",
    "qwancoder",
    "qwopus",
    "qwencoder",
    "gpt_oss",
    "codestral",
    "llama3_2",
    "llama3_1",
    "llama3",
    "llama",
    "mixtral",
    "mistral3",
    "mistral4",
    "mistral",
    "qwen3",
    "qwen",
    "laguna",
    "glm4moe",
    "glm4",
    "gemma3",
    "gemma2",
    "gemma",
    "granitemoeshared",
    "granitemoe",
    "granite",
    "phi",
    "deepseek2",
    "deepseek",
    "nemotron_h",
    "nemotron",
    "lfm2",
]

# GGUF architectures whose block is shared by several shipped families. For
# these, the raw architecture cannot pick a single settings block by itself, so
# a cached model falls through to name-based selection against this ordered
# candidate list (most specific first, base last — see
# _select_shared_arch_family()). Only `llama` is shared today: Llama-3.x,
# Mistral, Mixtral and Codestral all report `llama` in their GGUF metadata.
#
# Keep families that have their OWN GGUF architecture out of this list:
# mistral3/mistral4 and qwen35/qwen35moe are real architectures with their own
# blocks, so listing them here would let the plain `mistral`/`llama` name
# prefix shadow them. Verified: codestral/mistral/mixtral are `llama`; phi-2
# reports `phi2` (reached by name, not by this table); qwopus/qwancoder/
# qwencoder/laguna/granitemoeshared have no `llama` arch and are not reliably
# reachable by name — the custom-family work in ROADMAP's Later section is
# their proper home.
#
# The model name includes the org, so codestral must come before mistral:
# "mistralai/Codestral-22B" contains "mistral" via the "mistralai" org.
LLAMA_ARCH_CANDIDATES = [
    "codestral",
    "llama3_2",
    "llama3_1",
    "llama3",
    "mixtral",
    "mistral",
    "llama",  # base fallback (never name-matched; always valid if it exists)
]

# Architectures llama.cpp knows about (GGUF general.architecture values).
# A `defaults:` block named after any of them is valid, so settings for a
# new architecture can be configured without waiting for a chulengo release.
BUILTIN_ARCHES = frozenset(MODEL_ARCH_NAMES.values())

# Valid keys under `defaults:`: known families, any llama.cpp architecture,
# and the always-valid generic fallback.
VALID_DEFAULT_KEYS = frozenset({"generic", *KNOWN_FAMILIES}) | BUILTIN_ARCHES

# Valid top-level keys in the config file.
VALID_TOP_LEVEL_KEYS = frozenset({"defaults", "model_overrides"})

# Valid setting keys inside a family block in the config.
VALID_SETTING_KEYS = frozenset(
    {
        "ctx_size",
        "flash_attn",
        "reasoning",
        "jinja",
        "spec_type",
        "cache_type_k",
        "cache_type_v",
        "chat_template_file",
    }
)

# Keys allowed inside a `model_overrides` entry. Model overrides may carry an
# `alias` (written by `create`/`update`), which is meaningless in a family
# default block, so this is a superset of VALID_SETTING_KEYS.
VALID_OVERRIDE_KEYS = VALID_SETTING_KEYS | frozenset({"alias"})


def resolve_settings_file(cli_path: str | None = None) -> Path:
    """Resolve which config file to use, without side effects.

    Precedence, highest first:

    1. ``cli_path`` (the ``--config`` flag)
    2. the ``CHULENGO_CONFIG`` environment variable
    3. the default ``~/.config/chulengo/models.yaml``

    Relative paths are left relative (resolved against the current working
    directory) and ``~`` is expanded.
    """
    if cli_path:
        return Path(cli_path).expanduser()
    env_path = os.environ.get("CHULENGO_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_SETTINGS_FILE_PATH


def set_settings_file(cli_path: str | None = None) -> Path:
    """Point the module at the resolved config file and return that path."""
    global SETTINGS_FILE
    SETTINGS_FILE = resolve_settings_file(cli_path)
    return SETTINGS_FILE


def load_settings() -> dict:
    """Load settings from user config, falling back to package defaults.

    If no settings file exists, copies models.yaml from source to user config.
    Preserves comments and formatting using ruamel.yaml round-trip loader.
    """
    # If user config exists, use it with round-trip loader to preserve comments
    if SETTINGS_FILE.exists():
        yaml = ruamel.yaml.YAML()
        yaml.preserve_quotes = True
        with open(SETTINGS_FILE) as f:
            settings = yaml.load(f) or {}
        return settings

    # Try to find and load defaults from multiple sources
    settings: dict[str, Any] = {"defaults": {}, "model_overrides": {}}

    sources = [
        DEFAULT_SETTINGS_FILE,  # Package installation directory
        Path(__file__).parent / "models.yaml",  # Source directory (editable install)
    ]

    for source in sources:
        if source.exists():
            try:
                yaml = ruamel.yaml.YAML()
                yaml.preserve_quotes = True
                with open(source) as f:
                    settings = yaml.load(f) or {
                        "defaults": {},
                        "model_overrides": {},
                    }
                break
            except Exception:
                continue

    # If no defaults found, create minimal defaults
    if not settings.get("defaults"):
        settings = {
            "defaults": {
                "generic": {"ctx_size": 4096, "flash_attn": "off", "reasoning": "off"}
            },
            "model_overrides": {},
        }

    # Save to user config if we're using defaults (first run)
    save_settings(settings)

    return settings


def save_settings(settings: dict) -> None:
    """Save settings to the user's config file.

    Uses atomic write (write to temp file in same directory, then rename)
    so an interrupted save never leaves a corrupt models.yaml behind.
    Preserves comments and formatting using ruamel.yaml round-trip dumper.
    """
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        dir=SETTINGS_FILE.parent,
        prefix=SETTINGS_FILE.name + ".",
        suffix=".tmp",
        delete=False,
    )
    try:
        yaml = ruamel.yaml.YAML()
        yaml.preserve_quotes = True
        # Use round-trip dumper to preserve comments and formatting
        yaml.dump(settings, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        os.replace(tmp.name, SETTINGS_FILE)
    except BaseException:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


def get_hf_cache_path() -> str | None:
    """Get the HuggingFace cache path with same precedence as llama.cpp."""
    if os.environ.get("LLAMA_CACHE"):
        return os.environ["LLAMA_CACHE"]
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return os.environ["HUGGINGFACE_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    return os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "huggingface",
        "hub",
    )


def get_gguf_architecture(gguf_path: Path) -> str | None:
    """Read model architecture from GGUF file metadata.

    Uses general.architecture from GGUF metadata, which provides
    the authoritative model family identifier used by llama.cpp.

    Args:
        gguf_path: Path to the GGUF file

    Returns:
        The architecture string directly (e.g., "qwen3", "llama", "gemma3")
        or None if the file cannot be read or architecture field is missing.
    """
    try:
        reader = GGUFReader(str(gguf_path))
        field = reader.get_field("general.architecture")
        if field:
            return field.contents()
    except Exception:
        pass
    return None


def extract_gguf_metadata(gguf_path: Path) -> dict | None:
    """Extract comprehensive GGUF metadata in a single pass.

    Reads all tensor metadata (for parameter count) and key metadata fields
    without loading actual tensor data into memory.

    Args:
        gguf_path: Path to the GGUF file

    Returns:
        Dictionary with extracted metadata or None if file cannot be read
    """
    try:
        reader = GGUFReader(str(gguf_path))
    except Exception:
        return None

    # Calculate parameter count from tensors
    # Note: n_elements is pre-computed, no need to access tensor data
    param_count = sum(tensor.n_elements for tensor in reader.tensors)

    # Get architecture
    arch_field = reader.get_field("general.architecture")
    architecture = arch_field.contents() if arch_field else None

    result: dict[str, Any] = {
        "architecture": architecture,
        "parameter_count": param_count,
        "parameter_count_formatted": model_weight_count_rounded_notation(
            param_count, min_digits=3
        ),
    }

    # Extract LLM fields (architecture-specific)
    if architecture:
        arch_fields = {
            "vocab_size": f"{architecture}.vocab_size",
            "context_length": f"{architecture}.context_length",
            "embedding_length": f"{architecture}.embedding_length",
            "block_count": f"{architecture}.block_count",
        }
        for key, field_name in arch_fields.items():
            field = reader.get_field(field_name)
            if field:
                result[key] = field.contents()

    # Extract tokenizer info
    for field_name in ["model", "pre", "bos_id", "eos_id"]:
        field = reader.get_field(f"tokenizer.ggml.{field_name}")
        if field:
            result[f"tokenizer_{field_name}"] = field.contents()

    # Extract multimodal info
    for encoder_type in ["vision", "audio"]:
        field = reader.get_field(f"clip.has_{encoder_type}_encoder")
        if field:
            result[f"has_{encoder_type}_encoder"] = field.contents()

    result["is_multimodal"] = bool(
        result.get("has_vision_encoder") or result.get("has_audio_encoder")
    )

    # Extract general metadata
    for field_name in [
        "name",
        "basename",
        "finetune",
        "description",
        "author",
        "version",
    ]:
        field = reader.get_field(f"general.{field_name}")
        if field:
            result[f"general_{field_name}"] = field.contents()

    return result


def detect_model_family(
    model_name: str, families: list[str] | None = None
) -> str | None:
    """Detect the model family from the model name.

    ``families`` overrides the candidate list; order matters, so put more
    specific names first (a family that is also a prefix of another must come
    after it, or it shadows it). ``None`` means the built-in KNOWN_FAMILIES.

    Matching is a normalized substring test: hyphens and underscores are
    removed from the name, dots are kept as separators. So a minor version in
    the name ("Qwen3.5", "Llama-3.1") resolves to the base family (``qwen3``,
    ``llama3``) rather than a version-specific one (``qwen35``,
    ``llama3_1``); the versioned families are still reached by name when the
    name literally contains them (``qwen35``, ``llama3_1``).
    """
    if families is None:
        families = KNOWN_FAMILIES

    # Dots are deliberately NOT removed: they separate a minor version, so
    # "qwen3.5" cannot match "qwen35" and "llama3.1" cannot match "llama3_1".
    model_norm = re.sub(r"[-_]", "", model_name.lower())
    for family in families:
        if re.sub(r"[-_.]", "", family) in model_norm:
            return family

    return None


def _user_arch_families(defaults: dict) -> list[str]:
    """User-configured extra families, in file order.

    These are `defaults:` keys that name a llama.cpp architecture
    (BUILTIN_ARCHES) chulengo does not know natively (KNOWN_FAMILIES) —
    e.g. a user-added `plamo:` block. Prepending them to the built-in
    family list lets user blocks win name detection over shipped ones.
    """
    return [
        key for key in defaults if key in BUILTIN_ARCHES and key not in KNOWN_FAMILIES
    ]


def _select_shared_arch_family(
    model_name: str, candidates: list[str], defaults: dict
) -> str | None:
    """Pick the most specific family for a shared architecture like `llama`.

    ``candidates`` is ordered most specific first, with the base family last.
    The base (``candidates[-1]``) is a guaranteed fallback and is never name
    matched. Leading candidates are name matched with dots dropped — unlike
    plain name detection, the GGUF has already confirmed the architecture, so
    "Llama-3.1-8B" can resolve to ``llama3_1`` over ``llama3``. Returns None
    when no candidate has a ``defaults`` block.
    """
    model_norm = re.sub(r"[-_.]", "", model_name.lower())
    for family in candidates[:-1]:
        if family in defaults and re.sub(r"[-_.]", "", family) in model_norm:
            return family
    base = candidates[-1]
    return base if base in defaults else None


def _family_source_phrase(resolution: "FamilyResolution") -> str:
    """Describe where a resolved family came from, for `chulengo show`."""
    if resolution.source == "architecture":
        return f"GGUF architecture '{resolution.arch}'"
    if resolution.source == "architecture-name":
        return f"GGUF '{resolution.arch}' architecture, picked by model name"
    if resolution.source == "name":
        return "model name"
    return "the generic fallback"


class FamilyResolution(NamedTuple):
    """Result of resolving a model to its family default settings.

    ``warnings`` carries human-readable notes about a degraded resolution
    (e.g. a cached GGUF whose architecture is unreadable); callers decide
    whether and where to print them.

    ``family`` is the settings block resolved to (``None`` for the generic
    fallback); ``source`` says how it was chosen: "architecture" (block named
    after the GGUF arch), "architecture-name" (shared arch like ``llama``,
    narrowed by model name), "name" (no usable GGUF), or "generic". ``arch``
    is the raw GGUF architecture when it decided the family, else None.
    """

    settings: dict
    warnings: list[str]
    family: str | None = None
    source: str = "generic"
    arch: str | None = None


def resolve_model_family(model_name: str) -> FamilyResolution:
    """Resolve a model to its family default settings, collecting warnings.

    Resolution order, first match wins:

    1. ``general.architecture`` from the cached GGUF, if a block named
       after it exists under ``defaults`` (either the raw architecture or
       its underscore form, so "gpt-oss" matches the shipped "gpt_oss").
    2. name-based family detection, first against user-configured
       architecture blocks (any `defaults:` key naming a llama.cpp
       architecture chulengo does not know natively), then the built-in
       family list (for models without a usable GGUF).
    3. the ``generic`` fallback block.

    A cached GGUF that fails to pin down the family (unreadable
    architecture, or no matching ``defaults`` block) yields a warning
    instead of a silent guess; resolution still falls back to 2 and 3.
    Never prints; warnings are returned so callers can route them (e.g.
    to stderr) as they see fit.
    """
    settings = load_settings()
    defaults = settings.get("defaults", {})
    warnings: list[str] = []

    # Set when a cached GGUF exists but cannot pick a settings block: what
    # went wrong, and an optional hint for fixing it.
    gguf_note: str | None = None
    gguf_hint = ""

    # Priority 1: Try to get architecture from GGUF file metadata
    cache_info = get_model_cache_info(model_name)
    if cache_info and cache_info.get("gguf_path"):
        gguf_architecture = get_gguf_architecture(Path(cache_info["gguf_path"]))
        if gguf_architecture:
            # Shared arch (llama): the architecture alone names several
            # blocks, so narrow it to the most specific family by model name.
            if gguf_architecture == "llama":
                family = _select_shared_arch_family(
                    model_name, LLAMA_ARCH_CANDIDATES, defaults
                )
                if family is not None:
                    return FamilyResolution(
                        defaults[family].copy(),
                        [],
                        family=family,
                        source="architecture-name",
                        arch="llama",
                    )
            for key in (gguf_architecture, gguf_architecture.replace("-", "_")):
                if key in defaults:
                    return FamilyResolution(
                        defaults[key].copy(),
                        [],
                        family=key,
                        source="architecture",
                        arch=gguf_architecture,
                    )
            gguf_note = f"architecture '{gguf_architecture}' has no settings"
            gguf_hint = (
                f"; add a '{gguf_architecture}:' block under defaults: "
                f"in {SETTINGS_FILE}"
            )
        else:
            gguf_note = "could not read the architecture from the cached GGUF"

    # Priority 2: Detect family from model name (for models without GGUF in
    # cache). User-configured architecture blocks are consulted first so
    # they can sharpen detection without a chulengo release.
    family = detect_model_family(
        model_name, families=[*_user_arch_families(defaults), *KNOWN_FAMILIES]
    )

    if family and family in defaults:
        family_settings = defaults[family].copy()
        # Remove None values (for optional args)
        resolved = {k: v for k, v in family_settings.items() if v is not None}
        res_family, res_source = family, "name"
    else:
        # Fall back to generic defaults
        resolved = defaults.get("generic", {})
        res_family, res_source = None, "generic"

    if gguf_note:
        warnings.append(
            f"{gguf_note}; using {res_family or 'generic'} defaults from "
            f"the model name{gguf_hint}"
        )

    return FamilyResolution(resolved, warnings, family=res_family, source=res_source)


def get_default_settings(model_name: str) -> dict:
    """Get default settings for a model based on its family.

    Uses GGUF architecture from file metadata as primary source (no fallback).
    Falls back to name-based detection only when GGUF file is not accessible.
    Thin wrapper over resolve_model_family() for callers that do not care
    about resolution warnings.
    """
    return resolve_model_family(model_name).settings


def _validate_setting_block(
    block: Any,
    context: str,
    valid_keys: frozenset[str],
    issues: list[str],
) -> None:
    """Check one settings block (a family default or a model override).

    Appends human-readable issues for a non-mapping block or unknown keys.
    """
    if not isinstance(block, dict):
        issues.append(
            f"{context}: expected a mapping of settings, got {type(block).__name__}"
        )
        return
    for key in block:
        if key not in valid_keys:
            issues.append(f"{context}: unknown setting key '{key}'")


def _duplicate_spelling_issues(defaults: dict) -> list[str]:
    """Flag ``defaults:`` keys that are the same family spelled differently.

    Keys whose normalized spelling (separator characters removed, lowercased)
    collide — e.g. a user ``gpt-oss:`` block next to the shipped ``gpt_oss:``
    — are a typo symptom: one of them wins name detection, the other never
    does. Each key is individually valid, so this catches the collision that
    keyword checks miss.
    """
    issues: list[str] = []
    by_spelling: dict[str, list[str]] = {}
    for key in defaults:
        spelled = re.sub(r"[-_.]", "", str(key)).lower()
        by_spelling.setdefault(spelled, []).append(str(key))
    for keys in by_spelling.values():
        if len(keys) > 1:
            listed = "; ".join(sorted(keys, key=str))
            issues.append(
                f"defaults: keys {listed} are duplicate spellings of the same "
                "family (they differ only by separator characters)"
            )
    return issues


def doctor_config(settings: Any) -> list[str]:
    """Validate a loaded config against known families and setting keys.

    Returns a list of human-readable issues (empty list == config is valid).
    Checks the whole user config:
        1. Only known top-level keys are present (`defaults`,
           `model_overrides`).
        2. Every key under `defaults` is a known family or a llama.cpp
           architecture (flags typos).
        3. Every setting key inside a family block is a recognised setting.
        4. Every setting key inside a `model_overrides` entry is recognised
           (`alias` is allowed there, but not in family defaults).

    Malformed structures (non-mapping blocks) are reported as issues rather
    than raising, so `chulengo doctor` never tracebacks on a hand-edited file.
    """
    issues: list[str] = []

    if not isinstance(settings, dict):
        return [f"Config root must be a mapping, got {type(settings).__name__}"]

    # 1. Unknown top-level keys are almost always typos. key=str keeps this
    # from raising on mixed-type YAML keys (e.g. an int where a name belongs).
    for key in sorted(settings, key=str):
        if key not in VALID_TOP_LEVEL_KEYS:
            known = ", ".join(sorted(VALID_TOP_LEVEL_KEYS))
            issues.append(f"Unknown top-level key '{key}'. Known keys: {known}")

    # 2-3. Keys under `defaults` must be known (catches e.g. "qwen3 " or
    # "gemmma"); beyond the built-in families, any llama.cpp architecture
    # is a valid key.
    defaults = settings.get("defaults", {}) or {}
    if not isinstance(defaults, dict):
        issues.append(f"'defaults' must be a mapping, got {type(defaults).__name__}")
    else:
        for family in sorted(defaults, key=str):
            if family == "generic":
                # Always-valid fallback family; not a typo even if unused.
                continue
            if family not in VALID_DEFAULT_KEYS:
                known = ", ".join(sorted(KNOWN_FAMILIES))
                issues.append(
                    f"Unknown family '{family}' — likely a typo. Known "
                    f"families: {known}; any llama.cpp architecture name "
                    "(e.g. plamo, llama4) is also valid"
                )
        for family, cfg in defaults.items():
            _validate_setting_block(
                cfg, f"Family '{family}'", VALID_SETTING_KEYS, issues
            )
        # 3b. Two keys that differ only by separator (gpt-oss vs gpt_oss) are
        # the same family twice; one of them can never win name detection.
        issues.extend(_duplicate_spelling_issues(defaults))

    # 4. Per-model overrides (alias is valid here, but not in defaults).
    overrides = settings.get("model_overrides", {}) or {}
    if not isinstance(overrides, dict):
        issues.append(
            f"'model_overrides' must be a mapping, got {type(overrides).__name__}"
        )
    else:
        for model, cfg in overrides.items():
            _validate_setting_block(
                cfg, f"Model override '{model}'", VALID_OVERRIDE_KEYS, issues
            )

        # 5. Aliases must be slash-free and unique across models, otherwise
        # `serve`/`show`/`update`/`delete` cannot resolve them unambiguously.
        alias_owners: dict[str, list[str]] = {}
        for model, cfg in overrides.items():
            if not isinstance(cfg, dict):
                continue
            alias = cfg.get("alias")
            if alias is None:
                continue
            if "/" in str(alias):
                issues.append(
                    f"Model override '{model}': alias '{alias}' must not "
                    "contain '/' (it would look like a repo name)"
                )
            else:
                alias_owners.setdefault(str(alias), []).append(model)
        for alias, models in sorted(alias_owners.items()):
            if len(models) > 1:
                issues.append(
                    f"Alias '{alias}' is used by multiple models: {', '.join(models)}"
                )

    return issues


def resolve_model_name(settings: dict, name: str) -> tuple[str, str | None]:
    """Resolve a request name to the canonical repo name stored in config.

    ``name`` may be a repo name (e.g. ``org/repo``) or a stored alias. It is
    resolved against the ``model_overrides`` mapping:

    * an exact repo-name key wins (also passes through names not in config),
    * otherwise a name matching exactly one stored ``alias`` resolves to that
      repo key,
    * a name matching more than one alias is an error (duplicate alias).

    Returns ``(resolved_name, None)`` on success, or ``(None, message)`` when
    the name is an ambiguous alias.
    """
    overrides = settings.get("model_overrides", {}) or {}
    if not isinstance(overrides, dict):
        return name, None
    # A direct repo-name match is authoritative (and covers unconfigured repos).
    if name in overrides:
        return name, None
    matches = [
        repo
        for repo, cfg in overrides.items()
        if isinstance(cfg, dict) and cfg.get("alias") == name
    ]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return (
            None,
            f"Alias '{name}' matches multiple models: {', '.join(matches)}",
        )
    # Unknown name: pass through unchanged (serve/show can still handle it).
    return name, None


def _alias_conflict(settings: dict, model_name: str, alias: str | None) -> str | None:
    """Return an error message if ``alias`` is invalid or taken by another model.

    An alias must not contain ``/`` (it would look like a repo name and never
    resolve) and must not already be used by a different model.
    """
    if alias is None:
        return None
    if "/" in alias:
        return (
            f"Alias '{alias}' must not contain '/' (it would look like a repo "
            "name and never resolve)"
        )
    overrides = settings.get("model_overrides", {}) or {}
    if not isinstance(overrides, dict):
        return None
    for repo, cfg in overrides.items():
        if repo == model_name:
            continue
        if isinstance(cfg, dict) and cfg.get("alias") == alias:
            return f"Alias '{alias}' is already used by model '{repo}'"
    return None


def get_model_cache_info(model_name: str) -> dict | None:
    """Get model info from HF cache."""
    cache_path = get_hf_cache_path()
    if not cache_path:
        return None

    # Normalize model name for cache lookup
    # HF cache uses: models--{org}--{repo} path
    parts = model_name.split("/")
    if len(parts) == 2:
        org, repo = parts
        repo_cache_name = f"models--{org}--{repo}"
    else:
        repo_cache_name = model_name.replace("/", "--")

    model_cache_dir = Path(cache_path) / repo_cache_name

    if not model_cache_dir.exists():
        return None

    # Find the latest snapshot
    snapshots = list(model_cache_dir.glob("snapshots/*"))
    if not snapshots:
        return None

    latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)

    # Find GGUF files and separate mmproj files
    all_gguf_files = list(latest_snapshot.glob("*.gguf"))
    # Separate into regular GGUF files and mmproj files
    gguf_files = [f for f in all_gguf_files if not f.name.startswith("mmproj")]
    mmproj_files = [f for f in all_gguf_files if f.name.startswith("mmproj")]

    # Get the first GGUF file path for metadata reading (if available)
    gguf_path = str(gguf_files[0]) if gguf_files else None

    # Get the first mmproj file path if available
    mmproj_path = str(mmproj_files[0]) if mmproj_files else None

    return {
        "cache_dir": str(model_cache_dir),
        "snapshot": str(latest_snapshot),
        "gguf_files": [f.name for f in gguf_files],
        "gguf_path": gguf_path,
        "mmproj_files": [f.name for f in mmproj_files],
        "mmproj_path": mmproj_path,
    }


def cmd_ls(args: argparse.Namespace) -> int:
    """List available models from HF cache."""
    cache_path = get_hf_cache_path()
    if not cache_path:
        print("Error: Could not determine HF cache path", file=sys.stderr)
        return 1

    cache = Path(cache_path)

    print(f"Searching in: {cache}")

    found_models = []
    for model_dir in cache.glob("models--*--*"):
        # Get the snapshots directory
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            continue

        # Find all snapshot directories and use the latest one
        snapshot_dirs = list(snapshots_dir.glob("*"))
        if not snapshot_dirs:
            continue

        # Get the latest snapshot by modification time
        latest_snapshot = max(snapshot_dirs, key=lambda p: p.stat().st_mtime)

        # Find GGUF files (they are symlinks, glob follows them)
        gguf_files = [
            f for f in latest_snapshot.glob("*.gguf") if not f.name.startswith("mmproj")
        ]
        if gguf_files:
            repo_name = model_dir.name.replace("models--", "").replace("--", "/")
            size = sum(f.stat().st_size for f in gguf_files)
            size_str = f"{size / (1024**3):.2f}GB"
            gguf_name = gguf_files[0].name
            found_models.append((repo_name, size_str, gguf_name))

    if not found_models:
        print("No models found in cache")
        return 0

    print(f"\n{'Model':<55} {'Size':<10} File")
    print("-" * 80)
    for repo, size, gguf in sorted(found_models):
        print(f"{repo:<55} {size:<10} {gguf}")

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show model details and settings."""
    model_name = args.model

    # Get settings first
    settings = load_settings()

    # Allow `show <alias>` to resolve to the stored repo name.
    model_name, error = resolve_model_name(settings, model_name)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    model_overrides = settings.get("model_overrides", {})

    # Get cache info (optional - model might not be downloaded yet)
    cache_info = get_model_cache_info(model_name)

    # Show warning if model not in cache
    if not cache_info:
        print(
            f"Warning: Model '{model_name}' is not available in the HuggingFace cache.",
            file=sys.stderr,
        )
        print("Use 'chulengo ls' to list available models.", file=sys.stderr)
        return 1

    # Build display info
    print(f"Model: {model_name}")

    print("\nCache Location:")
    print(f"  Repository: {cache_info['cache_dir']}")
    print(f"  Snapshot: {cache_info['snapshot']}")

    print("\nAvailable GGUF files:")
    for gguf in cache_info["gguf_files"]:
        print(f"  {gguf}")

    # Extract and display GGUF metadata
    if cache_info.get("gguf_path"):
        metadata = extract_gguf_metadata(Path(cache_info["gguf_path"]))
        if metadata:
            print("\nGGUF Metadata:")
            print(f"  Architecture: {metadata.get('architecture', 'unknown')}")
            print(
                f"  Parameters: {metadata.get('parameter_count_formatted', 'unknown')}"
            )

            # Show multimodal capabilities
            # Multimodal can be detected from GGUF metadata OR from mmproj files
            # Extract metadata from mmproj file if available to check for vision/audio encoders
            mmproj_metadata = None
            if cache_info.get("mmproj_path"):
                mmproj_metadata = extract_gguf_metadata(Path(cache_info["mmproj_path"]))

            # Merge multimodal info from both sources
            has_vision_encoder = metadata.get("has_vision_encoder") or (
                mmproj_metadata and mmproj_metadata.get("has_vision_encoder")
            )
            has_audio_encoder = metadata.get("has_audio_encoder") or (
                mmproj_metadata and mmproj_metadata.get("has_audio_encoder")
            )
            is_multimodal = bool(has_vision_encoder or has_audio_encoder)

            print(f"\n  Multimodal: {'Yes' if is_multimodal else 'No'}")

            if is_multimodal:
                if has_vision_encoder:
                    print("    Vision encoder: Yes")
                if has_audio_encoder:
                    print("    Audio encoder: Yes")
                # Show mmproj files if they exist (indicates multimodal capability)
                if cache_info.get("mmproj_files"):
                    print(
                        f"    Multimodal projection files: {', '.join(cache_info['mmproj_files'])}"
                    )

            # Show architecture-specific details
            arch_details = []
            if "vocab_size" in metadata:
                arch_details.append(f"    Vocab size: {metadata['vocab_size']:,}")
            if "context_length" in metadata:
                arch_details.append(
                    f"    Context length: {metadata['context_length']:,}"
                )
            if "embedding_length" in metadata:
                arch_details.append(
                    f"    Embedding length: {metadata['embedding_length']:,}"
                )
            if "block_count" in metadata:
                arch_details.append(f"    Number of layers: {metadata['block_count']}")

            if arch_details:
                print("\n  Architecture Details:")
                for detail in arch_details:
                    print(detail)

            # Show tokenizer info
            tokenizer_info = []
            if "tokenizer_model" in metadata:
                tokenizer_info.append(f"  Model: {metadata['tokenizer_model']}")
            if "tokenizer_pre" in metadata:
                tokenizer_info.append(f"  Pre: {metadata['tokenizer_pre']}")
            if "tokenizer_bos_id" in metadata:
                tokenizer_info.append(f"  BOS token ID: {metadata['tokenizer_bos_id']}")
            if "tokenizer_eos_id" in metadata:
                tokenizer_info.append(f"  EOS token ID: {metadata['tokenizer_eos_id']}")

            if tokenizer_info:
                print("\n  Tokenizer:")
                for info in tokenizer_info:
                    print(f"  {info}")

            # Show general metadata if available
            general_info = []
            if metadata.get("general_name"):
                general_info.append(f"  Name: {metadata['general_name']}")
            if metadata.get("general_finetune"):
                general_info.append(f"  Finetune: {metadata['general_finetune']}")
            if metadata.get("general_basename"):
                general_info.append(f"  Base model: {metadata['general_basename']}")
            if metadata.get("general_description"):
                desc = metadata["general_description"]
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                general_info.append(f"  Description: {desc}")

            if general_info:
                print("\n  Model Info:")
                for info in general_info:
                    print(f"  {info}")

    # Resolve the family exactly as `serve` will, and show the SAME resolved
    # settings block, so `show` can never diverge from what `serve` applies.
    resolution = resolve_model_family(model_name)

    # Surface the family-resolution warnings `serve` would print too.
    for warning in resolution.warnings:
        print(f"Warning: {warning}", file=sys.stderr)

    if resolution.family:
        print(f"\nDetected family: {resolution.family}")
        print(f"  ({_family_source_phrase(resolution)})")
        print()
        print(f"Default settings for {resolution.family}:")
        for k, v in resolution.settings.items():
            # Convert boolean values to on/off strings
            display_val = v
            if isinstance(v, bool):
                display_val = "on" if v else "off"
            print(f"  --{k} {display_val}")
    else:
        print("\nCould not detect model family")

    # Show any model-specific overrides
    if model_name in model_overrides:
        print(f"\nCustom settings for {model_name}:")
        for k, v in model_overrides[model_name].items():
            # Convert boolean values to on/off strings
            display_val = v
            if isinstance(v, bool):
                display_val = "on" if v else "off"
            print(f"  --{k} {display_val}")

    return 0


def _is_on(value: Any) -> bool:
    """Interpret a config switch that may be a bool or an on/off-style string.

    The config is loaded as YAML 1.2, so `on`/`off` arrive as strings, while
    older configs or CLI flags may hold real booleans.
    """
    return str(value).lower() in ("true", "on", "yes", "1")


def build_llama_command(
    model_name: str,
    port: int | None = None,
    extra_args: list[str] | None = None,
    alias: str | None = None,
    family_resolution: FamilyResolution | None = None,
    # Override arguments
    ctx_size: int | None = None,
    flash_attn: str | None = None,
    reasoning: str | None = None,
    jinja: bool | None = None,
    spec_type: str | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    chat_template_file: str | None = None,
) -> list[str]:
    """Build the llama.cpp serve command with settings.

    ``family_resolution`` lets a caller that already resolved the family
    (e.g. cmd_serve, to print warnings) pass the result through; ``None``
    resolves here via get_default_settings(), as before.
    """
    cmd = ["llama", "serve"]

    # Add port if specified
    if port:
        cmd.extend(["--port", str(port)])

    # Resolve user overrides once; the stored alias lives here. A CLI --alias
    # wins over the alias saved by create/update.
    user_settings = load_settings().get("model_overrides", {}).get(model_name, {}) or {}
    effective_alias = alias if alias is not None else user_settings.get("alias")

    # Add alias if specified (CLI flag or stored config alias)
    if effective_alias:
        cmd.extend(["--alias", str(effective_alias)])

    # Add model
    cmd.extend(["--hf-repo", model_name])

    # Get default settings: from a pre-resolved family (cmd_serve resolves
    # once so it can print warnings), or resolve here as before.
    if family_resolution is not None:
        defaults = family_resolution.settings
    else:
        defaults = get_default_settings(model_name)

    # Merge settings (user overrides take precedence). `alias` is emitted as a
    # flag above, not a setting, so drop it from the merged dict.
    all_settings = {**defaults, **user_settings}
    all_settings.pop("alias", None)

    # Apply command-line overrides
    if ctx_size is not None:
        all_settings["ctx_size"] = ctx_size
    if flash_attn is not None:
        all_settings["flash_attn"] = flash_attn
    if reasoning is not None:
        all_settings["reasoning"] = reasoning
    if jinja is not None:
        all_settings["jinja"] = jinja
    if spec_type is not None:
        all_settings["spec_type"] = spec_type
    if cache_type_k is not None:
        all_settings["cache_type_k"] = cache_type_k
    if cache_type_v is not None:
        all_settings["cache_type_v"] = cache_type_v
    if chat_template_file is not None:
        all_settings["chat_template_file"] = chat_template_file

    # Add settings flags
    if all_settings.get("ctx_size"):
        cmd.extend(["--ctx-size", str(all_settings["ctx_size"])])

    # Handle flash_attn - can be boolean or string
    flash_attn_val = all_settings.get("flash_attn")
    if flash_attn_val is not None:
        cmd.extend(["--flash-attn", "on" if _is_on(flash_attn_val) else "off"])

    # Handle reasoning - can be boolean or string
    reasoning_val = all_settings.get("reasoning")
    if reasoning_val is not None:
        cmd.extend(["--reasoning", "on" if _is_on(reasoning_val) else "off"])

    # Handle jinja - can be boolean or string ("off" must not enable it)
    if _is_on(all_settings.get("jinja")):
        cmd.append("--jinja")

    if all_settings.get("spec_type"):
        cmd.extend(["--spec-type", str(all_settings["spec_type"])])

    if all_settings.get("cache_type_k"):
        cmd.extend(["--cache-type-k", str(all_settings["cache_type_k"])])

    if all_settings.get("cache_type_v"):
        cmd.extend(["--cache-type-v", str(all_settings["cache_type_v"])])

    if all_settings.get("chat_template_file"):
        cmd.extend(["--chat-template-file", str(all_settings["chat_template_file"])])

    return cmd


def cmd_create(args: argparse.Namespace) -> int:
    """Create settings for a model."""
    model_name = args.model
    settings = load_settings()

    if "model_overrides" not in settings:
        settings["model_overrides"] = {}

    # Reject an alias that would be ambiguous or look like a repo name.
    conflict = _alias_conflict(settings, model_name, args.alias)
    if conflict:
        print(f"Error: {conflict}", file=sys.stderr)
        return 1

    # Build settings dict from args
    new_settings: dict[str, Any] = {}

    if args.alias:
        new_settings["alias"] = args.alias

    if args.ctx_size:
        new_settings["ctx_size"] = args.ctx_size

    if args.flash_attn is not None:
        new_settings["flash_attn"] = args.flash_attn

    if args.reasoning is not None:
        new_settings["reasoning"] = args.reasoning

    if args.jinja:
        new_settings["jinja"] = True

    if args.spec_type:
        new_settings["spec_type"] = args.spec_type

    if args.cache_type_k:
        new_settings["cache_type_k"] = args.cache_type_k

    if args.cache_type_v:
        new_settings["cache_type_v"] = args.cache_type_v

    if args.chat_template_file:
        new_settings["chat_template_file"] = args.chat_template_file

    # Check if model already has settings
    was_new = model_name not in settings["model_overrides"]

    settings["model_overrides"][model_name] = new_settings

    save_settings(settings)

    if was_new:
        print(f"Settings created for {model_name}")
    else:
        print(f"Settings overwritten for {model_name}")

    return 0


def cmd_update(args: argparse.Namespace) -> int:
    """Update settings for a model."""
    settings = load_settings()

    # Allow `update <alias>` to target the stored repo name.
    model_name, error = resolve_model_name(settings, args.model)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if "model_overrides" not in settings:
        settings["model_overrides"] = {}

    # Reject an alias that would be ambiguous or look like a repo name.
    conflict = _alias_conflict(settings, model_name, args.alias)
    if conflict:
        print(f"Error: {conflict}", file=sys.stderr)
        return 1

    # Build settings dict from args
    updates: dict[str, Any] = {}

    if args.alias:
        updates["alias"] = args.alias

    if args.ctx_size:
        updates["ctx_size"] = args.ctx_size

    if args.flash_attn is not None:
        updates["flash_attn"] = args.flash_attn

    if args.reasoning is not None:
        updates["reasoning"] = args.reasoning

    if args.jinja:
        updates["jinja"] = True

    if args.spec_type:
        updates["spec_type"] = args.spec_type

    if args.cache_type_k:
        updates["cache_type_k"] = args.cache_type_k

    if args.cache_type_v:
        updates["cache_type_v"] = args.cache_type_v

    if args.chat_template_file:
        updates["chat_template_file"] = args.chat_template_file

    # Check if model exists - if not, create new settings
    was_new = False
    if model_name not in settings["model_overrides"]:
        was_new = True
        settings["model_overrides"][model_name] = {}

    # Get existing settings and apply updates
    existing = settings["model_overrides"][model_name].copy()
    existing.update(updates)
    settings["model_overrides"][model_name] = existing

    save_settings(settings)

    if was_new:
        print(f"Settings created for {model_name}")
    else:
        print(f"Settings updated for {model_name}")

    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete a model override."""
    settings = load_settings()

    # Allow `delete <alias>` to target the stored repo name.
    target, error = resolve_model_name(settings, args.target)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    model_overrides = settings.get("model_overrides", {})
    if target not in model_overrides:
        print(f"No override found for {target}", file=sys.stderr)
        return 1

    del settings["model_overrides"][target]
    save_settings(settings)
    print(f"Settings deleted for {target}")

    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve a model with llama.cpp."""
    # Allow `serve <alias>` to resolve to the stored repo name.
    resolved, error = resolve_model_name(load_settings(), args.model)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    model_name = resolved

    # Resolve family settings once, surface any warnings, and pass the
    # resolution through so build_llama_command does not resolve twice.
    resolution = resolve_model_family(model_name)
    for warning in resolution.warnings:
        print(f"Warning: {warning}", file=sys.stderr)

    # Build command
    cmd = build_llama_command(
        model_name=model_name,
        port=args.port,
        alias=args.alias,
        family_resolution=resolution,
        # Override arguments
        ctx_size=args.ctx_size,
        flash_attn=args.flash_attn,
        reasoning=args.reasoning,
        jinja=args.jinja,
        spec_type=args.spec_type,
        cache_type_k=args.cache_type_k,
        cache_type_v=args.cache_type_v,
        chat_template_file=args.chat_template_file,
    )

    if args.dry_run:
        # Print the exact command that would run (all defaults and overrides
        # resolved) without executing it.
        print("Dry run — not executing:")
        print(f"  {' '.join(cmd)}")
        return 0

    print(f"Running: {' '.join(cmd)}")

    # Execute the command
    try:
        # Use Popen to allow handling KeyboardInterrupt gracefully
        proc = subprocess.Popen(cmd)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            # Gracefully terminate the subprocess on Ctrl+C
            print("\nInterrupted, stopping server...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            return 130  # Standard exit code for SIGINT
    except FileNotFoundError:
        print(
            "Error: llama command not found. Please install llama.cpp.", file=sys.stderr
        )
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def load_user_config() -> tuple[dict | None, str | None]:
    """Read the user config file directly, without seeding defaults.

    Unlike load_settings(), this never creates or writes the config file: a
    validation command must not have side effects. Returns a ``(settings,
    error)`` tuple:

    * file missing -> ``(None, None)`` (there is nothing to validate)
    * valid YAML   -> ``(settings, None)``
    * bad YAML     -> ``(None, "<message>")``
    """
    if not SETTINGS_FILE.exists():
        return None, None

    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    try:
        with open(SETTINGS_FILE) as f:
            return yaml.load(f) or {}, None
    except Exception as e:
        return None, f"Could not parse config file {SETTINGS_FILE}: {e}"


def cmd_doctor(args: argparse.Namespace) -> int:
    """Validate the user config without creating or modifying it."""
    settings, error = load_user_config()

    if settings is None and error is None:
        print(
            f"No user config found at {SETTINGS_FILE}; nothing to validate. "
            "Defaults are used until you run 'chulengo create'."
        )
        return 0

    issues = [error] if error is not None else doctor_config(settings)

    if not issues:
        print("OK: config is valid. No issues found.")
        return 0

    print(f"Found {len(issues)} issue(s) in your config:\n")
    for issue in issues:
        print(f"    - {issue}")

    print(
        "Tip: run 'chulengo show' to inspect a model, or edit "
        f"{SETTINGS_FILE} directly."
    )
    return 1


def main() -> int:
    """Main entry point."""
    # Shared option so `--config` is accepted both before and after the
    # subcommand (e.g. `chulengo --config X ls` and `chulengo ls --config X`).
    # SUPPRESS keeps the subparser from clobbering a value given earlier.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help=(
            "Path to the config file (models.yaml). Overrides the "
            "CHULENGO_CONFIG environment variable and the default "
            "~/.config/chulengo/models.yaml"
        ),
    )

    parser = argparse.ArgumentParser(
        prog="chulengo",
        description="A small and wild llama.cpp wrapper for serving HuggingFace models",
        parents=[common],
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ls command
    ls_parser = subparsers.add_parser(
        "ls", help="List available models from HF cache", parents=[common]
    )
    ls_parser.set_defaults(func=cmd_ls)

    # show command
    show_parser = subparsers.add_parser(
        "show", help="Show model details and settings", parents=[common]
    )
    show_parser.add_argument("model", help="Model name to show")
    show_parser.set_defaults(func=cmd_show)

    # create command
    create_parser = subparsers.add_parser(
        "create", help="Create settings for a model", parents=[common]
    )
    create_parser.add_argument("model", help="Model name")
    create_parser.add_argument("--alias", "-a", help="Alias for the model")
    create_parser.add_argument("--ctx-size", type=int, help="Context size")
    create_parser.add_argument(
        "--flash-attn", choices=["on", "off"], help="Enable/disable flash attention"
    )
    create_parser.add_argument(
        "--reasoning", choices=["on", "off"], help="Enable/disable reasoning"
    )
    create_parser.add_argument(
        "--jinja", action="store_true", help="Use Jinja template"
    )
    create_parser.add_argument("--spec-type", help="Speculative decoding type")
    create_parser.add_argument("--cache-type-k", help="Cache type for keys")
    create_parser.add_argument("--cache-type-v", help="Cache type for values")
    create_parser.add_argument(
        "--chat-template-file", help="Path to chat template file"
    )
    create_parser.set_defaults(func=cmd_create)

    # update command
    update_parser = subparsers.add_parser(
        "update", help="Update settings for a model", parents=[common]
    )
    update_parser.add_argument("model", help="Model name")
    update_parser.add_argument("--alias", "-a", help="Alias for the model")
    update_parser.add_argument("--ctx-size", type=int, help="Context size")
    update_parser.add_argument(
        "--flash-attn", choices=["on", "off"], help="Enable/disable flash attention"
    )
    update_parser.add_argument(
        "--reasoning", choices=["on", "off"], help="Enable/disable reasoning"
    )
    update_parser.add_argument(
        "--jinja", action="store_true", help="Use Jinja template"
    )
    update_parser.add_argument("--spec-type", help="Speculative decoding type")
    update_parser.add_argument("--cache-type-k", help="Cache type for keys")
    update_parser.add_argument("--cache-type-v", help="Cache type for values")
    update_parser.add_argument(
        "--chat-template-file", help="Path to chat template file"
    )
    update_parser.set_defaults(func=cmd_update)

    # delete command
    delete_parser = subparsers.add_parser(
        "delete", help="Delete settings for a model", parents=[common]
    )
    delete_parser.add_argument("target", help="Model name to delete settings for")
    delete_parser.set_defaults(func=cmd_delete)

    # serve command
    serve_parser = subparsers.add_parser(
        "serve", help="Serve a model with llama.cpp", parents=[common]
    )
    serve_parser.add_argument("model", help="Model name to serve")
    serve_parser.add_argument(
        "--port", "-p", type=int, default=11853, help="Port to serve on"
    )
    serve_parser.add_argument("--alias", "-a", help="Alias for the model")
    # Override flags
    serve_parser.add_argument("--ctx-size", type=int, help="Override context size")
    serve_parser.add_argument(
        "--flash-attn", choices=["on", "off"], help="Override flash attention"
    )
    serve_parser.add_argument(
        "--reasoning", choices=["on", "off"], help="Override reasoning"
    )
    # default=None (not False) so an absent flag leaves the config's jinja alone
    serve_parser.add_argument(
        "--jinja", action="store_true", default=None, help="Enable Jinja template"
    )
    serve_parser.add_argument("--spec-type", help="Override speculative decoding type")
    serve_parser.add_argument("--cache-type-k", help="Override cache type for keys")
    serve_parser.add_argument("--cache-type-v", help="Override cache type for values")
    serve_parser.add_argument("--chat-template-file", help="Path to chat template file")
    serve_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact command that would run without executing it",
    )
    serve_parser.set_defaults(func=cmd_serve)

    # doctor command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Validate the user config (families, setting keys)",
        parents=[common],
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    args = parser.parse_args()
    set_settings_file(getattr(args, "config", None))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
