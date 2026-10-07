#!/usr/bin/env python3
"""
Chulengo - A small and wild llama.cpp wrapper

A lightweight wrapper that simplifies serving HuggingFace models with llama.cpp
by applying sensible defaults based on model family.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from gguf import GGUFReader
from gguf.utility import model_weight_count_rounded_notation

# Default settings file path
SETTINGS_FILE = Path.home() / ".config" / "chulengo" / "models.yaml"
DEFAULT_SETTINGS_FILE = Path(os.path.realpath(__file__)).parent / "models.yaml"


# Known model families. Kept as a module-level constant so family detection and
# config validation (chulengo doctor) share a single source of truth.
KNOWN_FAMILIES = [
    "qwen35moe",
    "qwen35",
    "gpt_oss",
    "qwancoder",
    "qwopus",
    "codestral",
    "llama3_2",
    "llama3_1",
    "llama3",
    "llama",
    "qwen3",
    "qwen",
    "laguna",
    "mixtral",
    "mistral",
    "mistral3",
    "mistral4",
    "glm4moe",
    "glm4",
    "gemma3",
    "gemma2",
    "gemma",
    "granitemoe",
    "granite",
    "granitemoeshared",
    "phi",
    "deepseek2",
    "deepseek",
    "nemotron_h",
    "nemotron",
    "qwencoder",
]

# "generic" is the always-valid fallback family, so it counts as known.
VALID_FAMILIES = frozenset({"generic", *KNOWN_FAMILIES})

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


def load_settings() -> dict:
    """Load settings from user config, falling back to package defaults.

    If no settings file exists, copies models.yaml from source to user config.
    """
    # If user config exists, use it
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            settings = yaml.safe_load(f) or {}
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
                with open(source) as f:
                    settings = yaml.safe_load(f) or {
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
        with tmp as f:
            yaml.dump(settings, f, default_flow_style=False)
            f.flush()
            os.fsync(f.fileno())
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


def detect_model_family(model_name: str) -> str | None:
    """Detect the model family from the model name."""
    model_lower = model_name.lower()

    # Order matters - check more specific family names first
    families = [
        "qwen35moe",
        "qwen35",
        "gpt_oss",
        "qwancoder",
        "qwopus",  # Explicit check for qwopus before qwen
        "codestral",
        "llama3_2",
        "llama3_1",
        "llama3",
        "llama",
        "qwen3",
        "qwen",
        "laguna",
        "mixtral",
        "mistral",
        "mistral3",
        "mistral4",  # new Mixtral uses mistral4 architecture
        "glm4moe",
        "glm4",
        "gemma3",
        "gemma2",
        "gemma",
        "granitemoe",
        "granite",
        "granitemoeshared",
        "phi",
        "deepseek2",
        "deepseek",
        "nemotron_h",
        "nemotron",
        "qwencoder",
    ]

    for family in families:
        # Normalize both by removing underscores and hyphens
        family_norm = family.replace("_", "").replace("-", "")
        model_norm = model_lower.replace("_", "").replace("-", "")

        # For llama variants, check for patterns like "llama-3" or "llama3"
        if family in ("llama3", "llama3_1", "llama3_2"):
            base = "llama3"
            if base in model_norm and base != family:
                # Check if this is actually a llama3 variant
                import re

                # Check for llama followed by 3 (with potential dot)
                if not re.search(r"llama[\._-]?3", model_lower):
                    continue
            if family_norm in model_norm:
                return family

        # Standard check for other families
        if family_norm in model_norm:
            return family

    return None


def get_default_settings(model_name: str) -> dict:
    """Get default settings for a model based on its family.

    Uses GGUF architecture from file metadata as primary source (no fallback).
    Falls back to name-based detection only when GGUF file is not accessible.
    """
    settings = load_settings()
    defaults = settings.get("defaults", {})

    # Priority 1: Try to get architecture from GGUF file metadata
    cache_info = get_model_cache_info(model_name)
    if cache_info and cache_info.get("gguf_path"):
        gguf_architecture = get_gguf_architecture(Path(cache_info["gguf_path"]))
        if gguf_architecture and gguf_architecture in defaults:
            return defaults[gguf_architecture].copy()

    # Priority 2: Detect family from model name (for models without GGUF in cache)
    family = detect_model_family(model_name)

    if family and family in defaults:
        family_settings = defaults[family].copy()
        # Remove None values (for optional args)
        return {k: v for k, v in family_settings.items() if v is not None}

    # Fall back to generic defaults
    return defaults.get("generic", {})


def doctor_config(settings: dict) -> list[str]:
    """Validate a loaded config against known families and setting keys.

    Returns a list of human-readable issues (empty list == config is valid).
    Checks two things:
        1. Every family key under `defaults` is a known family (flags typos).
        2. Every setting key inside a family block is a recognised setting
           (flags unknown setting keys).
    """
    issues: list[str] = []

    defaults = settings.get("defaults", {}) or {}

    # 1. Family keys must be known (catches e.g. "qwen3 " or "gemmma").
    for family in sorted(defaults):
        if family == "generic":
            # Always-valid fallback family; not a typo even if unused.
            continue
        if family not in VALID_FAMILIES:
            known = ", ".join(sorted(VALID_FAMILIES))
            issues.append(
                f"Unknown family '{family}' — likely a typo. "
                f"Known families: {known}"
            )

    # 2. Setting keys inside each family block must be recognised.
    for family, cfg in defaults.items():
        if not isinstance(cfg, dict):
            continue
        for key in cfg:
            if key not in VALID_SETTING_KEYS:
                issues.append(f"Family '{family}': unknown setting key '{key}'")

    return issues


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

    # Detect family and show settings
    family = detect_model_family(model_name)
    if family:
        print(f"\nDetected family: {family}")
        print()
        defaults = settings.get("defaults", {})
        if family in defaults:
            print(f"Default settings for {family}:")
            for k, v in defaults[family].items():
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


def build_llama_command(
    model_name: str,
    port: int | None = None,
    extra_args: list[str] | None = None,
    alias: str | None = None,
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
    """Build the llama.cpp serve command with settings."""
    cmd = ["llama", "serve"]

    # Add port if specified
    if port:
        cmd.extend(["--port", str(port)])

    # Add alias if specified
    if alias:
        cmd.extend(["--alias", alias])

    # Add model
    cmd.extend(["--hf-repo", model_name])

    # Get default settings
    defaults = get_default_settings(model_name)
    user_settings = load_settings().get("model_overrides", {}).get(model_name, {})

    # Merge settings (user overrides take precedence)
    all_settings = {**defaults, **user_settings}

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
        val_str = (
            "on" if str(flash_attn_val).lower() in ("true", "on", "yes", "1") else "off"
        )
        cmd.extend(["--flash-attn", val_str])

    # Handle reasoning - can be boolean or string
    reasoning_val = all_settings.get("reasoning")
    if reasoning_val is not None:
        val_str = (
            "on" if str(reasoning_val).lower() in ("true", "on", "yes", "1") else "off"
        )
        cmd.extend(["--reasoning", val_str])

    if all_settings.get("jinja"):
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
    model_name = args.model
    settings = load_settings()

    if "model_overrides" not in settings:
        settings["model_overrides"] = {}

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
    target = args.target
    settings = load_settings()

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
    model_name = args.model

    # Build command
    cmd = build_llama_command(
        model_name=model_name,
        port=args.port,
        alias=args.alias,
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


def cmd_doctor(args: argparse.Namespace) -> int:
    """Validate the user config against known families and setting keys."""
    settings = load_settings()
    issues = doctor_config(settings)

    if not issues:
        print("OK: config is valid. No issues found.")
        return 0

    print(f"Found {len(issues)} issue(s) in your config:\n")
    for issue in issues:
        print(f"    - {issue}")

    print(
        "Tip: run 'chulengo show' to inspect a model, or edit "
        "~/.config/chulengo/models.yaml directly."
    )
    return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="chulengo",
        description="A small and wild llama.cpp wrapper for serving HuggingFace models",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ls command
    ls_parser = subparsers.add_parser("ls", help="List available models from HF cache")
    ls_parser.set_defaults(func=cmd_ls)

    # show command
    show_parser = subparsers.add_parser("show", help="Show model details and settings")
    show_parser.add_argument("model", help="Model name to show")
    show_parser.set_defaults(func=cmd_show)

    # create command
    create_parser = subparsers.add_parser("create", help="Create settings for a model")
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
    update_parser = subparsers.add_parser("update", help="Update settings for a model")
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
    delete_parser = subparsers.add_parser("delete", help="Delete settings for a model")
    delete_parser.add_argument("target", help="Model name to delete settings for")
    delete_parser.set_defaults(func=cmd_delete)

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Serve a model with llama.cpp")
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
    serve_parser.add_argument(
        "--jinja", action="store_true", help="Enable Jinja template"
    )
    serve_parser.add_argument("--spec-type", help="Override speculative decoding type")
    serve_parser.add_argument("--cache-type-k", help="Override cache type for keys")
    serve_parser.add_argument("--cache-type-v", help="Override cache type for values")
    serve_parser.add_argument("--chat-template-file", help="Path to chat template file")
    serve_parser.set_defaults(func=cmd_serve)

    # doctor command
    doctor_parser = subparsers.add_parser(
        "doctor", help="Validate the user config against known families"
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
