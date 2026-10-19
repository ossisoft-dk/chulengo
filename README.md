# Chulengo

A small and wild llama.cpp wrapper that makes serving HuggingFace models with llama.cpp a breeze.

## Overview

Ollama is great. It's easy and convenient. But it's slower than pure llama.cpp.

Chulengo is a tiny wrapper for llama.cpp that makes it easy to serve any HuggingFace model with sensible defaults based on model family.

Instead of:

```bash
llama serve \
  --port 11853 \
  --hf-repo Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF:Q4_K_M \
  --ctx-size 204800 \
  --flash-attn on \
  --reasoning off \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
```

You can simply run:

```bash
chulengo serve Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF:Q4_K_M
```

## Features

- **Model Type Detection**: Automatically detects and applies sensible defaults for common model types (Qwen, Llama, Gemma, Mistral, etc.)
- **GGUF Architecture Detection**: Uses GGUF's `general.architecture` field as the primary source for model family identification
- **Settings Persistence**: Store and manage model-specific settings in a YAML file
- **Cache Integration**: Works with HuggingFace cache to discover and display model information
- **Lightweight**: An invisible layer on top of llama.cpp with minimal overhead

## Installation

First, install `uv` if you haven't already: <https://docs.astral.sh/uv/>

Then, from the chulengo project directory, run:

```bash
uv tool install -e .
```

This installs chulengo and all its dependencies (`pyyaml`, `gguf`) in an isolated environment, making the `chulengo` command immediately available in your terminal.

> **Updating after changes**
>
> `uv tool install -e .` installs a copy of the code, so after editing `chulengo.py`
> you need to reinstall to pick up your changes:
>
> ```bash
> uv tool install -e . --force
> ```
>
> If `chulengo` suddenly reports a command that you know you just added (e.g.
> `doctor`), it's almost certainly this stale copy — reinstall as above and it
> will reappear.

## Uninstall

```bash
uv tool uninstall chulengo
```

To also remove your settings:

```bash
rm -rf ~/.config/chulengo
```

## Usage

### List Models

List all GGUF models available in your HuggingFace cache:

```bash
chulengo ls
```

### Show Model Details

Display information about a model and its current settings:

```bash
chulengo show Qwen/Qwen3-8B
```

### Create Model Settings

Create settings for a model:

```bash
chulengo create Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF:Q4_K_M \
  --ctx-size 204800 \
  --flash-attn on \
  --reasoning off \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
```

**Note:** If settings already exist, they will be overwritten.

### Update Model Settings

Update settings for an existing model (or create if it doesn't exist):

```bash
chulengo update Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF:Q4_K_M \
  --ctx-size 262144
```

**Note:** If the model doesn't have existing settings, they will be created for you.

### Delete Model Settings

Delete settings for a model:

```bash
chulengo delete my-model-name
```

**Note:** The model must have existing settings to delete.

### Check Configuration

Validate your settings file (`~/.config/chulengo/models.yaml`) without editing
it by hand. `doctor` flags unknown model families, unknown setting keys in
defaults and model overrides, unknown top-level keys, and malformed blocks.
Family keys include any llama.cpp architecture name (see
[Model Detection](#model-detection)), so a `plamo:` or `llama4:` block is
valid config, not a typo:

```bash
chulengo doctor
```

It exits non-zero when it finds issues, so it is safe to use in a script or
shell prompt. It only reads your config — it never writes or seeds a file, and
reports (rather than crashes on) missing or invalid YAML.

### Serve a Model

Serve a model with llama.cpp, using detected defaults:

```bash
chulengo serve Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF:Q4_K_M
```

With custom port:

```bash
chulengo serve Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF:Q4_K_M --port 8001
```

By default, Chulengo serves on port **11853**. You can specify a different port with the `--port` or `-p` flag.

By default the server binds to **localhost only** (`127.0.0.1`), so other machines on your network cannot reach it. Bind to a specific interface with `--bind` (passed to llama as `--host`):

```bash
chulengo serve Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF:Q4_K_M --bind 0.0.0.0
```

> **Security note:** `llama serve` has no built-in authentication. Exposing it on `0.0.0.0` or a LAN interface lets anyone who can reach that port send requests and trigger model loads — only do that behind a reverse proxy, firewall, or VPN that authenticates.
>
> Binding to a non-loopback address makes `chulengo serve` warn on stderr before launch; a `--bind` value ending in `.sock` is a Unix socket (reachable only by local processes) and triggers no warning.

Override settings from command line:

```bash
chulengo serve Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF:Q4_K_M \
  --ctx-size 262144 \
  --reasoning off
```

See the exact command that `serve` would run (family defaults and overrides
all resolved) without executing it:

```bash
chulengo serve Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF:Q4_K_M --dry-run
```

`--dry-run` prints the full `llama serve` command, shell-quoted so it can be
copied and run as-is, then exits without launching anything.

## Model Aliases

Give a long repo name a short handle so you can serve it without typing the
whole thing. Set an alias with `create` or `update`:

```bash
chulengo create LiquidAI/LFM2.5-230M-GGUF --alias lfm
chulengo update LiquidAI/LFM2.5-230M-GGUF --alias lfm
```

Then serve by either name — both resolve to the same model and emit the stored
alias as llama.cpp's `--alias` (a `--alias` on the command line wins over the
stored one):

```bash
chulengo serve lfm
chulengo serve LiquidAI/LFM2.5-230M-GGUF
```

`show`, `update`, and `delete` accept the alias the same way. Aliases must be
unique and must not contain `/` (that would look like a repo name); `create`,
`update`, and `doctor` all enforce this.

## Model Detection

Chulengo uses two sources to determine the appropriate model settings:

### Primary: GGUF Architecture

When a model's GGUF file is available in the HuggingFace cache, Chulengo reads the `general.architecture` field directly from the GGUF metadata. This is the authoritative source used by llama.cpp. The matching `defaults:` block may be named after the raw architecture or its underscore form — a `gpt-oss` GGUF picks up the shipped `gpt_oss` block.

Some architectures are shared by several families. The `llama` architecture covers Llama-3.x, Mistral, Mixtral and Codestral, so a cached llama-arch model is narrowed to the most specific shipped block by model name: `meta-llama/Llama-3.1-8B` gets the `llama3_1` block (not the plain `llama` one), a `Mistral-7B` gets `mistral`, and so on. Families with their own architecture (`mistral3`, `mistral4`, `qwen35`, ...) are matched by that architecture directly.

A cached GGUF that fails to pin down a family is reported instead of silently guessed: when the architecture cannot be read, or when it has no `defaults:` block, `serve` and `show` print a warning to stderr and fall back to name-based detection (and, failing that, the `generic` block). The warning names the fallback actually used, e.g. `Warning: could not read the architecture from the cached GGUF; using qwen3 defaults from the model name`, and for a missing block also suggests adding it: `architecture 'plamo' has no settings; using qwen3 defaults from the model name; add a 'plamo:' block under defaults: in ~/.config/chulengo/models.yaml`.

### Fallback: Name-based Detection

When the GGUF file is not accessible (e.g., model not downloaded), Chulengo falls back to pattern matching on the model name. Architecture blocks you have added yourself (see below) are matched first, ahead of the built-in family list. Matching is a normalized substring test: hyphens and underscores are removed, but dots separate a minor version — so `Qwen3.5` resolves to `qwen3` (not `qwen35`) and `Llama-3.1`/`Llama-3.2` to `llama3`. The versioned blocks (`qwen35`, `llama3_1`) apply when a cached GGUF reports that architecture, not from a dotted name alone. More specific families are checked before their prefixes (`mistral3` before `mistral`, `qwencoder` before `qwen`), so no family is shadowed.

`chulengo show` displays the same resolved settings block `serve` uses and says where the family came from — the GGUF architecture (with the raw arch name) or the model name — so the two commands can never disagree about defaults.

### Adding a New Architecture

Settings for an architecture Chulengo does not know natively can be added without a release: put one block under `defaults:` in `~/.config/chulengo/models.yaml`, named after the GGUF `general.architecture` value. Any llama.cpp architecture name is valid, e.g. `plamo` or `llama4`:

```yaml
defaults:
  plamo:
    ctx_size: 8192
    flash_attn: on
    jinja: on
```

Such a block is used when a cached GGUF reports that architecture, and — because user-added blocks are matched before the built-in families — also when the model name suggests it, even for models that are not downloaded yet. `chulengo doctor` accepts these blocks and still flags real typos.

## Supported Model Types

Chulengo includes sensible defaults for these model types:

| Model Type | Default Settings                                                                            |
| ---------- | ------------------------------------------------------------------------------------------- |
| qwen3      | ctx-size=16384, flash-attn=on, cache-type-k=q8_0, cache-type-v=q8_0, jinja                  |
| qwen       | ctx-size=8192, flash-attn=on, jinja                                                         |
| qwen35     | ctx-size=16384, flash-attn=on, cache-type-k=q8_0, cache-type-v=q8_0, jinja                  |
| qwen35moe  | ctx-size=32768, flash-attn=on, cache-type-k=q8_0, cache-type-v=q8_0, spec-type=draft, jinja |
| qwopus     | ctx-size=204800, flash-attn=on, reasoning=off, cache-type-k=q8_0, cache-type-v=q8_0         |
| laguna     | ctx-size=262144, flash-attn=on, jinja                                                       |
| llama3     | ctx-size=8192, flash-attn=on, jinja                                                         |
| llama      | ctx-size=4096, flash-attn=on                                                                |
| gemma3     | ctx-size=8192, flash-attn=off, cache-type-k=q8_0, cache-type-v=q8_0, spec-type=draft        |
| gemma2     | ctx-size=4096, flash-attn=off                                                               |
| mistral3   | ctx-size=8192, flash-attn=on, jinja                                                         |
| mixtral4   | ctx-size=8192, flash-attn=on, jinja, cache-type-k=q8_0, cache-type-v=q8_0                   |
| deepseek   | ctx-size=16384, flash-attn=on                                                               |
| deepseek2  | ctx-size=16384, flash-attn=on                                                               |
| glm4       | ctx-size=8192, flash-attn=on, jinja                                                         |
| glm4moe    | ctx-size=32768, flash-attn=on, jinja, cache-type-k=q8_0, cache-type-v=q8_0, spec-type=draft |
| granite    | ctx-size=4096, flash-attn=on, jinja                                                         |
| codestral  | ctx-size=32768, flash-attn=on, jinja, cache-type-k=q8_0, cache-type-v=q8_0                  |
| lfm2       | ctx-size=32768, flash-attn=on, jinja                                                        |

## Configuration File

Settings are stored in `~/.config/chulengo/models.yaml`. You can also customize the defaults file in the same directory as `chulengo.py`.

### Alternate Config Location

To use a config file other than the default — for a per-project config or for
pointing tests at an alternate file — set `CHULENGO_CONFIG` to a path, or pass
`--config <path>` to any command:

```bash
CHULENGO_CONFIG=./config/models.yaml chulengo show my-model
chulengo --config ./config/models.yaml doctor
chulengo doctor --config ./config/models.yaml
```

The `--config` flag takes precedence over the `CHULENGO_CONFIG` environment
variable, which takes precedence over the default location. `~` is expanded and
relative paths are resolved against the current working directory.

## Goals

- **Performance**: Be an almost invisible layer on top of llama.cpp
- **Simplicity**: Reduce the need to remember all the flags
- **Extensibility**: Easy to add new model types and settings

## License

See LICENSE file.
