# Chulengo

A small and wild llama.cpp wrapper that makes serving HuggingFace models with llama.cpp a breeze.

## Overview

Ollama is great. It's easy and convenient. But it's slower than pure llama.cpp.

Chulengo is a tiny wrapper for llama.cpp that makes it easy to serve any HuggingFace model with sensible defaults based on model family.

Instead of:

```bash
llama serve \
  --port 8001 \
  --alias qwopus \
  --hf-repo Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF:Q4_K_M \
  --ctx-size 204800 \
  --spec-type draft-mtp \
  --reasoning off \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja \
  --chat-template-file $HOME/.config/llama.cpp/templates/froggeric/Qwen-Fixed-Chat-Templates/chat_template.jinja
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

Make sure you have llama.cpp installed and available in your PATH. Then:

```bash
pip install pyyaml  # if not already installed

# Either run with python:
python3 chulengo.py --help

# Or create a symlink for convenience (add to ~/.local/bin):
ln -sf $(pwd)/chulengo.py ~/.local/bin/chulengo
```

## Uninstall

To remove Chulengo:

```bash
# Remove the symlink
rm ~/.local/bin/chulengo

# Remove your settings (optional - keeps model configurations)
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
  --alias qwopus \
  --ctx-size 204800 \
  --spec-type draft-mtp \
  --reasoning off \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja
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

### Serve a Model

Serve a model with llama.cpp, using detected defaults:

```bash
chulengo serve Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF:Q4_K_M
```

With custom port:

```bash
chulengo serve Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF:Q4_K_M --port 8001
```

Override settings from command line:

```bash
chulengo serve Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF:Q4_K_M \
  --ctx-size 262144 \
  --reasoning off
```

## Model Detection

Chulengo uses two sources to determine the appropriate model settings:

### Primary: GGUF Architecture

When a model's GGUF file is available in the HuggingFace cache, Chulengo reads the `general.architecture` field directly from the GGUF metadata. This is the authoritative source used by llama.cpp.

### Fallback: Name-based Detection

When the GGUF file is not accessible (e.g., model not downloaded), Chulengo falls back to pattern matching on the model name.

## Supported Model Types

Chulengo includes sensible defaults for these model types:

| Model Type | Default Settings                                                                            |
| ---------- | ------------------------------------------------------------------------------------------- |
| qwen3      | ctx-size=16384, flash-attn=on, cache-type-k=q8_0, cache-type-v=q8_0, jinja                  |
| qwen       | ctx-size=8192, flash-attn=on, jinja                                                         |
| qwen35     | ctx-size=16384, flash-attn=on, cache-type-k=q8_0, cache-type-v=q8_0, jinja                  |
| qwen35moe  | ctx-size=32768, flash-attn=on, cache-type-k=q8_0, cache-type-v=q8_0, spec-type=draft, jinja |
| qwopus     | ctx-size=204800, spec-type=draft-mtp, flash-attn=on, cache-type-k=q8_0                      |
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

## Configuration File

Settings are stored in `~/.config/chulengo/models.yaml`. You can also customize the defaults file in the same directory as `chulengo.py`.

## Goals

- **Performance**: Be an almost invisible layer on top of llama.cpp
- **Simplicity**: Reduce the need to remember all the flags
- **Extensibility**: Easy to add new model types and settings

## License

See LICENSE file.
