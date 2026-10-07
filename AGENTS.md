# Agent / Contributor Instructions

Read this before editing. It exists so you (human or AI agent) make changes
that match the repo's conventions and verify them reliably.

## Style: black is the source of truth

This repo is **black-formatted: 4-space indent, 88-column lines, single quotes
left as-is.** Do not hand-format, and do not "fix" indentation, line length,
or quotes to taste. If you are unsure what is valid, run black.

- Formatter: `black` (config in `pyproject.toml` under `[tool.black]`).
- Linter: `ruff` (config under `[tool.ruff]`) — lint only; it flags real
  problems (unused imports, undefined names, etc.), not style. It is
  configured to not conflict with black (`E501` ignored).

Never introduce a different indentation width. The historical 3-space style
has been retired; 4-space is now canonical.

Two practical corollaries, learned the hard way:

- The 3-space/4-space rule is about **Python source only**.
- Markdown keeps its own existing bullet-continuation width; do not re-indent it to match Python.
- **Never hand-type runs of whitespace to match or construct indentation.**
- In editor and agent pipelines, runs of spaces you emit are unreliable and can silently come out wrong.
- That produces token-valid code that passes black and pytest yet does not match the file ladder.
- Instead: let black produce the whitespace, or derive indentation programmatically (len minus lstrip, chr(32) times n).
- When a whitespace edit cannot be found twice, suspect the space run, not the file.

## Setting keys: the six-spot checklist

A "setting key" is a llama.cpp serve flag chulengo knows about (e.g.
`ctx_size`, `flash_attn`, `reasoning`, `jinja`, `spec_type`,
`cache_type_k`, `cache_type_v`, `chat_template_file`). **Adding one means
touching ~6 places.** Missing one produces a plausible-but-broken feature.
Check all of them:

1. `argparse` — add the flag to **each** of the `create`, `update`, and
   `serve` subparsers in `main()` (they are declared separately, not shared).
2. `cmd_create()` — read `args.<key>` into `new_settings`.
3. `cmd_update()` — read `args.<key>` into `updates`.
4. `build_llama_command()` — accept the kwarg, apply it to `all_settings`,
   and emit the corresponding llama.cpp flag(s) into `cmd`.
5. `cmd_show()` — display the value in the "Default settings for <family>"
   and/or "Custom settings" sections (booleans rendered as on/off).
6. `models.yaml` — add the key under the relevant family `defaults:` (and/or
   document it), plus a test.

Note: key names are currently **repeated string literals**, not central
constants. Keep them consistent by hand across all six spots. (A future
cleanup may centralize them; see ROADMAP.md.)

Also keep `create` and `update` in sync — they build their settings dicts
from the same flag set and should stay identical in shape.

- For Python files the tools are the authority: write, then uv run black and uv run ruff (this is what catches hand-typed whitespace drift). For Markdown keep bullets single-line so no continuation indentation is needed.
## Verification loop

After any change, run:

```
uv sync --extra dev
uv run pytest -q
uv run ruff check chulengo.py tests/
uv run black --check chulengo.py tests/
```

All four must be clean. There is **no CI** yet — the above is the only gate,
so run it yourself; do not rely on the test run alone.

**Caveat:** the tests use brittle **global mocks** — `mocker.patch` of
`pathlib.Path.exists` and `builtins.open`. If a test starts failing after a
refactor, suspect mock coupling (the patch no longer hits the right code path)
before assuming your logic is wrong. Prefer local, targeted mocks when adding
tests.

## Layout

- `chulengo.py` — the entire CLI (argparse, config load/save, GGUF metadata
  extraction, family detection, command builders, `serve`). Single module by
  design; it is a thin wrapper over llama.cpp and should stay that way.
- `models.yaml` — per-family default settings (shipped default; copied to the
  user's `~/.config/chulengo/models.yaml` on first run).
- `tests/` — pytest suite.
- `README.md` — user-facing overview.
- `ROADMAP.md` — direction and open items; check it before starting new
  features.

## Boundaries

chulengo is a **thin wrapper** around llama.cpp. Do not reimplement serving,
model download, or quantization here — pass through to the `llama` binary.
See the "Not planned" section of ROADMAP.md.