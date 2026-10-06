# Roadmap

This file tracks the direction of chulengo and where we are heading. It is a
living document — items move between sections as they are picked up, and
community feedback is welcome on [issues](https://github.com/ossisoft-dk/chulengo/issues).

## Done

- CLI for inspecting, configuring, and serving HuggingFace models with llama.cpp
   (`ls`, `show`, `create`, `update`, `delete`, `serve`)
- Per-family default settings with GGUF `general.architecture` as the primary
   signal and name-based family detection as fallback
- User config at `~/.config/chulengo/models.yaml`, seeded from the packaged
   defaults on first run
- Per-model overrides (`model_overrides`) that take precedence over family
   defaults
- Multimodal (vision/audio) detection from GGUF metadata and `mmproj` files
- Graceful Ctrl+C handling when serving

## In progress

*(nothing yet — see Next for what is currently planned)*

## Next (near-term)

- **Config file hygiene**
   - Atomic writes (write-to-temp + rename) so an interrupted `create`/`update`
     never corrupts `models.yaml`
   - Preserve comments in user config on update (or offer a `chulengo config
     view` that annotates instead)
   - `chulengo doctor`: validate the user config against known families and
     flag typos/unknown keys
- **Family detection**
   - Reduce fallback reliance on name matching: warn (not just silently guess)
     when a cached GGUF exists but its architecture is unknown
   - Configurable family list so new architectures can be added without a
     release
- **Serving**
   - `--bind` passthrough and a clear note about the default network exposure
     of `llama serve`
   - `chulengo status` / `stop`: track the pid of a served model and stop it
     cleanly instead of relying on Ctrl+C in the original terminal
   - Print the exact command used (already done) and offer `--dry-run`
- **Multi-GPU support** (GPU selection is currently fully implicit —
   llama.cpp's defaults decide, which on multi-GPU boxes can place layers on
   a device you don't want, e.g. an iGPU)
   - GPU setting keys plumbed end-to-end like the existing ones: `ngl`,
     `split_mode`, `tensor_split`, `main_gpu`
   - Setting precedence, lowest to highest: `machine` section (new, per-host
     defaults in the config) → family default → per-model override → CLI flag
   - `chulengo gpus` subcommand: list the GPUs as llama.cpp will see them
     (device index, backend, VRAM) so `-mg`/`-ts` values are discoverable
     instead of guessed

## Later (ideas, not committed)

- Multi-model serving profiles (one config entry → several replicas/aliases)
- `chulengo bench`: wrapper around llama-bench with saved per-model results
- Model download progress/mirror passthrough (`llama download`)
- Chat completions smoke test (`chulengo ping`) against a running server
- Windows and non-XDG config locations (follow platformdirs)
- Packaging as a standalone binary (e.g. via uv/pipx tool install) with
   llama.cpp auto-discovery

## Not planned

- Reimplementing llama.cpp functionality — chulengo stays a thin wrapper and
   defers to upstream for serving, downloads, and quantization
- A web/GUI frontend
- Centralized model registry or cloud features

## Contributing

1. Comment on an open issue or file a new one describing what you need.
2. Roadmap items without an issue are fair game to pick up — link your PR to
   the section above.
3. "Later" items move to "Next" when someone (ideally a maintainer) starts
   on them; "Done" items stay in history so users can see the trajectory.