# Roadmap

This file tracks the direction of chulengo and where we are heading. It is a
living document — items move between sections as they are picked up, and
community feedback is welcome on [issues](https://github.com/ossisoft-dk/chulengo/issues).

## Next (near-term)

- **Serving**
  - `--bind` passthrough and a clear note about the default network exposure
    of `llama serve`
  - `chulengo status` / `stop`: track the pid of a served model and stop it
    cleanly instead of relying on Ctrl+C in the original terminal
- **Multi-GPU support** (GPU selection is currently fully implicit —
  llama.cpp's defaults decide, which on multi-GPU boxes can place layers on
  a device you don't want, e.g. an iGPU)
  - GPU setting keys plumbed end-to-end like the existing ones: `ngl`,
    `split_mode`, `tensor_split`, `main_gpu`
  - Setting precedence, lowest to highest: `machine` section (new, per-host
    defaults in the config) → family default → per-model override → CLI flag
  - `chulengo gpus` subcommand: list the GPUs as llama.cpp will see them
    (device index, backend, VRAM, **gfx target**, and a live **HIP-init
    OK/FAIL** probe) so `-mg`/`-ts` values are discoverable and a broken
    or mismatched card is visible at a glance instead of a raw HIP
    stacktrace

## Later (ideas, not committed)

- Custom family names that aren't GGUF architectures (e.g. `devstral` running on `mistral3`), via an optional `families:` section linking a name to an arch
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

## Known issues

- **AMD single-target prebuilt vs a multi-gfx box**
  - _Symptom:_ `llama serve` dies during model load with
    `hip_code_object.cpp: ...getStatFunc... Assertion 'err == hipSuccess' failed`
    on one AMD GPU, while another card on the same box loads
    fine — even with the failing card fully idle. Not a chulengo, model,
    or device-ownership problem.
  - _Cause:_ some llama.cpp installers (e.g. the `llama.app` install
    script) download a **single-gfx-target** prebuilt binary: a probe
    inspects the machine and picks **one** gfx target, so a box with
    multiple RDNA4 cards on **different** gfx targets gets a binary that
    only covers one of them. Cards whose reported target differs from
    the build's then fail the HIP code-object load. The probe's pick is
    **not stable across installs** on a multi-gfx box, so the
    under-covered card can change between updates.
  - _Interim fix (works):_ for the under-covered card, set
    `HSA_OVERRIDE_GFX_VERSION=<the build's target>` for that serve run,
    or system-wide. This makes the card present as the build's target so
    its prebuilt kernels load. Low-risk when the card is a patch-level
    match to that target.
  - _Durable fix:_ self-build llama.cpp with **all** the box's targets
    (e.g. `CMAKE_HIP_ARCHITECTURES="gfx1200;gfx1201"`) for one fat binary
    that covers every card natively — no override, and immune to the
    probe re-picking a different target on the next update. Trade-off:
    you own the build and lose the installer's one-command updates.
  - _Example:_ a box with a gfx12.0.0 card and a gfx12.0.1 card, where
    the prebuilt was built for gfx12.0.1 — the gfx12.0.0 card needs the
    override (or the dual-target build) to run.

## Contributing

1. Comment on an open issue or file a new one describing what you need.
2. Roadmap items without an issue are fair game to pick up — link your PR to
   the section above.
3. "Later" items move to "Next" when someone (ideally a maintainer) starts on
   them; finished items are removed from the roadmap, with git history as the
   record of what shipped.
