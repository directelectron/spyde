# Releasing / upgrading the SpotUNet disk detector

This is the single source of truth for **shipping a revised model**. The detector
is meant to be revised indefinitely until it does everything perfectly; this is
the repeatable per-iteration path so each new model reaches users cleanly.

## How the registry resolves a model

`spyde/models/registry.py` merges three manifest layers (later overrides earlier
on `id`; the latest non-null `default` wins):

```
bundled  spyde/models/weights/registry.json   (ships in the wheel, pinned)
  < remote/user  ~/.spyde/models/registry.json (fetched from Hugging Face, or hand-edited)
```

- Each model has an **immutable, versioned `id`** (`spotunet-base16-v1`,
  `-v2`, …). Never reuse an id for new weights — a cached/downloaded `.pt` must
  never be silently swapped under a user.
- `default` points at the current best. Promoting a new model = bump `default`.
- Weights come from a `source`: `{"type": "bundled", "file": …}` (in-package) or
  `{"type": "hf", "repo": …, "file": …}` (downloaded to `~/.spyde/models`).

## A. Ship a new model WITHOUT re-releasing SpyDE (the common case)

Users pick it up via **Find Vectors → Model dropdown → refresh** (the
`fv_refresh_models` action calls `registry.refresh_remote_registry()`).

1. Train / iterate in the **`yoloDiffraction`** repo
   (`scripts/train.py` / `train_edge.py` / …). The checkpoint stores its own
   `base` / `in_ch` / `levels`.
2. **Validate** it beats the current default on the real-scale benchmark:
   `python -m spyde.tests.benchmark_neural_spots`.
3. Upload the `.pt` to the HF repo (`cssfrancis/spyde-spotunet`) under a
   **versioned filename**, e.g. `spotunet-base16-v2.pt`.
4. Add a versioned entry to the repo's `registry.json` and (once accepted as
   best) set `default` to it:
   ```json
   {
     "id": "spotunet-base16-v2",
     "label": "SpotUNet base16 v2",
     "version": 2,
     "notes": "Better recall on thick samples / grain boundaries.",
     "arch": {"base": 16, "in_ch": 1, "levels": 2},
     "source": {"type": "hf", "repo": "cssfrancis/spyde-spotunet",
                "file": "spotunet-base16-v2.pt"}
   }
   ```
5. Done — on the next refresh the model appears in the dropdown and (if promoted)
   becomes the default. No SpyDE build required.

## B. Promote a model INTO the bundle (for a SpyDE release)

Makes the newest model the **offline / first-run default**. Do this when a model
has proven itself and you're cutting a SpyDE release.

1. Copy the validated `.pt` into `spyde/models/weights/`.
2. Add (or update) its entry in `spyde/models/weights/registry.json` with
   `"source": {"type": "bundled", "file": "<name>.pt"}` and bump the bundled
   `default`.
3. Confirm `pyproject.toml` `[tool.setuptools.package-data]` still globs
   `models/weights/*.pt` (it does) so the new file ships in the wheel and the
   frozen build.

Until promoted, the bundled default stays pinned (reproducible offline) and
remote-refresh (path A) delivers the upgrade.
