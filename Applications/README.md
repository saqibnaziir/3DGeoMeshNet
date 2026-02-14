# Applications

This folder contains demo applications built on the 3DGeoMeshNet model: **interpolation**, **extrapolation**, and **noising/denoising**. Run all scripts from the **repository root** so that `config` and `utils` imports work (e.g. `python Applications/interpolation.py`).

## Interpolation

**File:** `interpolation.py`

Interpolates between two 3D meshes in latent space and saves a sequence of intermediate meshes.

![3DGeoMeshNet Architecture](main/Img/Inter_Extr.jpg)


- **Usage:** Edit the path variables at the bottom of the script (`INTERPOLATION_FOLDER`, `CONFIG_PATH`, `OUTPUT_FOLDER`), then run:
  ```bash
  python Applications/interpolation.py
  ```
- Or call `mesh_interpolation(interpolation_folder, config_path, output_folder, steps=10, verbose=True)` from your own code.

## Extrapolation

**File:** `extrapolation.py`

Extrapolates beyond two meshes in latent space (extending the trajectory defined by two input meshes).

- **Usage:** Edit the path variables in the script’s `if __name__ == "__main__"` block, then run:
  ```bash
  python Applications/extrapolation.py
  ```
- Or use `mesh_interpolation_or_extrapolation(..., mode='extrapolation')` in code.

## Noising / Denoising

**File:** `noising.py`

Adds Gaussian noise along vertex normals to meshes (e.g. to create synthetic noisy inputs for denoising experiments).

- **Usage:** Set `source_folder` and `target_folder` at the top of the script, then run:
  ```bash
  python Applications/noising.py
  ```

Additional denoising-related scripts (e.g. PSNR evaluation, mesh cropping) can be placed under `Applications/Denoising/` if needed.
