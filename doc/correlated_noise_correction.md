# Correlated Noise Flux Error Correction in Slimfarmer

## Problem

IMCOM coadded images have strongly correlated pixel noise because the coadding process mixes neighboring pixels. Slimfarmer's Fisher-matrix flux error assumes independent pixels:

```
flux_err = 1/sqrt(D),    D = sum_i w_i T_i^2
```

This underestimates the true flux uncertainty by a factor of ~5 for Dec25 IMCOM simulations.

## Method

### The flux estimator

Slimfarmer estimates flux by minimizing weighted residuals:

```
f_hat = sum_i(w_i T_i d_i) / sum_i(w_i T_i^2)
```

where `d_i` is the data, `w_i = 1/sigma_i^2` is the weight, and `T_i` is the unit template (PSF-convolved source profile).

### Variance with correlated noise

The noise covariance is `Cov(j,k) = r(dx,dy) sigma_j sigma_k`, where:
- `sigma_i` is the per-pixel noise (varies spatially with depth)
- `r(dx,dy)` is the correlation coefficient (depends only on pixel separation)

The variance of the flux estimate is:

```
Var(f_hat) = sum_{j,k} h_j r(j-k) h_k / D^2
```

where `h_i = sqrt(w_i) T_i`.

### Correction factor kappa

```
kappa^2 = sum_{j,k} h_j r(j-k) h_k / D
```

The corrected flux error is `kappa * flux_err`. For uncorrelated noise `r = delta`, giving `kappa = 1`.

The template weighting `h_i` appears because the flux estimator gives more weight to bright pixels — if those pixels are correlated, the effective noise is amplified.

### Estimating r(dx, dy)

The correlation coefficient is estimated from noise realizations (CPR layers 24-27):

1. Estimate per-pixel sigma directly from the noise realizations: `sigma_est = std(realizations, axis=0)`, smoothed with a 15-pixel uniform filter to reduce noise from few realizations.
2. Normalize: `n_norm = n / sigma_est`
3. Compute average autocorrelation via FFT on the full image: `r(dx,dy) = IFFT(<|FFT(n_norm)|^2>) / N_pix`

This approach normalizes by the **actual noise amplitude** rather than the weight map, avoiding unit mismatches between noise layers and the weight/variance maps.

## Implementation

### Pipeline flow

1. **`FarmerImage.__init__`**: loads noise realizations, estimates `r(dx,dy)` from their own smoothed std, stores as `noise_corr`.

2. **`process_groups()`**: fits all sources via multiprocessing. Each group caches compact data (`h_vals`, `nzy`, `nzx`, `D`) for kappa computation. The `noise_corr` arrays are temporarily removed from `band_config` before forking workers to reduce memory.

3. **`compute_kappa()`**: uses cached data + `noise_corr` to compute `kappa^2 = sum h_j r(j-k) h_k / D` per source. No Tractor rendering, no FFT — just a fast numpy pairwise sum on ~100 nonzero pixels per source.

4. **`build_catalog()`**: applies kappa to `flux_err`, `flux_err_noshot`, `flux_err_des`.

### Multiprocessing

- Uses `mp.Pool` with `apply_async` + per-group timeout (120s) to handle Tractor C-level segfaults gracefully.
- `maxtasksperchild=20` restarts workers periodically to prevent memory leaks.
- Large arrays (`noise_corr`) removed from `band_config` before forking, restored after.

### Files modified

- **`utils.py`**:
  - `prepare_images_from_cpr()`: extracts noise realizations from CPR layers 24-27
  - `get_params()`: applies kappa to `flux_err`, `flux_err_noshot`, `flux_err_des`; outputs `flux_err_kappa`

- **`image.py`**:
  - `FarmerImage.__init__()`: loads noise realizations, estimates `noise_corr` from their own std
  - `compute_kappa()`: post-fit kappa using cached data
  - `run_photometry()`: calls `compute_kappa()` after fitting

- **`_group.py`**:
  - `_cache_kappa_data()`: caches compact `h_vals, nzy, nzx, D` per source during fitting

### Usage

```python
import slimfarmer

# From CPR file (returns 5 paths)
sci, wht, psf, eff_gain, noise_reals = slimfarmer.prepare_images_from_cpr(
    cpr_path, work_dir
)

cat = slimfarmer.run_photometry(
    science_path=sci, weight_path=wht, eff_gain_path=eff_gain,
    psf_path=psf, noise_reals_path=noise_reals,
    band='F158', zeropoint=ZP,
)

# flux_err, flux_err_noshot, flux_err_des are already corrected by kappa
# flux_err_kappa contains the correction factor for diagnostics
```

### Multi-band usage

```python
cat = slimfarmer.run_photometry(
    bands={
        'H1': {'science': sci, 'weight': wht, 'psf': psf,
               'zeropoint': ZP, 'eff_gain': eff_gain,
               'noise_reals': noise_reals_path},
        ...
    },
    detection_band='combined',
    ncpus=48,
)
```

### Catalog columns

When `noise_reals` is provided:
- `flux_err`: corrected by kappa
- `flux_err_noshot`: corrected by kappa
- `flux_err_des`: corrected by kappa
- `flux_err_kappa`: the correction factor itself
- `flux_err_tractor_origin`: original uncorrected Fisher error

When `noise_reals` is not provided, kappa = 1 and all errors are unchanged.

## flux_err_des: DES/ngmix-style residual-scaled error

In addition to the correlated noise correction (kappa), slimfarmer computes `flux_err_des` following the DES/ngmix convention.

### Background

The Tractor optimizer (like `scipy.optimize.leastsq`) returns the Fisher inverse `inv(J^T J)` as the parameter covariance. This assumes the residual variance is exactly 1 — i.e., the noise model is perfect. When the model is imperfect (PSF mismatch, neighbor contamination, incorrect noise model), the actual residual variance `s^2 = chi2/dof` differs from 1.

The correct parameter covariance is ([scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html), [ngmix](https://github.com/esheldon/ngmix)):

```
Cov(params) = inv(J^T J) * s^2
```

### Implementation

```
flux_err_des = sqrt(chi2/dof) * flux_err_tractor
```

where:
- `chi2 = sum((data - model)^2 * invvar)` on the source footprint
- `dof = n_footprint_pixels - n_params`
- `flux_err_tractor` = Tractor's Fisher-based error (from `inv(J^T J)`)

The computation is restricted to the source footprint (pixels where model > 1% of peak) to avoid diluting `chi2/dof` with background pixels in the larger group region.

### Interpretation

- `chi2/dof ~ 1`: good fit → `flux_err_des ~ flux_err_tractor`
- `chi2/dof > 1`: model mismatch → `flux_err_des > flux_err_tractor` (inflated to account for unmodeled systematics)

## Validation

See `doc/Flux_error_validation_colab.ipynb`, Part 3.

## Key design decisions

1. **Normalize by noise realization std, not weight map**: avoids unit mismatches between noise layers and the Sigma-derived weight map.

2. **Smooth the per-pixel sigma estimate**: with only 3-4 realizations, raw `std(axis=0)` is noisy. A 15-pixel uniform filter stabilizes the estimate while preserving spatial variation.

3. **r depends only on (dx, dy)**: assumes IMCOM coadding produces spatially uniform correlations. Depth variation is captured by the per-pixel sigma.

4. **r estimated on the full image**: small group cutouts give biased periodogram estimates.

5. **Cache compact data during fitting**: stores only 1D arrays (`h_vals, nzy, nzx`) and scalar `D` per source — small enough to pickle back from worker processes without memory issues.

6. **No Tractor rendering in compute_kappa**: avoids OOM from creating Tractor engines for hundreds of sources on the full image.

7. **Multiprocessing timeout**: `apply_async` with per-group timeout handles Tractor segfaults without hanging the pool.
