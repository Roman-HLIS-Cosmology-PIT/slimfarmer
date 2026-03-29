# slimfarmer
Simpler version of the farmer which combines SEP for detection and Tractor for photometry measurement

Detection (SEP) → grouping → model selection → forced photometry, with optional neighbor subtraction for crowded fields.

  1. Detection (SEP) — run once, not iterative

  SEP (Source Extractor in Python) is called a single time to produce a source catalog and segmentation map. There is no iterative detection, no source subtraction loop, and no re-detection. The output is a fixed list of positions and a per-pixel segmentation map assigning each
  detected pixel to a source.

  2. Grouping

  Sources whose segmaps overlap (after dilation by dilation_radius = 0.07") are placed into the same group and co-fitted simultaneously. This prevents Tractor from ignoring neighbors when fitting a blended source. A separate, larger dilation (fit_dilation_radius = 0.24") defines the
  pixel footprint Tractor actually sees — large enough to capture profile wings without merging distant sources into oversized groups.

  3. Model selection

  For each group, Tractor tries a sequence of increasingly complex models: PointSource → SimpleGalaxy → ExpGalaxy / DevGalaxy → FixedCompositeGalaxy. It selects the simplest model whose reduced chi² is acceptable.

  4. Forced photometry

  With morphology fixed to the best model, Tractor re-fits only the flux of every source. Optionally (neighbor_subtraction = True), nearby group models are subtracted from the data before the final flux measurement to reduce cross-contamination.




## Installation

```bash
cd slimfarmer
pip install -e .
```

Requires: `numpy`, `astropy>=5.0`, `scipy`, `sep`, `tqdm`, `tractor`.
Optional: `pathos` (parallel processing), `galsim` (flux unit conversion).

---

## Quick start

### Single-band

```python
import slimfarmer

cat = slimfarmer.run_photometry(
    science_path = 'roman_image.fits',
    weight_path  = 'roman_weight.fits',
    psf_path     = 'PSF_F158.fits',
    band         = 'F158',
    zeropoint    = 26.511,
    output_path  = 'catalog.fits',   # optional
    ncpus        = 8,                # 0 = serial
)
print(cat['id', 'ra', 'dec', 'F158_flux', 'F158_flux_err', 'name'])
```

### Multi-band
Have not tested yet

```python
cat = slimfarmer.run_photometry(
    bands={
        'F158': {'science': 'F158.fits', 'weight': 'F158_wht.fits',
                 'psf': 'F158_psf.fits', 'zeropoint': 26.511},
        'F106': {'science': 'F106.fits', 'weight': 'F106_wht.fits',
                 'psf': 'F106_psf.fits', 'zeropoint': 26.310},
    },
    detection_band = 'F158',
    output_path    = 'catalog.fits',
)
```

### Custom configuration

```python
from slimfarmer import Config

cfg = slimfarmer.run_photometry(
    ...,
    config = slimfarmer.Config(
        thresh             = 8.0,
        dilation_radius    = 0.07 * u.arcsec,
        fit_dilation_radius= 0.24 * u.arcsec,
        neighbor_subtraction = True,
        ncpus              = 48,
    ),
)
```

---

## Improvements over The Farmer

slimfarmer is a streamlined reimplementation of [The Farmer](https://github.com/astroweaver/the_farmer) (Weaver et al. 2023) with several algorithmic improvements motivated by Roman IMCOM image characteristics.

### 1. Decoupled grouping and fitting regions

**The Farmer** uses a single dilation radius for two distinct purposes: deciding which sources to co-fit simultaneously (grouping) and defining the pixel footprint used for fitting (the fitting region). A large dilation correctly captures profile wings in the fit but also chains together distant sources into oversized groups, causing optimizer failures.

**slimfarmer** separates these with two independent parameters:

| Parameter | Purpose |
|---|---|
| `dilation_radius = 0.07"` | Grouping only — which sources are co-fitted |
| `fit_dilation_radius = 0.24"` | Fitting footprint — how many pixels Tractor sees |

This allows Tractor to fit profile wings (needed for correct flux normalization) without merging distant sources into unstable multi-source groups.

**Why it matters**: a DevGalaxy (de Vaucouleurs) fit restricted to the compact segmap (~0.42" radius) sees only ~55% of the model flux within the fitting region. Tractor compensates by inflating the total amplitude, causing ~55% flux overestimation. With `fit_dilation_radius = 0.24"`, the fitting footprint extends to ~0.9" radius, capturing ~90% of the profile and reducing the bias to <5%.

### 2. Neighbor subtraction

When `fit_dilation_radius` is large, the expanded fitting footprint of one group overlaps pixels belonging to sources in neighboring groups. Those neighbors are not included in the current group's Tractor fit, so their flux is absorbed by the target source.

slimfarmer resolves this with an optional two-pass approach (`neighbor_subtraction = True`):

1. **Pass 1** — fit all groups normally with the expanded footprint → approximate models for every source
2. **Pass 2** — for each group, subtract PSF-convolved model images of nearby sources (from other groups) from the data, then re-run forced photometry

This cleanly removes cross-group contamination from the expanded fitting region.


---

## Key configuration parameters

| Parameter | Default | Description |
|---|---|---|
| `thresh` | `8.0` | SEP detection threshold (× background RMS) |
| `minarea` | `4` | Minimum source area in pixels |
| `dilation_radius` | `0.07"` | Segmap dilation for grouping — controls which sources are co-fitted |
| `fit_dilation_radius` | `0.24"` | Fitting footprint expansion — larger region gives Tractor profile-wing data, reducing flux overestimation for extended sources |
| `group_buffer` | `2.0"` | Extra padding around each group cutout |
| `group_size_limit` | `10` | Groups larger than this are skipped |
| `neighbor_subtraction` | `True` | Second forced-phot pass: subtract neighboring group models before fitting; needed when `fit_dilation_radius` is large |
| `neighbor_radius` | `5.0"` | Radius within which neighbors are subtracted |
| `renorm_psf` | `1.0` | Normalise PSF stamp to this sum; `1.0` = unbiased |
| `ncpus` | `0` | Worker processes; `0` = serial |

---

## Output catalog columns

| Column | Description |
|---|---|
| `id` | Detection ID (1-indexed) |
| `ra`, `dec` | Sky position (deg) |
| `x`, `y` | Pixel position |
| `a`, `b`, `theta` | SEP shape moments |
| `group_id`, `group_pop` | Group assignment and size |
| `{band}_flux` | Fitted flux (image DN) |
| `{band}_flux_err` | Tractor Fisher flux uncertainty (see note below) |
| `{band}_flux_err_des` | DES-style residual-based flux uncertainty |
| `{band}_flux_err_corr` | Corrected flux uncertainty including size-propagation term (recommended) |
| `{band}_mag` | AB magnitude |
| `name` | Model type: `PointSource`, `SimpleGalaxy`, `ExpGalaxy`, `DevGalaxy`, `FixedCompositeGalaxy` |
| `logre`, `logre_err` | Log half-light radius (log arcsec) and its uncertainty from model selection (ExpGalaxy / DevGalaxy) |
| `logre_exp`, `logre_exp_err` | Exp component log half-light radius and uncertainty (FixedCompositeGalaxy) |
| `logre_dev`, `logre_dev_err` | Dev component log half-light radius and uncertainty (FixedCompositeGalaxy) |
| `ee1_err`, `ee2_err` | Ellipticity component uncertainties from model selection |
| `total_rchisq` | Reduced chi² of forced photometry |

### Flux error columns

slimfarmer provides three flux error estimates. They differ in what sources of uncertainty they include:

| Column | Formula | What it captures |
|---|---|---|
| `flux_err` | `sqrt(diag(H⁻¹))` at forced-phot stage | Pixel noise only, for a **fixed** template |
| `flux_err_des` | `sqrt(χ²_local / Σw·T² / dof)` | Inflates by local √rχ² — catches model mismatch or neighbor contamination |
| `flux_err_corr` | `sqrt(flux_err² + (dF/d·logre × logre_err)²)` | Pixel noise **plus** size-propagation uncertainty (recommended) |

**Why `flux_err` underestimates the true scatter**: slimfarmer uses two-pass photometry — model selection fits the galaxy size and shape, then forced photometry freezes them and fits only flux. The Fisher error `flux_err` is the formal Cramér-Rao bound *at the forced-phot stage*, assuming the template (size, shape) is perfectly known. It does not account for the fact that the template was inferred from the same noisy data in model selection.

Because flux and size are correlated within the fitting footprint (a larger template shifts flux to wider annuli, changing the matched-filter response), noise-induced scatter in `logre` from model selection propagates into scatter in the forced flux. This is the **size-propagation** term:

```
σ_prop = |dF/d·logre| × logre_err
flux_err_corr = sqrt(flux_err² + σ_prop²)
```

`dF/d·logre` is estimated numerically by perturbing `logre` by ±0.05 (5% in radius) and evaluating the linear matched-filter flux on the fitting footprint. `logre_err` comes from the model-selection Fisher matrix, stored correctly from that stage.

**Empirical validation** (single bright ExpGalaxy, mag=21, HLR=0.5", read-noise dominated):

| Error estimate | Value | MC ground truth (`std` over 100 realisations) |
|---|---|---|
| `flux_err` | 0.64 DN | — |
| `flux_err_corr` | 0.83 DN | 1.14 DN |

`flux_err_corr` closes ~60% of the gap. The residual underestimate occurs because `logre_err` from the Fisher matrix is a formal lower bound — the actual population scatter in `logre` across noise realisations can be larger due to nonlinear model-selection effects (local minima, profile-type switching). For science use, `flux_err_corr` is the recommended single-object error; for calibrating the full error budget, MC simulations remain the ground truth.

---

## Diagnosing individual sources

### Track model selection stages

```python
from slimfarmer.track import track_source

result = track_source(
    source_id       = 151,
    science_path    = 'roman_image.fits',
    weight_path     = 'roman_weight.fits',
    psf_path        = 'PSF_F158.fits',
    band            = 'F158',
    zeropoint       = 26.511,
    truth_pos_path  = 'galaxy_positions.parquet',   # optional
    truth_flux_path = 'galaxy_fluxes.parquet',      # optional
    truth_flux_col  = 'roman_flux_H158',            # optional
    plot            = True,
    plot_out        = 'source_151.png',
)
# result keys: stages, final_model, obs_flux_nm, true_flux_nm, flux_ratio
```

### Full photometry diagnostic (notebook)

```python
import slimfarmer
from slimfarmer.track import _get_flux_converters
from astropy.io import fits
import pandas as pd

DATA = '/path/to/data'

obs_to_nm, truth_to_nm = _get_flux_converters(DATA+'/roman_image.fits', 'F158')
with fits.open(DATA+'/roman_image.fits') as h:
    oversamplepix = abs(h[0].header['CDELT2']) * 3600.

cfg = slimfarmer.Config()
img = slimfarmer.FarmerImage(
    bands={'F158': {
        'science':   DATA+'/roman_image.fits',
        'weight':    DATA+'/roman_weight.fits',
        'psf':       DATA+'/PSF_F158.fits',
        'zeropoint': 26.511,
    }},
    detection_band='F158', config=cfg,
)
img.detect()

truth = pd.read_parquet(DATA+'/galaxy_fluxes.parquet').merge(
        pd.read_parquet(DATA+'/galaxy_positions.parquet'), on='galaxy_id')

result = slimfarmer.diagnose_source(
    151, img, truth, obs_to_nm, truth_to_nm, oversamplepix)
result['fig'].savefig('diag_151.png', dpi=150, bbox_inches='tight')
```

The diagnostic plot shows five panels: science image, weight map, model image, chi residuals, and radial profile (data vs model).

---

## Testing

Run the bias test against the Roman H158 simulation:

```bash
cd /path/to/Roman_photometry
python test_slimfarmer.py
```

**Success criterion**: `0.97 ≤ median(obs_flux / true_flux) ≤ 1.03`

The test runs the full pipeline on `slimfarmer_outputroman_image.fits`, cross-matches to the truth catalog, and reports the median flux ratio and scatter.

---

## Notebooks

| Notebook | Description |
|---|---|
| `doc/tutorial.ipynb` | Full walkthrough: CPR → images → photometry → diagnostics |
| `doc/demo.ipynb` | Quick demo using pre-prepared FITS files |

```bash
jupyter notebook slimfarmer/doc/tutorial.ipynb
```

## Reading Roman IMCOM CPR files

```python
sci_path, wht_path, psf_path = slimfarmer.prepare_images_from_cpr(
    cpr_path = 'im3x2-H0_00_00.cpr.fits.gz',
    work_dir = 'output/',
    # psf_fwhm_arcsec = 0.240,  # override default Roman H158 PSF FWHM
)
```

Requires `pyimcom` for reading the CPR format.
