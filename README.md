# slimfarmer

A streamlined multi-band photometry pipeline that uses **SEP** for source detection and **Tractor** for galaxy model fitting, with correlated-noise corrections derived from noise realizations. Designed for Roman IMCOM coadds and inspired by [The Farmer](https://github.com/astroweaver/the_farmer) (Weaver et al. 2023).

For algorithmic details, derivations, and validation results, see [`docs/research_note.md`](docs/research_note.md).

---

## Pipeline overview

1. **Detection** — single SEP pass produces a source catalog and segmentation map (no iterative re-detection).
2. **Grouping** — overlapping segmaps (after dilation) are co-fitted as a group to handle blends.
3. **Model selection** — for each group, Tractor tries `PointSource → SimpleGalaxy → ExpGalaxy / DevGalaxy → FixedCompositeGalaxy` and picks the simplest acceptable model. Note that FixedCompositeGalaxy is similar to `Cmodel`, except that the FixedCompositeGalaxy varies both the exponential profile and the dev profile at the same time, while `Cmodel` fits the exponential profile and the dev profile separately.  
4. **Forced photometry** — morphology is frozen, and only flux is refit; an optional second pass subtracts neighbor models (although this is not the default).
5. **Kappa correction** — correlated-noise inflation factor applied per source from noise realizations.

---

## Installation

```bash
cd slimfarmer
pip install -e .
```

**Required:** `numpy`, `astropy>=5.0`, `scipy`, `sep`, `tqdm`, `tractor`
**Optional:** `pathos` (parallel), `galsim` (flux unit conversion), `pyimcom` (CPR file reading)

---

## Quick start

### Single-band

```python
import slimfarmer

cat = slimfarmer.run_photometry(
    science_path='roman_image.fits',
    weight_path ='roman_weight.fits',
    psf_path    ='PSF_F158.fits',
    band        ='F158',
    zeropoint   =26.511,
    output_path ='catalog.fits',
    ncpus       =8,
)
```

### Multi-band

```python
cat = slimfarmer.run_photometry(
    bands={
        'F158': {'science': 'F158.fits', 'weight': 'F158_wht.fits',
                 'psf': 'F158_psf.fits', 'zeropoint': 26.511,
                 'noise_reals': 'F158_nr.fits'},
        'F106': {'science': 'F106.fits', 'weight': 'F106_wht.fits',
                 'psf': 'F106_psf.fits', 'zeropoint': 26.310,
                 'noise_reals': 'F106_nr.fits'},
    },
    detection_band='F158',
    output_path   ='catalog.fits',
    ncpus         =16,
)
```

### Custom configuration

```python
import astropy.units as u

cat = slimfarmer.run_photometry(
    ...,
    config=slimfarmer.Config(
        thresh              =3.0,
        dilation_radius     =0.2 * u.arcsec,
        fit_dilation_radius =0.2 * u.arcsec,
        neighbor_subtraction=True,
        ncpus               =16,
    ),
)
```

---

## Key configuration parameters

| Parameter | Default | Description |
|---|---|---|
| `thresh` | `3.0` | SEP detection threshold (× background RMS) |
| `minarea` | `5` | Minimum source area in pixels |
| `dilation_radius` | `0.2"` | Segmap dilation for grouping |
| `fit_dilation_radius` | `0.2"` | Fitting footprint expansion (captures profile wings) |
| `group_buffer` | `0.01"` | Extra padding around each group cutout |
| `group_size_limit` | `10` | Skip groups larger than this |
| `neighbor_subtraction` | `False` | Two-pass mode that subtracts neighbor models |
| `neighbor_radius` | `5.0"` | Radius for neighbor subtraction |
| `noshot` | `False` | If True, use background-only weights for kappa |
| `ncpus` | `0` | Worker processes (`0` = serial) |
| `paddingpixel` | `34` | Boundary padding for flagging edge sources |
| `save_model_image` | `True` | Write `<output>_model.fits` and `<output>_residual.fits` |

See [`docs/research_note.md`](docs/research_note.md) for the rationale behind decoupling `dilation_radius` and `fit_dilation_radius`.

---

## Output catalog

### Position and shape
| Column | Description |
|---|---|
| `id` | Detection ID (1-indexed) |
| `ra`, `dec`, `ra_err`, `dec_err` | Sky position and uncertainty (deg) |
| `x`, `y` | Pixel position |
| `a`, `b`, `theta` | SEP shape moments |
| `group_id`, `group_pop` | Group assignment and size |
| `flag` | Bitwise OR of detection flags (see below) |
| `name` | Model type: `PointSource`, `SimpleGalaxy`, `ExpGalaxy`, `DevGalaxy`, `FixedCompositeGalaxy` |
| `logre`, `logre_err` | Log half-light radius and uncertainty (single-component galaxies) |
| `logre_exp`, `logre_dev` | Component sizes (composite galaxies) |
| `total_rchisq` | Reduced chi² of forced photometry |

### Flux columns (per band)
| Column | Description |
|---|---|
| `{band}_flux` | Fitted flux (image DN) |
| `{band}_flux_err` | **Recommended.** Marginalized Fisher error including shot noise + size propagation + kappa |
| `{band}_flux_err_noshot` | Same, background noise only |
| `{band}_flux_err_des` | DES residual-based error × marginalization ratio + size propagation + kappa |
| `{band}_flux_err_kappa` | Correlated-noise inflation factor applied to all error columns |
| `{band}_flux_err_tractor_origin` | Raw fixed-template Fisher error (for diagnostics) |

See [`docs/research_note.md`](docs/research_note.md) for derivations and MC validation.

### Detection flags
| Bit | Value | Source |
|---|---|---|
| 0 | 1 | Has neighbors (SEP) |
| 1 | 2 | Was blended (SEP) |
| 2 | 4 | Saturated pixel (SEP) |
| 3 | 8 | Truncated at boundary (SEP) |
| 4 | 16 | Aperture data corrupted (SEP) |
| 5 | 32 | Isophotal data corrupted (SEP) |
| 6 | 64 | Memory overflow during deblending (SEP) |
| 7 | 128 | Memory overflow during extraction (SEP) |
| 8 | 256 (`0x0100`) | Within `paddingpixel` of image edge (slimfarmer) |

---

## Truth matching

```python
from slimfarmer.utils import match_spatial, match_spatial_mag

# Pure spatial matching (S)
idx, sep = match_spatial(cat['ra'], cat['dec'],
                         truth['ra'], truth['dec'])

# Spatial + magnitude matching (S+M), single band
idx, sep, dmag = match_spatial_mag(
    cat['ra'], cat['dec'], cat['mag_F158'],
    truth['ra'], truth['dec'], truth['mag_F158'],
    radius_arcsec=0.6, mag_thresh=1.0,
)

# Spatial + multi-band magnitude matching
idx, sep, dmag = match_spatial_mag(
    cat['ra'], cat['dec'],
    np.column_stack([cat['mag_Y'], cat['mag_J'], cat['mag_H']]),
    truth['ra'], truth['dec'],
    np.column_stack([truth['mag_Y'], truth['mag_J'], truth['mag_H']]),
    radius_arcsec=0.6, mag_thresh=1.0,
)
```

The S+M metric is `sqrt(Σ Δmag_i²)`. Unmatched sources get `idx = -1`.

---

## Reading Roman IMCOM CPR files

```python
sci, wht, psf, eff_gain, noise_reals = slimfarmer.prepare_images_from_cpr(
    cpr_path='im3x2-H1_00_00.cpr.fits.gz',
    work_dir='output/',
)
```

Requires `pyimcom`.

---

## Diagnostics

### Track a single source through model selection

```python
from slimfarmer.track import track_source

result = track_source(
    source_id=151,
    science_path='roman_image.fits',
    weight_path ='roman_weight.fits',
    psf_path    ='PSF_F158.fits',
    band        ='F158',
    zeropoint   =26.511,
    plot=True, plot_out='source_151.png',
)
```

### Full diagnostic plot

```python
result = slimfarmer.diagnose_source(
    151, img, truth, obs_to_nm, truth_to_nm, oversamplepix)
result['fig'].savefig('diag_151.png', dpi=150)
```

Shows science image, weight map, model, chi residuals, and radial profile.

---

## Notebooks

| Notebook | Description |
|---|---|
| `docs/tutorial.ipynb` | End-to-end walkthrough: CPR → images → photometry → diagnostics |
| `docs/demo.ipynb` | Quick demo with pre-prepared FITS files |
| `docs/Flux_error_validation_colab.ipynb` | Flux error MC validation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Roman-HLIS-Cosmology-PIT/slimfarmer/blob/main/doc/Flux_error_validation_colab.ipynb) |

---

## Testing

```bash
python test_slimfarmer.py
```

**Success criterion:** `0.97 ≤ median(obs_flux / true_flux) ≤ 1.03`

---

## Documentation

- **[`docs/research_note.md`](docs/research_note.md)** — algorithmic details: decoupled grouping/fitting regions, marginalized Fisher errors, kappa correction derivation, and MC validation results.
