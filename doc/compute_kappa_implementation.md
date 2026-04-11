# `compute_kappa` — implementation notes

Companion to `correlated_noise_correction.md`, which explains *why* we need
a correction factor `kappa` on the diagonal-Fisher flux error when pixels
are correlated. This doc explains *how* `compute_kappa` is implemented in
`slimfarmer/image.py:550`, and — just as importantly — how it avoids
blowing up memory when you have thousands of sources and five bands.

Source of truth: `slimfarmer/image.py` (`FarmerImage.compute_kappa`) and
`slimfarmer/_group.py` (`SourceGroup._cache_kappa_data`).

---

## 1. What we're computing, in one line

For each source `s` and band `b`:

```
kappa_b(s) = sqrt( max( h^T R h / D , 1 ) )
```

where (all per source, per band):

- `T(y,x)` is the unit template — the PSF-convolved source profile,
  **normalised** so that if you scaled it by `flux` you'd recover the
  model image. In the code this is `unit_t = engine.getModelImage() / flux`.
- `h(y,x) = sqrt(invvar_bg(y,x)) * T(y,x)` — the template weighted by the
  independent-pixel noise, on the *background-only* inverse-variance.
- `D = sum_i invvar_bg_i * T_i^2` — the diagonal-Fisher denominator (the
  quantity that would give you the flux error if the pixels were
  independent: `flux_err_diag = 1/sqrt(D)`).
- `R(dy, dx)` is the 2D pixel-noise correlation function, estimated from
  the noise realizations and normalised so that `R(0, 0) = 1` (it lives
  in `band_config[band]['noise_corr']`).

The result is stored on each source model as
`model.flux_err_noisereal_kappa[band]`, and used downstream to rescale
`flux_err_des`.

`kappa == 1` means pixels are uncorrelated → no correction. For Dec25
IMCOM coadds kappa ends up ~5.

---

## 2. Why the naive implementation is expensive

`h^T R h` is the contraction of a vector with a matrix. If you wrote it
literally:

```python
kappa_sq = h.flatten() @ R_full @ h.flatten()    # DON'T DO THIS
```

then `R_full` is an `(Npix × Npix)` matrix. For an IMCOM block of
`2108²` ≈ 4.4 M pixels, that's 4.4 M × 4.4 M ≈ 1.9 × 10¹³ entries —
about **150 TB in float64**. Completely unusable. Two structural
observations let us avoid ever materialising it:

**(a) R is translation-invariant.** Background noise statistics don't
depend on where you are in the image, so `Cov(pixel_j, pixel_k)`
depends only on the offset `(dy, dx) = k − j`, not on `j` and `k`
separately. All the information in that huge matrix fits in a 2D array
of the same shape as one image: the correlation function `r(dy, dx)`.
That's `noise_corr`, stored once per band under
`band_config[band]['noise_corr']`.

**(b) Translation-invariant matrix-vector products are convolutions.**
Given translation-invariance, the quadratic form simplifies to

```
h^T R h = sum_{y,x} h(y,x) * (R * h)(y,x)
```

where `R * h` denotes 2D convolution. Convolutions are cheap via FFT:

```python
Rk = rfft2(noise_corr)
Hk = rfft2(H)
conv = irfft2(Rk * Hk, s=(ny, nx))
hTRh = (H * conv).sum()
```

That replaces the `O(Npix²)` matmul with `O(Npix * log Npix)` FFTs on
arrays of *one-image* size. This is the core of why `compute_kappa` is
tractable at all.

---

## 3. The per-source cache (the main memory trick)

Even after (a) and (b), storing the full 2D template `H` for every
source is still wasteful. `H` has the same shape as the image (call it
`ny × nx`), but is **zero almost everywhere** — a galaxy's PSF-convolved
profile touches maybe ~few-hundred pixels out of millions. So instead of
pickling a dense `(ny, nx)` template per source (that would be tens of
megabytes per source times thousands of sources — easily hundreds of GB
per tile), the group-fitting code caches only the nonzero values and
their indices.

This happens in `SourceGroup._cache_kappa_data` (`_group.py:837`):

```python
src_copy = copy.deepcopy(model)
temp_engine = Tractor([img], [src_copy])
temp_engine.bands = [band]
temp_engine.freezeParam('images')
unit_t = temp_engine.getModelImage(0) / flux     # normalised template

D = float(np.sum(invvar_bg * unit_t ** 2))        # scalar
h = np.sqrt(invvar_bg) * unit_t                   # 2D but mostly zero
nzy, nzx = np.nonzero(h)                          # short index arrays

model._kappa_cache[band] = {
    'h_vals': h[nzy, nzx].astype(np.float32),     # only the nonzero values
    'nzy':    nzy.astype(np.int16),               # y-coordinate of each
    'nzx':    nzx.astype(np.int16),               # x-coordinate of each
    'D':      D,                                  # scalar
}
```

What gets saved per source per band:

| field    | shape       | dtype    | typical size           |
|----------|-------------|----------|------------------------|
| `h_vals` | `(Nnz,)`    | float32  | a few KB (Nnz ~ 10³)   |
| `nzy`    | `(Nnz,)`    | int16    | 2× smaller than int64  |
| `nzx`    | `(Nnz,)`    | int16    |                        |
| `D`      | scalar      | float64  | 8 bytes                |

`int16` here is safe because IMCOM blocks are 2108 px on a side, which
fits easily in ±32767. Using `int16` instead of `int64` cuts the index
memory by 4×.

Contrast with the alternative of storing the full `(ny, nx)` `h` array:
float64 at 2354 × 2354 = **44 MB per source per band**. Over ~1000
sources × 5 bands that's ~220 GB, which you cannot pickle to a worker,
let alone keep in RAM. The sparse cache typically comes out to a few KB
per source per band — a ~10⁴× reduction.

The comment on the method says it explicitly:

> Only stores 1D arrays (h_vals, nzy, nzx) and scalar D — NOT the large
> pairwise dy/dx matrices, to keep pickle size small.

"Pickle size" matters because fitted group results are shipped back from
worker processes to the main process, so anything attached to the model
objects has to round-trip through pickle. Keeping `_kappa_cache` tiny
keeps that cheap.

---

## 4. The `compute_kappa` loop itself

Once every source has its tiny `_kappa_cache`, the main-process loop in
`FarmerImage.compute_kappa` reconstructs `H` on the fly and evaluates
the convolution:

```python
def compute_kappa(self):
    import gc

    for band in self.bands:
        bc = self.band_config[band]
        noise_corr = bc.get('noise_corr')
        if noise_corr is None:
            continue
        ny, nx = noise_corr.shape

        # (A) FFT of the noise correlation, hoisted out of the per-source loop
        Rk = np.fft.rfft2(noise_corr)

        for source_id, model in tqdm(self.model_catalog.items()):
            if not hasattr(model, 'flux_err_noisereal_kappa'):
                model.flux_err_noisereal_kappa = {}

            cache = getattr(model, '_kappa_cache', {}).get(band)
            if cache is None:
                model.flux_err_noisereal_kappa[band] = 1.0
                continue

            # (B) Rehydrate H from the sparse cache
            h_vals = cache['h_vals'].astype(np.float64)
            nzy    = cache['nzy'].astype(np.intp)
            nzx    = cache['nzx'].astype(np.intp)
            D      = cache['D']

            H = np.zeros((ny, nx), dtype=np.float64)
            np.add.at(H, (nzy % ny, nzx % nx), h_vals)

            # (C) FFT-based evaluation of h^T R h
            Hk   = np.fft.rfft2(H)
            conv = np.fft.irfft2(Rk * Hk, s=(ny, nx))
            hTRh = float(np.sum(H * conv))

            kappa_sq = hTRh / D
            model.flux_err_noisereal_kappa[band] = np.sqrt(max(kappa_sq, 1.0))

        # (D) free the band's big arrays before touching the next band
        del noise_corr, Rk; gc.collect()
```

Four things to notice, each of them a deliberate memory or performance
choice:

### (A) Hoisting `Rk`

`rfft2(noise_corr)` depends **only on the band**, not on the source. The
old code computed it inside the per-source loop; moving it outside gave
a ~33% speedup (see the project memory entry). It also means we hold
exactly one `Rk` array in RAM during the source loop, not one per
iteration. For each band, `noise_corr` and its FFT together are a few
hundred MB at most; they're the dominant per-band working set.

### (B) Rehydrating H sparsely

The sparse `(h_vals, nzy, nzx)` triple gets scattered into a fresh `H`
via `np.add.at(H, (nzy % ny, nzx % nx), h_vals)`. Two details:

- `np.add.at` (as opposed to `H[nzy, nzx] = h_vals`) is the *unbuffered*
  assignment: it correctly handles any accidental duplicate index, even
  though in practice `nonzero` won't produce duplicates.
- The `% ny` / `% nx` wrap is defensive — `noise_corr` has the shape of
  one image, and for stitched inputs the cache was populated using
  whatever the group's local canvas was. The modulo guards against a
  mismatch rather than forcing the two shapes to agree by construction.

Allocating `H = np.zeros((ny, nx))` each iteration is O(Npix) — it's not
free, but it's the right tradeoff: the alternative is to keep a single
pre-allocated scratch buffer, which complicates cleanup and introduces
state that's easy to get wrong when the inner loop raises. For kappa
computation it's not the bottleneck.

### (C) `h^T R h` via FFT

The three FFT lines are the whole point. `rfft2` → multiply → `irfft2`
is a linear-in-Npix-log-Npix correlation evaluation. After that, `hTRh`
is just `(H * conv).sum()`.

`max(kappa_sq, 1.0)` floors the correction at 1 — we never use kappa to
*shrink* an error bar, only to inflate it. In principle tiny numerical
noise could produce `kappa_sq < 1` for completely isolated sources
where `R ≈ delta`; the floor removes that edge case.

### (D) Free the per-band workspace between bands

After each band's loop, `del noise_corr, Rk; gc.collect()` releases the
large band-level arrays so the next band's `noise_corr` doesn't live
alongside the previous one. With five bands and hundreds of MB each,
skipping this would keep peak RSS ~5× higher than necessary. The
`gc.collect()` is belt-and-braces: `del` drops the reference, but
CPython's generational GC may not immediately return the pages to the
allocator on its own.

Note what is **not** freed: `model._kappa_cache` stays on each source.
That's deliberate — it's tiny per source, and it's what lets a second
`compute_kappa` call be almost free.

---

## 5. Back-of-envelope memory vs. a naive implementation

For a single IMCOM block with ~5000 detected sources, 5 bands, and
`2108² ≈ 4.4 M` pixels:

| approach                                      | peak working set                 |
|-----------------------------------------------|----------------------------------|
| Naive: store `R_full` once                    | ~150 TB (R_full alone)           |
| Store R as `noise_corr` + per-source `H` dense | ~44 MB × 5000 × 5 ≈ 1 TB         |
| Current: `noise_corr` + sparse cache + per-band del | ~300 MB (one band's `noise_corr` + `Rk`) + ~tens of MB of caches |

The current scheme is literally ~four orders of magnitude leaner than
storing dense templates, and seven orders of magnitude leaner than
materialising the full covariance matrix. Neither alternative is just
"wasteful" — both are impossible on any node you'd actually run on.

---

## 6. Where the pieces live

- **Build the per-band `noise_corr`:** `FarmerImage.__init__`,
  `image.py:126` — reads the noise realisation cube, estimates a
  smoothed per-pixel sigma, computes `|FFT|²` averaged over realisations,
  and inverse-FFTs to get `r(dy, dx)`.
- **Populate each source's `_kappa_cache`:**
  `SourceGroup._cache_kappa_data`, `_group.py:837` — runs once per
  group, per band, as part of the fit finalisation. This is the
  memory-saving hot spot.
- **Apply the correction:** `FarmerImage.compute_kappa`, `image.py:550`
  — the loop above. Runs once after all groups are fit.
- **Consume the correction:** downstream code in `image.py` /
  `utils.py` scales `flux_err_des` by `kappa` before writing the
  catalog. Search for `flux_err_noisereal_kappa` to follow the trail.

---

## 7. When it might matter to revisit this

- **If sources span much larger groups than one IMCOM block**, the
  `int16` indices in the cache are no longer enough — raise them to
  `int32` (still 2× smaller than `int64`).
- **If you start modelling stitched canvases bigger than ±32767 px** on
  a side — same issue.
- **If you add per-pixel sigma variation that breaks translation
  invariance** (e.g. gradient-corrected noise models), `noise_corr`
  stops being a sufficient summary and you lose the FFT shortcut.
  You'd have to factor `h = sqrt(invvar) * T` and push the spatially
  varying part into `h` rather than `R`.
- **If you observe `kappa` spending a lot of time in the per-source
  FFT**, consider doing a single batched FFT: stack several sources'
  `H` into one 3D array and FFT along the spatial axes once. This
  trades memory for throughput and is worth it only if the per-source
  allocation turns out to dominate.

Otherwise, `compute_kappa` is in a good place: correct, tractable, and
explicit about the tradeoffs that make it tractable.
