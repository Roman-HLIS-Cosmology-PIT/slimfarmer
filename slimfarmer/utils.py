"""Utility functions for slimfarmer — adapted from the_farmer/farmer/utils.py."""

import os
import logging
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.ndimage import label, binary_dilation, binary_fill_holes
import astropy.units as u
from collections import OrderedDict

from tractor.ellipses import EllipseESoft, EllipseE
from tractor.galaxy import ExpGalaxy, SoftenedFracDev
from tractor import PointSource, DevGalaxy, FixedCompositeGalaxy, Fluxes
from astrometry.util.util import Tan
from tractor import ConstantFitsWcs
from astropy.wcs import WCS
from astropy.nddata import Cutout2D



class SimpleGalaxy(ExpGalaxy):
    """Exponential galaxy with fixed 0.45" half-light radius and circular shape."""
    shape = EllipseE(0.45, 0., 0.)

    def __init__(self, *args):
        super().__init__(*args)

    def getName(self):
        return 'SimpleGalaxy'

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1)

    def isParamFrozen(self, pname):
        if pname == 'shape':
            return True
        return super().isParamFrozen(pname)


def read_wcs(wcs, scl=1):
    """Convert an astropy WCS to a Tractor ConstantFitsWcs.

    Re-centres the TAN approximation on the image to avoid accumulating
    TAN-vs-STG projection error when CRPIX is far from the image (e.g. a
    small cutout from a large IMCOM mosaic where CRPIX ~ 40000 px away).

    The CD matrix is recomputed for the new CRVAL using finite differences
    and expressed in FITS intermediate coordinates:
        d(xi)  / d(pix) = +d(RA)  / d(pix) * cos(CRVAL_Dec)
        d(eta) / d(pix) =  d(Dec) / d(pix)
    """
    t = Tan()
    h, w = wcs.array_shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0          # 0-indexed image centre
    ra_c,  dec_c  = wcs.all_pix2world(cx,     cy, 0)
    ra_xp, dec_xp = wcs.all_pix2world(cx + 1, cy, 0)
    ra_yp, dec_yp = wcs.all_pix2world(cx, cy + 1, 0)
    cos_dec = np.cos(np.deg2rad(dec_c))
    # FITS CD matrix: d(xi, eta) / d(pixel), xi = +(RA-CRVAL_RA)*cos(CRVAL_Dec)
    t.set_crpix((cx * scl) + 1, (cy * scl) + 1)     # FITS 1-indexed
    t.set_crval(float(ra_c), float(dec_c))
    t.set_cd((ra_xp - ra_c) * cos_dec / scl, (ra_yp - ra_c) * cos_dec / scl,
             (dec_xp - dec_c)          / scl, (dec_yp - dec_c)          / scl)
    t.set_imagesize(int(w * scl), int(h * scl))
    return ConstantFitsWcs(t)


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = [int(w / 2), int(h / 2)]
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = np.zeros((h, w), dtype=int)
    mask[dist <= radius] = 1
    return mask


def get_fwhm(img):
    dx, dy = np.nonzero(img > np.nanmax(img) / 2.)
    try:
        fwhm = np.mean([dx[-1] - dx[0], dy[-1] - dy[0]])
    except (IndexError, ValueError):
        fwhm = np.nan
    return float(np.nanmin([1.0, fwhm]))


def clean_catalog(catalog, mask, segmap=None):
    """Remove sources in masked pixels; renumber segmap in-place."""
    if segmap is not None:
        assert mask.shape == segmap.shape
    x = np.round(catalog['x']).astype(int)
    y = np.round(catalog['y']).astype(int)
    h, w = mask.shape
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    keep = np.zeros(len(catalog), dtype=bool)
    keep[valid] = ~mask[y[valid], x[valid]]
    cleancat = catalog[keep]
    if segmap is not None:
        max_seg = max(len(catalog), int(np.max(segmap)))
        mapping = np.zeros(max_seg + 1, dtype=segmap.dtype)
        kept = np.where(keep)[0]
        mapping[kept + 1] = np.arange(1, len(kept) + 1, dtype=segmap.dtype)
        segmap[:] = mapping[segmap]
        return cleancat, segmap
    return cleancat


def dilate_and_group(catalog, segmap, radius=0, fill_holes=False):
    """Morphological dilation + union-find grouping, same logic as the_farmer."""
    logger = logging.getLogger('slimfarmer.grouping')
    segmask = (segmap > 0).astype(int)

    if radius is not None and radius > 0:
        struct = create_circular_mask(2 * radius, 2 * radius, radius=radius)
        segmask = binary_dilation(segmask, structure=struct).astype(int)
    if fill_holes:
        segmask = binary_fill_holes(segmask).astype(int)

    groupmap, n_groups = label(segmask)

    seg_mask = segmap > 0
    seg_vals = segmap[seg_mask]
    grp_vals = groupmap[seg_mask]
    seg_to_groups = {}
    for seg_id in np.unique(seg_vals):
        grps = np.unique(grp_vals[seg_vals == seg_id])
        if len(grps) > 1:
            seg_to_groups[seg_id] = grps
    group_mapping = np.arange(n_groups + 1)
    for seg_id, grps in seg_to_groups.items():
        primary = grps[0]
        for bad in grps[1:]:
            group_mapping[bad] = primary
    for i in range(1, len(group_mapping)):
        root = i
        while group_mapping[root] != root:
            root = group_mapping[root]
        group_mapping[i] = root
    groupmap = group_mapping[groupmap]

    unique_groups = np.unique(groupmap[groupmap > 0])
    final_mapping = np.zeros(int(groupmap.max()) + 1, dtype=groupmap.dtype)
    final_mapping[unique_groups] = np.arange(1, len(unique_groups) + 1)
    groupmap = final_mapping[groupmap]

    segid, idx = np.unique(segmap.flatten(), return_index=True)
    group_ids = groupmap.flatten()[idx[segid > 0]]
    unique_gids, gid_counts = np.unique(group_ids, return_counts=True)
    gid_to_count = dict(zip(unique_gids, gid_counts))
    group_pops = np.array([gid_to_count[gid] for gid in group_ids], dtype=np.int16)

    logger.debug(f'Found {int(groupmap.max())} groups for {int(segmap.max())} sources.')
    return group_ids, group_pops, groupmap


def segmap_to_dict(segmap):
    """Convert 2-D segmap array → dict {source_id: (y_pixels, x_pixels)}."""
    y, x = np.nonzero(segmap)
    all_segs = segmap[y, x]
    unique_segs, inverse = np.unique(all_segs, return_inverse=True)
    out = {}
    for idx, seg_id in enumerate(unique_segs):
        sel = np.where(inverse == idx)[0]
        out[int(seg_id)] = (y[sel], x[sel])
    return out


def set_priors(model, priors):
    """Freeze or add Gaussian priors to a Tractor model (same as the_farmer)."""
    if priors is None:
        return model
    params = model.getNamedParams()
    for name, idx in params.items():
        if name == 'pos':
            rule = priors.get('pos', 'none')
            if rule in ('fix', 'freeze'):
                model[idx].freezeAllParams()
            elif rule != 'none':
                sigma = rule.to(u.deg).value
                ra_val = float(model[idx][0])
                dec_val = float(model[idx][1])
                if np.isfinite(ra_val) and np.isfinite(dec_val):
                    model[idx].addGaussianPrior('ra', mu=ra_val, sigma=sigma)
                    model[idx].addGaussianPrior('dec', mu=dec_val, sigma=sigma)
        elif name == 'fracDev':
            if priors.get('fracDev', 'none') in ('fix', 'freeze'):
                model.freezeParam(idx)
        elif name in ('shape', 'shapeDev', 'shapeExp'):
            if priors.get('shape', 'none') in ('fix', 'freeze'):
                for i, _ in enumerate(model[idx].getParamNames()):
                    if i != 0:
                        model[idx].freezeParam(i)
            reff_rule = priors.get('reff', 'none')
            if reff_rule in ('fix', 'freeze'):
                model[idx].freezeParam(0)
            elif isinstance(reff_rule, tuple):
                # (Quantity, 'fix') → set logre to log(value_arcsec) then freeze
                target_qty, action = reff_rule
                reff_arcsec = target_qty.to(u.arcsec).value if hasattr(target_qty, 'to') else float(target_qty)
                logre_target = float(np.log(reff_arcsec))
                model[idx].logre = logre_target
                if action in ('fix', 'freeze'):
                    model[idx].freezeParam(0)
            elif reff_rule != 'none':
                sigma_arcsec = reff_rule.to(u.arcsec).value
                logre_val = float(model[idx].logre)
                if np.isfinite(logre_val):
                    reff_current = max(np.exp(logre_val), 1e-3)
                    sigma_logre = sigma_arcsec / reff_current
                    model[idx].addGaussianPrior('logre', mu=logre_val, sigma=sigma_logre)
    return model


def get_params(model, band, zeropoint):
    """Extract fit parameters from a Tractor model into an OrderedDict."""
    source = OrderedDict()

    name = 'PointSource' if isinstance(model, PointSource) else model.name
    source['name'] = name

    source['ra'] = model.pos.ra * u.deg
    source['ra_err'] = np.sqrt(abs(model.variance.pos.ra)) * u.deg
    source['dec'] = model.pos.dec * u.deg
    source['dec_err'] = np.sqrt(abs(model.variance.pos.dec)) * u.deg

    if hasattr(model, 'statistics'):
        for stat, val in model.statistics.items():
            if stat not in ('model', 'variance') and np.isscalar(val):
                source[f'total_{stat}'] = val

    if isinstance(model, (ExpGalaxy, DevGalaxy)) and name not in ('SimpleGalaxy',):
        var_shape = model.variance.shape
        source['logre'] = model.shape.logre
        source['logre_err'] = np.sqrt(abs(var_shape.logre))
        source['reff'] = np.exp(model.shape.logre) * u.arcsec
        source['ellip'] = model.shape.e
        source['ellip_err'] = np.sqrt(abs(var_shape.e))
        source['ee1'] = model.shape.ee1
        source['ee1_err'] = np.sqrt(abs(var_shape.ee1))
        source['ee2'] = model.shape.ee2
        source['ee2_err'] = np.sqrt(abs(var_shape.ee2))
        boa = (1. - abs(model.shape.e)) / (1. + abs(model.shape.e))
        source['ba'] = boa
        source['pa'] = (90. + np.rad2deg(model.shape.theta)) * u.deg
    elif isinstance(model, FixedCompositeGalaxy):
        source['fracdev'] = model.fracDev.clipped()
        source['softfracdev'] = model.fracDev.getValue()
        source['logre_exp'] = model.shapeExp.logre
        source['ee1_exp']   = model.shapeExp.ee1
        source['ee2_exp']   = model.shapeExp.ee2
        source['logre_dev'] = model.shapeDev.logre
        source['ee1_dev']   = model.shapeDev.ee1
        source['ee2_dev']   = model.shapeDev.ee2
        try:
            var_exp = model.variance.shapeExp
            var_dev = model.variance.shapeDev
            source['logre_exp_err'] = np.sqrt(abs(var_exp.logre))
            source['ee1_exp_err']   = np.sqrt(abs(var_exp.ee1))
            source['ee2_exp_err']   = np.sqrt(abs(var_exp.ee2))
            source['logre_dev_err'] = np.sqrt(abs(var_dev.logre))
            source['ee1_dev_err']   = np.sqrt(abs(var_dev.ee1))
            source['ee2_dev_err']   = np.sqrt(abs(var_dev.ee2))
        except Exception:
            for key in ('logre_exp_err', 'ee1_exp_err', 'ee2_exp_err',
                        'logre_dev_err', 'ee1_dev_err', 'ee2_dev_err'):
                source[key] = 0.0

    try:
        flux = model.getBrightness().getFlux(band)
        flux_var = model.variance.getBrightness().getFlux(band)
        flux_err_tractor = np.sqrt(abs(flux_var)) if flux_var > 0 else 0.0
    except Exception:
        flux, flux_err_tractor = 0.0, 0.0

    flux_err_corr = flux_err_tractor
    if hasattr(model, 'flux_err_corr'):
        flux_err_corr = model.flux_err_corr.get(band, flux_err_tractor)

    flux_err_des_raw = 0.0
    if hasattr(model, 'flux_err_des'):
        flux_err_des_raw = model.flux_err_des.get(band, 0.0)

    sigma_prop_sq = max(0.0, flux_err_corr ** 2 - flux_err_tractor ** 2)

    flux_err_noshot_raw = flux_err_tractor
    if hasattr(model, 'flux_err_noshot_raw'):
        flux_err_noshot_raw = model.flux_err_noshot_raw.get(band, flux_err_tractor)
    flux_err_noshot = float(np.sqrt(flux_err_noshot_raw ** 2 + sigma_prop_sq))

    flux_err_shot_raw = flux_err_tractor
    if hasattr(model, 'flux_err_shot_raw'):
        flux_err_shot_raw = model.flux_err_shot_raw.get(band, flux_err_tractor)
    flux_err_shot = float(np.sqrt(flux_err_shot_raw ** 2 + sigma_prop_sq))

    # Scale flux_err_des by the marginalization ratio so it captures both
    # model mismatch (chi2/dof) AND parameter-marginalization inflation.
    flux_err_shot_fixed = flux_err_tractor
    if hasattr(model, 'flux_err_shot_raw_fixed'):
        flux_err_shot_fixed = model.flux_err_shot_raw_fixed.get(band, flux_err_tractor)
    marg_ratio = flux_err_shot_raw / flux_err_shot_fixed if flux_err_shot_fixed > 0 else 1.0
    flux_err_des = float(np.sqrt((flux_err_des_raw * marg_ratio) ** 2 + sigma_prop_sq))

    # Correlated-noise correction from noise realizations.
    # When noshot=True, pass background-only noise realizations.
    # When noshot=False, pass noise realizations that include shot noise.
    kappa = 1.0
    if hasattr(model, 'flux_err_noisereal_kappa'):
        kappa = model.flux_err_noisereal_kappa.get(band, 1.0)
    flux_err_shot *= kappa
    flux_err_noshot *= kappa
    flux_err_des *= kappa

    source[f'{band}_flux'] = flux
    source[f'{band}_flux_err'] = flux_err_shot
    source[f'{band}_flux_err_des'] = flux_err_des
    source[f'{band}_flux_err_noshot'] = flux_err_noshot
    source[f'{band}_flux_err_tractor_origin'] = flux_err_tractor
    source[f'{band}_flux_err_kappa'] = kappa
    #source[f'{band}_flux_ujy'] = flux * 10 ** (-0.4 * (zeropoint - 23.9))
    #if flux > 0:
    #    source[f'{band}_mag'] = -2.5 * np.log10(flux) + zeropoint
    #else:
    #    source[f'{band}_mag'] = np.nan

    return source


def prepare_images_from_cpr(cpr_path, work_dir,
                            psf_fwhm_arcsec=None,
                            sci_name='roman_image.fits',
                            wht_name='roman_weight.fits',
                            psf_name='PSF_F158.fits',
                            eff_gain_name='roman_eff_gain.fits',
                            noise_reals_name='roman_noise_reals.fits',
                            pix_size= 0.11,
                            gain = 1.458,
                            exptime = 107.52398,
                            overwrite=True,
                            truth=False,
                            positionsize=None, realization=False, mask=None
                            ):
    """
    Extract science image, weight map, PSF stamp, and effective gain map from
    an IMCOM CPR FITS file.

    The CPR format stores the IMCOM output as a compressed multi-layer cube:
      HDU 0, layer  0 : science image (background-subtracted, DN/px/s)
      HDU 0, layer  1 : Poisson signal map
      HDU 0, layer 21 : noise realisation (used to calibrate the variance scale)
      HDU 6           : Sigma map (correlated noise amplitude, in log10-bels)
      HDU 8           : Neff map  (effective coverage, in log10-bels)

    Weight map = 1 / correlated_variance (background noise only; no source shot
    noise).  Source shot noise is intentionally excluded here so that it can be
    estimated from the fitted model during photometry.

    Effective gain map (eff_gain): converts a model flux value to its expected
    Poisson variance as  poisson_var = model_flux / eff_gain.
    In scaled image units: eff_gain = 107.52398 * Neff / scaling_factor,
    where scaling_factor = gain / (pix_size / native_pix_scale)².

    PSF is approximated as a 2-D Gaussian with the Roman H158 effective FWHM.
    The stamp is normalised to sum = 1.

    Parameters
    ----------
    cpr_path        : str   — path to *.cpr.fits.gz
    work_dir        : str   — output directory (created if absent)
    psf_fwhm_arcsec : float — Gaussian PSF FWHM in arcsec.
                              Default: Roman H158 effective PSF FWHM (~0.240").
    sci_name        : str   — output filename for science image
    wht_name        : str   — output filename for weight map (background only)
    psf_name        : str   — output filename for PSF stamp
    eff_gain_name   : str   — output filename for effective gain map
    overwrite       : bool  — if False, skip calculation when all output files exist

    Returns
    -------
    sci_path, wht_path, psf_path, eff_gain_path : str
    """
    os.makedirs(work_dir, exist_ok=True)

    sci_path         = os.path.join(work_dir, sci_name)
    wht_path         = os.path.join(work_dir, wht_name)
    psf_path         = os.path.join(work_dir, psf_name)
    eff_gain_path    = os.path.join(work_dir, eff_gain_name)
    noise_reals_path = os.path.join(work_dir, noise_reals_name)

    if not overwrite and all(os.path.exists(p) for p in (sci_path, wht_path, psf_path, eff_gain_path, noise_reals_path)):
        logger = logging.getLogger('slimfarmer')
        logger.info(f'Skipping CPR preparation — all outputs exist in {work_dir}')
        return sci_path, wht_path, psf_path, eff_gain_path, noise_reals_path

    try:
        from pyimcom.compress.compressutils import ReadFile
        from pyimcom.diagnostics.outimage_utils.helper import HDU_to_bels
    except ImportError as e:
        raise ImportError(
            'pyimcom is required to read CPR files. '
            'Install it from https://github.com/hirata10/furry-parakeet'
        ) from e

    from astropy.modeling.models import Gaussian2D


    # ── Read CPR ──────────────────────────────────────────────────────────────
    cpr    = ReadFile(cpr_path)
    if truth:
        sci    = cpr[0].data[0][1].astype(np.float32)
    elif realization:
        sci    = cpr[0].data[0][1].astype(np.float32)+cpr[0].data[0][26].astype(np.float32)
    else:
        sci    = cpr[0].data[0][0].astype(np.float32)
    header = cpr[0].header

    pix_scale = abs(header['CDELT2']) * 3600.  # arcsec/px

    # ── Variance map ──────────────────────────────────────────────────────────
    Sigma       = 10 ** (HDU_to_bels(cpr[6]) * cpr[6].data[0])
    Neff        = 10 ** (HDU_to_bels(cpr[8]) * cpr[8].data[0])
    scalefactor = np.sum(cpr[0].data[0][21] ** 2)

    factor      = gain / (pix_size / pix_scale) ** 2
    sci         = sci * factor

    # ── Noise realizations (layers 20-23) ────────────────────────────────
    noise_reals = np.stack(
        [cpr[0].data[0][k].astype(np.float32) for k in range(24, 28)]
    ) * factor  # (4, Ny, Nx)

    # Background-only (correlated) variance — source shot noise excluded.
    # Shot noise from sources must be estimated from the model during fitting.
    corr_var    = ((scalefactor / np.sum(Sigma)) * Sigma).astype(np.float32) * factor ** 2
    wht         = np.where(corr_var > 0, 1.0 / corr_var, 0.).astype(np.float32)
    if mask is not None:
        wcs2d = WCS(header).celestial
        ny, nx = sci.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        ra, dec = wcs2d.all_pix2world(xx, yy, 0)   # shape (2108, 2108), degrees
        wht[mask.get_values_pos(ra, dec, lonlat=True, valid_mask=True)]=0

    # Effective gain map: poisson_var = model_flux / eff_gain (in scaled units).
    eff_gain    = (exptime * Neff / factor).astype(np.float32)

    # ── Gaussian PSF stamp ────────────────────────────────────────────────────
    if psf_fwhm_arcsec is None:
        psf_fwhm_arcsec = 0.9265387 * 2.3548 * 0.11  # Roman H158 default (~0.240")
    sigma_pix  = (psf_fwhm_arcsec / 2.3548) / pix_scale
    stamp_size = max(31, int(np.ceil(10 * sigma_pix)) | 1)  # odd, ≥10-sigma wide
    c          = stamp_size // 2
    yy, xx     = np.mgrid[0:stamp_size, 0:stamp_size]
    psf        = Gaussian2D(1, c, c, sigma_pix, sigma_pix)(xx, yy).astype(np.float32)
    psf       /= psf.sum()
    if positionsize is not None:
        position, size = positionsize
        print(position, size)
        wcs = WCS(header, naxis=2)
        sci = Cutout2D(sci, position, size, wcs=wcs)
        header.update(sci.wcs.to_header())
        sci = sci.data
        wht = Cutout2D(wht, position, size, wcs=wcs).data
        eff_gain = Cutout2D(eff_gain, position, size, wcs=wcs).data
        noise_reals = np.stack(
            [Cutout2D(noise_reals[k], position, size, wcs=WCS(header, naxis=2)).data
             for k in range(noise_reals.shape[0])]
        )

    # ── Write outputs ─────────────────────────────────────────────────────────
    fits.writeto(sci_path,         sci,         header=header, overwrite=True)
    fits.writeto(wht_path,         wht,         header=header, overwrite=True)
    fits.writeto(psf_path,         psf,                        overwrite=True)
    fits.writeto(eff_gain_path,    eff_gain,    header=header, overwrite=True)
    fits.writeto(noise_reals_path, noise_reals, header=header, overwrite=True)

    logger = logging.getLogger('slimfarmer')
    logger.info(f'Science     → {sci_path}  shape={sci.shape}')
    logger.info(f'Weight      → {wht_path}  range=[{wht.min():.3g}, {wht.max():.3g}]  (background only)')
    logger.info(f'Eff gain    → {eff_gain_path}  range=[{eff_gain.min():.3g}, {eff_gain.max():.3g}]')
    logger.info(f'Noise reals → {noise_reals_path}  shape={noise_reals.shape}')
    logger.info(f'PSF         → {psf_path}  {stamp_size}×{stamp_size}px  '
                f'FWHM={psf_fwhm_arcsec:.3f}"  sum={psf.sum():.6f}')

    return sci_path, wht_path, psf_path, eff_gain_path, noise_reals_path


# Tile id is "I1_I2", where I1 corresponds to the +x (CRPIX1) axis and
# I2 corresponds to the +y (CRPIX2) axis. Empirically verified by reading
# adjacent CPR headers (see notes in prepare_stitched_block).
_NEIGHBOR_OFFSETS = [(d1, d2) for d1 in (-1, 0, 1) for d2 in (-1, 0, 1)
                    if (d1, d2) != (0, 0)]


def _neighbor_tile_id(tile, d1, d2, max_i1=39, max_i2=39):
    i1, i2 = (int(x) for x in tile.split('_'))
    n1, n2 = i1 + d1, i2 + d2
    if 0 <= n1 <= max_i1 and 0 <= n2 <= max_i2:
        return f'{n1:02d}_{n2:02d}'
    return None


def _stitched_offset_px(center_header, neighbor_header):
    """Pixel offset (dx, dy) of neighbor's (0,0) on the central block's grid.

    Assumes both headers share CRVAL/CDELT/CTYPE (a single global tangent
    projection) and only differ in CRPIX. Asserts integer offset.
    """
    for key in ('CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2'):
        cv, nv = center_header[key], neighbor_header[key]
        if isinstance(cv, str):
            if cv != nv:
                raise ValueError(f'WCS mismatch on {key}: {cv!r} vs {nv!r}')
        else:
            if not np.isclose(cv, nv, rtol=0, atol=1e-10):
                raise ValueError(f'WCS mismatch on {key}: {cv} vs {nv}')
    dx = neighbor_header['CRPIX1'] - center_header['CRPIX1']
    dy = neighbor_header['CRPIX2'] - center_header['CRPIX2']
    # Note: CRPIX_neighbor - CRPIX_center is the *negative* of the offset of
    # the neighbor's origin in the center's grid. CRPIX is the position of the
    # sky reference point in each image's pixel grid; if the neighbor's CRPIX1
    # is smaller, the neighbor's grid starts at a larger pixel coord in the
    # center's frame.
    off_x = -dx
    off_y = -dy
    if abs(off_x - round(off_x)) > 1e-6 or abs(off_y - round(off_y)) > 1e-6:
        raise ValueError(f'Non-integer pixel offset: ({off_x}, {off_y})')
    return int(round(off_x)), int(round(off_y))


def prepare_stitched_block(cpr_base, work_dir, tile, *,
                           buffer_arcsec, block_size_px, block_overlap_px,
                           bands=('Y1', 'J1', 'H1'),
                           truth=False, realization=False, mask=None,
                           overwrite_per_block=False,
                           psf_fwhm_arcsec=None):
    """Build per-band stitched images for a 3x3 IMCOM block neighborhood.

    For the central tile and each of the 8 neighbors (when present), this
    invokes :func:`prepare_images_from_cpr` into ``work_dir/<tile>/<band>/``
    and pastes a thin border of neighbor pixels onto an enlarged central
    canvas.

    Note on caching: each call decompresses 9 CPRs (center + 8 neighbors)
    into the *current* work_dir. When auto_submit.sh gives every tile its own
    work_dir, nothing is shared across Slurm tasks — each tile pays the full
    9x prep cost. This is intentional (Option A); see discussion in
    repo history. If you want cross-tile cache reuse, point all Slurm tasks
    at a shared `work_dir` or refactor to a separate cache dir. The result is written to
    ``work_dir/<central_tile>/stitched/<band>/`` and a ``stitched_meta.json``
    is dropped alongside it for downstream cropping/flagging.

    Returns
    -------
    pathall : dict
        ``{band: [sci, wht, psf, eff_gain, noise_reals]}`` mirroring what
        :func:`prepare_images_from_cpr` returns but pointing at the stitched
        files. PSF is reused unmodified from the central block.
    stitched_meta : dict
    """
    import json

    central_header = None
    pix_scale_arcsec = None
    pathall = {}
    neighbors_used = None

    for band in bands:
        # Resolve per-band PSF FWHM: scalar applies to all bands; dict looks up
        # by band (missing entries → default).
        if isinstance(psf_fwhm_arcsec, dict):
            band_psf_fwhm = psf_fwhm_arcsec.get(band)
        else:
            band_psf_fwhm = psf_fwhm_arcsec

        # ── 1. ensure central + neighbors are prepared on disk (cached) ──────
        per_tile_paths = {}
        tile_to_load = [tile] + [_neighbor_tile_id(tile, d1, d2)
                                 for d1, d2 in _NEIGHBOR_OFFSETS]
        tile_to_load = [t for t in tile_to_load if t is not None]
        for t in tile_to_load:
            cpr_path = os.path.join(cpr_base, f'{band}_coadds', f'im3x2-{band}_{t}.cpr.fits.gz')
            band_dir = os.path.join(work_dir, t, band)
            sci_p, wht_p, psf_p, eff_p, nr_p = prepare_images_from_cpr(
                cpr_path=cpr_path, work_dir=band_dir,
                overwrite=overwrite_per_block,
                truth=truth, realization=realization, mask=mask,
                psf_fwhm_arcsec=band_psf_fwhm,
            )
            per_tile_paths[t] = (sci_p, wht_p, psf_p, eff_p, nr_p)

        # ── 2. read central WCS, compute buffer in pixels ────────────────────
        with fits.open(per_tile_paths[tile][0]) as hd:
            center_header = hd[0].header.copy()
        if central_header is None:
            central_header = center_header
            pix_scale_arcsec = abs(center_header['CDELT2']) * 3600.0
            buf_px = int(np.ceil(buffer_arcsec / pix_scale_arcsec))
            canvas = block_size_px + 2 * buf_px
            neighbors_used = []

        # ── 3. allocate stitched arrays for this band ────────────────────────
        with fits.open(per_tile_paths[tile][0]) as hd:
            sci0 = hd[0].data
        with fits.open(per_tile_paths[tile][4]) as hd:
            nr0 = hd[0].data
        n_real = nr0.shape[0]

        sci_canvas = np.zeros((canvas, canvas), dtype=np.float32)
        wht_canvas = np.zeros((canvas, canvas), dtype=np.float32)
        eff_canvas = np.zeros((canvas, canvas), dtype=np.float32)
        nr_canvas  = np.zeros((n_real, canvas, canvas), dtype=np.float32)

        # ── 4. paste central block ───────────────────────────────────────────
        def _paste(canvas_arr, src, dx_canvas, dy_canvas):
            """Paste 2-D src onto canvas_arr at canvas-pixel origin (dx, dy).

            Crops to the canvas. Last-write-wins (matters only inside the
            34-px overlap regions, where the values agree to numerical
            precision since they came from the same IMCOM coadd anyway).
            """
            ny, nx = src.shape[-2:]
            x0 = max(0, dx_canvas)
            y0 = max(0, dy_canvas)
            x1 = min(canvas, dx_canvas + nx)
            y1 = min(canvas, dy_canvas + ny)
            if x0 >= x1 or y0 >= y1:
                return
            sx0 = x0 - dx_canvas
            sy0 = y0 - dy_canvas
            sx1 = sx0 + (x1 - x0)
            sy1 = sy0 + (y1 - y0)
            if src.ndim == 2:
                canvas_arr[y0:y1, x0:x1] = src[sy0:sy1, sx0:sx1]
            else:
                canvas_arr[:, y0:y1, x0:x1] = src[:, sy0:sy1, sx0:sx1]

        # ── 5. paste each neighbor FIRST, central LAST ───────────────────────
        # Order matters: the central block's pixels must win in the central
        # 2108x2108 region. We accomplish this by pasting all neighbors first
        # (filling the buffer ring + redundantly the inner overlap region)
        # and then pasting the central block on top, so the inner region is
        # exactly the central tile's own data with no contamination from
        # neighbor IMCOM solutions.
        for d1, d2 in _NEIGHBOR_OFFSETS:
            n_tile = _neighbor_tile_id(tile, d1, d2)
            if n_tile is None or n_tile not in per_tile_paths:
                continue
            with fits.open(per_tile_paths[n_tile][0]) as hd:
                n_sci = hd[0].data
                n_header = hd[0].header
            off_x, off_y = _stitched_offset_px(center_header, n_header)
            # Each block has `block_overlap_px` pixels shared on each of its
            # four sides, so the pixel-offset of an adjacent block's origin
            # is block_size_px - 2 * block_overlap_px.
            # Tile-id index 1 → +x axis (CRPIX1); index 2 → +y axis (CRPIX2).
            expected = block_size_px - 2 * block_overlap_px
            if off_x != d1 * expected or off_y != d2 * expected:
                raise ValueError(
                    f'Block offset for {n_tile} relative to {tile}: got '
                    f'({off_x}, {off_y}), expected ({d1 * expected}, {d2 * expected}). '
                    f'Check block_overlap_px (={block_overlap_px}) and block_size_px '
                    f'(={block_size_px}).'
                )
            dx_canvas = buf_px + off_x
            dy_canvas = buf_px + off_y
            _paste(sci_canvas, n_sci, dx_canvas, dy_canvas)
            with fits.open(per_tile_paths[n_tile][1]) as hd:
                _paste(wht_canvas, hd[0].data, dx_canvas, dy_canvas)
            with fits.open(per_tile_paths[n_tile][3]) as hd:
                _paste(eff_canvas, hd[0].data, dx_canvas, dy_canvas)
            with fits.open(per_tile_paths[n_tile][4]) as hd:
                _paste(nr_canvas, hd[0].data, dx_canvas, dy_canvas)
            if band == bands[0]:
                neighbors_used.append(n_tile)

        # Central block last — overrides any neighbor pixels in the inner
        # 2108x2108 so the central region exactly matches the central tile's
        # own data.
        _paste(sci_canvas, sci0, buf_px, buf_px)
        with fits.open(per_tile_paths[tile][1]) as hd:
            _paste(wht_canvas, hd[0].data, buf_px, buf_px)
        with fits.open(per_tile_paths[tile][3]) as hd:
            _paste(eff_canvas, hd[0].data, buf_px, buf_px)
        _paste(nr_canvas, nr0, buf_px, buf_px)

        # ── 6. build stitched WCS by shifting CRPIX ──────────────────────────
        new_header = center_header.copy()
        new_header['CRPIX1'] = float(center_header['CRPIX1']) + buf_px
        new_header['CRPIX2'] = float(center_header['CRPIX2']) + buf_px
        new_header['NAXIS1'] = canvas
        new_header['NAXIS2'] = canvas

        # ── 7. write stitched files ──────────────────────────────────────────
        out_dir = os.path.join(work_dir, tile, 'stitched', band)
        os.makedirs(out_dir, exist_ok=True)
        sci_path = os.path.join(out_dir, 'roman_image.fits')
        wht_path = os.path.join(out_dir, 'roman_weight.fits')
        eff_path = os.path.join(out_dir, 'roman_eff_gain.fits')
        nr_path  = os.path.join(out_dir, 'roman_noise_reals.fits')
        psf_path = per_tile_paths[tile][2]  # reuse the central block's PSF stamp

        fits.writeto(sci_path, sci_canvas, header=new_header, overwrite=True)
        fits.writeto(wht_path, wht_canvas, header=new_header, overwrite=True)
        fits.writeto(eff_path, eff_canvas, header=new_header, overwrite=True)
        fits.writeto(nr_path,  nr_canvas,  header=new_header, overwrite=True)

        pathall[band] = [sci_path, wht_path, psf_path, eff_path, nr_path]

    stitched_meta = {
        'central_tile': tile,
        'buf_px': int(buf_px),
        'pix_scale_arcsec': float(pix_scale_arcsec),
        'block_size_px': int(block_size_px),
        'block_overlap_px': int(block_overlap_px),
        'canvas_px': int(canvas),
        'neighbors_used': sorted(neighbors_used or []),
        'buffer_arcsec': float(buffer_arcsec),
    }
    meta_path = os.path.join(work_dir, tile, 'stitched_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(stitched_meta, f, indent=2)

    return pathall, stitched_meta


def get_detection_kernel(filter_kernel):
    """Load or generate a convolution filter for SEP detection."""
    if isinstance(filter_kernel, str):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'conv_filters', filter_kernel)
        if os.path.exists(filename):
            return np.array(np.array(ascii.read(filename, data_start=1)).tolist())
        raise FileNotFoundError(f'Convolution filter not found: {filename}')
    elif np.isscalar(filter_kernel):
        from astropy.convolution import Gaussian2DKernel
        return np.array(Gaussian2DKernel(x_stddev=filter_kernel / 2.35, factor=1))
    raise ValueError(f'Unknown filter_kernel type: {type(filter_kernel)}')



import galsim, galsim.roman as groman
def meanall_new(imageall, wall, effiall, bandall=['Y106', 'J129', 'H158']):
    oneoverzpall= []
    for band in bandall:
        oneoverzpall.append(1.0/(10**(galsim.roman.getBandpasses()[band].zeropoint/2.5)))
    oneoverzpall = np.array(oneoverzpall)/np.sum(oneoverzpall)
    scalearray  = oneoverzpall 
    
    #scalearray = [0.4849, 0.4777,0.4628]
    n = np.sum(scalearray)
    sumall = 0
    var_num = 0
    effi_num=0
    for img, w, eff, s in zip(imageall, wall, effiall, scalearray):
        sumall += img*s
        var_num += 1/w*(s**2)
        effi_num += img / eff*(s**2)
    sci = sumall/n
    wall = n**2/var_num
    effi_equivalent = n * sci / effi_num 

    return sci, wall, effi_equivalent   


def match_spatial(cat_ra, cat_dec, truth_ra, truth_dec):
    """Pure spatial matching (S): closest truth object per detected source.

    Parameters
    ----------
    cat_ra, cat_dec : array — detected source coordinates (degrees).
    truth_ra, truth_dec : array — truth catalogue coordinates (degrees).

    Returns
    -------
    idx : int array, shape (N_cat,) — index into truth catalogue for each
        detected source (nearest neighbour).
    sep : float array, shape (N_cat,) — on-sky separation in arcsec.
    """
    from astropy.coordinates import SkyCoord
    cat_coord = SkyCoord(cat_ra, cat_dec, unit='deg')
    truth_coord = SkyCoord(truth_ra, truth_dec, unit='deg')
    idx, d2d, _ = cat_coord.match_to_catalog_sky(truth_coord)
    return idx, d2d.to(u.arcsec).value


def match_spatial_mag(cat_ra, cat_dec, cat_mags,
                      truth_ra, truth_dec, truth_mags,
                      radius_arcsec=0.6, mag_thresh=1.0):
    """Spatial + magnitude matching (S+M), supporting multiple bands.

    For each detected source, find truth objects within *radius_arcsec*.
    Among those, select the one with the smallest multi-band magnitude
    distance sqrt(dmag1² + dmag2² + ...), provided that distance is less
    than *mag_thresh*.  Sources with no qualifying neighbour are marked
    unmatched (index = -1).

    Parameters
    ----------
    cat_ra, cat_dec : arrays — detected catalogue coordinates (degrees).
    cat_mags : array, shape (N_cat,) or (N_cat, N_bands) — detected magnitudes.
    truth_ra, truth_dec : arrays — truth catalogue coordinates (degrees).
    truth_mags : array, shape (N_truth,) or (N_truth, N_bands) — truth magnitudes.
    radius_arcsec : float — spatial search radius (default 0.6", ~3 Roman px).
    mag_thresh : float — maximum allowed magnitude distance (default 1.0).

    Returns
    -------
    idx : int array, shape (N_cat,) — index into truth catalogue, or -1 if
        unmatched.
    sep : float array, shape (N_cat,) — separation in arcsec (-1 if unmatched).
    dmag : float array, shape (N_cat,) — magnitude distance (NaN if unmatched).
    """
    from astropy.coordinates import SkyCoord

    cat_coord = SkyCoord(cat_ra, cat_dec, unit='deg')
    truth_coord = SkyCoord(truth_ra, truth_dec, unit='deg')

    cat_mags = np.atleast_2d(np.asarray(cat_mags, dtype=np.float64))
    truth_mags = np.atleast_2d(np.asarray(truth_mags, dtype=np.float64))
    if cat_mags.shape[0] != len(cat_coord):
        cat_mags = cat_mags.T
    if truth_mags.shape[0] != len(truth_coord):
        truth_mags = truth_mags.T

    idx_out = np.full(len(cat_coord), -1, dtype=int)
    sep_out = np.full(len(cat_coord), -1.0)
    dmag_out = np.full(len(cat_coord), np.nan)

    idxc, idxt, d2d, _ = truth_coord.search_around_sky(
        cat_coord, radius_arcsec * u.arcsec)

    sep_vals = d2d.to(u.arcsec).value

    for i in range(len(cat_coord)):
        mask = idxc == i
        if not np.any(mask):
            continue
        t_idx = idxt[mask]
        t_sep = sep_vals[mask]
        diff = cat_mags[i] - truth_mags[t_idx]
        t_dist = np.sqrt(np.sum(diff ** 2, axis=1))
        valid = t_dist < mag_thresh
        if not np.any(valid):
            continue
        best = np.argmin(t_dist[valid])
        idx_out[i] = t_idx[valid][best]
        sep_out[i] = t_sep[valid][best]
        dmag_out[i] = t_dist[valid][best]

    return idx_out, sep_out, dmag_out
