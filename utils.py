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
                model[idx].addGaussianPrior('ra', mu=model[idx][0], sigma=sigma)
                model[idx].addGaussianPrior('dec', mu=model[idx][1], sigma=sigma)
        elif name == 'fracDev':
            if priors.get('fracDev', 'none') in ('fix', 'freeze'):
                model.freezeParam(idx)
        elif name in ('shape', 'shapeDev', 'shapeExp'):
            if priors.get('shape', 'none') in ('fix', 'freeze'):
                for i, _ in enumerate(model[idx].getParamNames()):
                    if i != 0:
                        model[idx].freezeParam(i)
            if priors.get('reff', 'none') in ('fix', 'freeze'):
                model[idx].freezeParam(0)
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
        flux = model.getBrightness().getFlux(band)
        flux_var = model.variance.getBrightness().getFlux(band)
        flux_err = np.sqrt(abs(flux_var)) if flux_var > 0 else 0.0
    except Exception:
        flux, flux_err = 0.0, 0.0

    source[f'{band}_flux'] = flux
    source[f'{band}_flux_err'] = flux_err
    source[f'{band}_flux_ujy'] = flux * 10 ** (-0.4 * (zeropoint - 23.9))
    if flux > 0:
        source[f'{band}_mag'] = -2.5 * np.log10(flux) + zeropoint
    else:
        source[f'{band}_mag'] = np.nan

    return source


def prepare_images_from_cpr(cpr_path, work_dir,
                            psf_fwhm_arcsec=None,
                            sci_name='roman_image.fits',
                            wht_name='roman_weight.fits',
                            psf_name='PSF_F158.fits'):
    """
    Extract science image, weight map, and PSF stamp from an IMCOM CPR FITS file.

    The CPR format stores the IMCOM output as a compressed multi-layer cube:
      HDU 0, layer  0 : science image (background-subtracted, DN/px/s)
      HDU 0, layer  1 : Poisson signal map
      HDU 0, layer 21 : noise realisation (used to calibrate the variance scale)
      HDU 6           : Sigma map (correlated noise amplitude, in log10-bels)
      HDU 8           : Neff map  (effective coverage, in log10-bels)

    Variance = correlated_term + Poisson_term:
      correlated_term = (sum(noise_layer²) / sum(Sigma)) * Sigma
      Poisson_term    = max(signal_map, 0) / 107.52398 / Neff

    Weight map = 1 / variance (inverse variance, as expected by Tractor).

    PSF is approximated as a 2-D Gaussian with the Roman H158 effective FWHM.
    The stamp is normalised to sum = 1.

    Parameters
    ----------
    cpr_path        : str   — path to *.cpr.fits.gz
    work_dir        : str   — output directory (created if absent)
    psf_fwhm_arcsec : float — Gaussian PSF FWHM in arcsec.
                              Default: Roman H158 effective PSF FWHM (~0.240").
    sci_name        : str   — output filename for science image
    wht_name        : str   — output filename for weight map
    psf_name        : str   — output filename for PSF stamp

    Returns
    -------
    sci_path, wht_path, psf_path : str
    """
    try:
        from pyimcom.compress.compressutils import ReadFile
        from pyimcom.diagnostics.outimage_utils.helper import HDU_to_bels
    except ImportError as e:
        raise ImportError(
            'pyimcom is required to read CPR files. '
            'Install it from https://github.com/hirata10/furry-parakeet'
        ) from e

    from astropy.modeling.models import Gaussian2D

    os.makedirs(work_dir, exist_ok=True)

    # ── Read CPR ──────────────────────────────────────────────────────────────
    cpr    = ReadFile(cpr_path)
    sci    = cpr[0].data[0][0].astype(np.float32)
    header = cpr[0].header

    pix_scale = abs(header['CDELT2']) * 3600.  # arcsec/px

    # ── Variance map ──────────────────────────────────────────────────────────
    Sigma       = 10 ** (HDU_to_bels(cpr[6]) * cpr[6].data[0])
    Neff        = 10 ** (HDU_to_bels(cpr[8]) * cpr[8].data[0])
    scalefactor = np.sum(cpr[0].data[0][21] ** 2)
    varmap      = ((scalefactor / np.sum(Sigma)) * Sigma
                   + np.maximum(cpr[0].data[0][1].astype(np.float32), 0)
                   / 107.52398 / Neff)
    varmap      = varmap.astype(np.float32)
    wht         = np.where(varmap > 0, 1.0 / varmap, 0.).astype(np.float32)

    # ── Gaussian PSF stamp ────────────────────────────────────────────────────
    if psf_fwhm_arcsec is None:
        psf_fwhm_arcsec = 0.9265387 * 2.3548 * 0.11  # Roman H158 default (~0.240")
    sigma_pix  = (psf_fwhm_arcsec / 2.3548) / pix_scale
    stamp_size = max(31, int(np.ceil(10 * sigma_pix)) | 1)  # odd, ≥10-sigma wide
    c          = stamp_size // 2
    yy, xx     = np.mgrid[0:stamp_size, 0:stamp_size]
    psf        = Gaussian2D(1, c, c, sigma_pix, sigma_pix)(xx, yy).astype(np.float32)
    psf       /= psf.sum()

    # ── Write outputs ─────────────────────────────────────────────────────────
    sci_path = os.path.join(work_dir, sci_name)
    wht_path = os.path.join(work_dir, wht_name)
    psf_path = os.path.join(work_dir, psf_name)

    fits.writeto(sci_path, sci, header=header, overwrite=True)
    fits.writeto(wht_path, wht, header=header, overwrite=True)
    fits.writeto(psf_path, psf,                overwrite=True)

    logger = logging.getLogger('slimfarmer')
    logger.info(f'Science  → {sci_path}  shape={sci.shape}')
    logger.info(f'Weight   → {wht_path}  range=[{wht.min():.3g}, {wht.max():.3g}]')
    logger.info(f'PSF      → {psf_path}  {stamp_size}×{stamp_size}px  '
                f'FWHM={psf_fwhm_arcsec:.3f}"  sum={psf.sum():.6f}')

    return sci_path, wht_path, psf_path


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
