"""Forced photometry on arbitrary images using slimfarmer catalog models.

Given a slimfarmer catalog (from Roman), reconstruct the Tractor source
models (position + morphology frozen) and fit only the flux on a new image
with its own PSF, weight map, and WCS.

Usage
-----
>>> from slimfarmer.forced import forced_photometry
>>> result = forced_photometry(
...     catalog_path='catalog.fits',
...     sci_path='rubin_r.fits',
...     wht_path='rubin_r_wht.fits',
...     psf_path='rubin_r_psf.fits',
...     band='r',
...     zeropoint=27.0,
... )
"""
import logging
import os
import numpy as np
from collections import OrderedDict

from astropy.io import fits
from astropy.table import Table, Column
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u

from tractor import (PointSource, Fluxes, RaDecPos,
                     Image, ConstantFitsWcs, Tractor)
from tractor.galaxy import ExpGalaxy, DevGalaxy, FixedCompositeGalaxy, SoftenedFracDev
from tractor.ellipses import EllipseESoft
from tractor.basics import FluxesPhotoCal, ConstantSky
from astrometry.util.util import Tan

from .utils import SimpleGalaxy
from .image import HybridPixelizedPSF
from tractor import PixelizedPSF

logger = logging.getLogger('slimfarmer.forced')

RUBIN_BANDS = ('u', 'g', 'r', 'i', 'z', 'y')


def rubin_coadd_wcs(header):
    """Build a corrected WCS from a Rubin coadd FITS header.

    LSST/Rubin coadd patches store WCS in parent (tract) pixel coords.
    The LTV1/LTV2 keywords give the local→parent offset.  We shift CRPIX
    by LTV so that the WCS maps local (patch) pixel coords to sky.
    """
    h = header.copy()
    h['CRPIX1'] = float(header['CRPIX1']) + float(header.get('LTV1', 0))
    h['CRPIX2'] = float(header['CRPIX2']) + float(header.get('LTV2', 0))
    return WCS(h)


def load_skymap_info(path):
    """Load the skymap_info.json with per-patch inner_bbox and tract corners."""
    import json as _json
    with open(path) as f:
        return _json.load(f)


def patch_inner_bbox_local(skymap_info, tract, patch, header):
    """Return the inner bbox in local (patch) pixel coords as (x0, y0, x1, y1).

    inner_bbox in the JSON is in parent (tract) pixel coords.
    We convert to local using the LTV keywords from the FITS header.
    """
    ib = skymap_info['tracts'][str(tract)]['patches'][str(patch)]['inner_bbox']
    ltv1 = float(header.get('LTV1', 0))
    ltv2 = float(header.get('LTV2', 0))
    return (ib[0] + ltv1, ib[1] + ltv2, ib[2] + ltv1, ib[3] + ltv2)


def nearest_tract(skymap_info, ra, dec):
    """Return the tract ID nearest to each (ra, dec) position.

    Uses angular distance to tract centers as the ownership criterion
    (Voronoi tessellation by tract center). This naturally handles
    the 2D overlap geometry where all tract pairs overlap in both RA
    and Dec.

    Parameters
    ----------
    skymap_info : dict
    ra, dec : float or array

    Returns
    -------
    tract_ids : int or array of int
    """
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    tracts = skymap_info['tracts']
    tract_ids = list(tracts.keys())
    centers_ra = np.array([tracts[t]['center_ra'] for t in tract_ids])
    centers_dec = np.array([tracts[t]['center_dec'] for t in tract_ids])

    cos_dec = np.cos(np.radians(dec))
    best = np.full(len(ra), -1, dtype=int)
    best_dist = np.full(len(ra), np.inf)
    for i, tid in enumerate(tract_ids):
        dra = (ra - centers_ra[i]) * cos_dec
        ddec = dec - centers_dec[i]
        dist = dra ** 2 + ddec ** 2
        closer = dist < best_dist
        best[closer] = int(tid)
        best_dist[closer] = dist[closer]
    return best if len(best) > 1 else int(best[0])


def source_in_exclusive_region(ra, dec, tract, patch, wcs, header,
                               skymap_info):
    """Check if a source at (ra, dec) falls within the exclusive region of
    this patch.

    Two conditions must both hold:
    1. The source's pixel position falls within the patch's ``inner_bbox``
       (handles intra-tract overlap — 200px margin on each side).
    2. The source's nearest tract center is this tract (handles inter-tract
       overlap via Voronoi ownership).

    Returns a boolean array.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)

    # 1. Check patch inner bbox (handles intra-tract overlap)
    x, y = wcs.all_world2pix(ra, dec, 0)
    ib = patch_inner_bbox_local(skymap_info, tract, patch, header)
    in_patch_inner = ((x >= ib[0]) & (x <= ib[2]) &
                      (y >= ib[1]) & (y <= ib[3]))

    # 2. Check tract ownership (handles inter-tract overlap)
    owned = nearest_tract(skymap_info, ra, dec) == int(tract)

    return in_patch_inner & owned


def find_rubin_coadds(ra_min, ra_max, dec_min, dec_max,
                      data_base, bands=None):
    """Find all Rubin coadds overlapping an RA/Dec bounding box.

    Parameters
    ----------
    ra_min, ra_max, dec_min, dec_max : float
        Sky bounding box in degrees.
    data_base : str
        Root of the Rubin data tree (e.g. ``/path/to/RomanRubin``).
        Expected layout: ``{data_base}/{band}/coadd_{tract}_{patch}.fits``.
    bands : sequence of str, optional
        Bands to search (default: u, g, r, i, z, y).

    Returns
    -------
    list of dict
        Each dict has keys: band, tract, patch, sci_path, psf_path,
        ra_min, ra_max, dec_min, dec_max, wcs.
    """
    import glob as _glob

    if bands is None:
        bands = RUBIN_BANDS

    results = []
    for band in bands:
        band_dir = os.path.join(data_base, band)
        if not os.path.isdir(band_dir):
            continue
        for f in sorted(_glob.glob(os.path.join(band_dir, 'coadd_*.fits'))):
            name = os.path.basename(f)
            parts = name.replace('coadd_', '').replace('.fits', '').split('_')
            tract, patch = int(parts[0]), int(parts[1])

            h = fits.getheader(f, 1)
            wcs = rubin_coadd_wcs(h)
            ny, nx = int(h['NAXIS2']), int(h['NAXIS1'])
            corners = wcs.all_pix2world(
                [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)], 0)
            crn_ra_min = float(corners[:, 0].min())
            crn_ra_max = float(corners[:, 0].max())
            crn_dec_min = float(corners[:, 1].min())
            crn_dec_max = float(corners[:, 1].max())

            if (crn_ra_max > ra_min and crn_ra_min < ra_max and
                    crn_dec_max > dec_min and crn_dec_min < dec_max):
                psf_path = os.path.join(band_dir,
                                        f'psf_{tract}_{patch}.fits')
                results.append({
                    'band': band,
                    'tract': tract,
                    'patch': patch,
                    'sci_path': f,
                    'psf_path': psf_path if os.path.exists(psf_path) else None,
                    'ra_min': crn_ra_min,
                    'ra_max': crn_ra_max,
                    'dec_min': crn_dec_min,
                    'dec_max': crn_dec_max,
                    'wcs': wcs,
                })
    return results


def roman_tile_footprint(tile, cpr_base, band='H1', buffer_arcsec=6.0):
    """Return (ra_min, ra_max, dec_min, dec_max) for a Roman tile,
    optionally expanded by buffer_arcsec on each side."""
    import os as _os
    cpr_path = _os.path.join(cpr_base, f'{band}_coadds',
                             f'im3x2-{band}_{tile}.cpr.fits.gz')
    h = fits.open(cpr_path)[0].header
    wcs = WCS(h).celestial
    pix_scale = abs(h['CDELT2']) * 3600.0
    block = 2108
    buf_px = int(np.ceil(buffer_arcsec / pix_scale))
    yy, xx = np.array([[-buf_px, -buf_px, block + buf_px, block + buf_px],
                        [-buf_px, block + buf_px, -buf_px, block + buf_px]])
    ra, dec = wcs.all_pix2world(xx, yy, 0)
    return float(ra.min()), float(ra.max()), float(dec.min()), float(dec.max())


def find_rubin_coadds_for_tile(tile, cpr_base, data_base,
                               bands=None, buffer_arcsec=6.0):
    """Find all Rubin coadds overlapping a Roman tile (including stitch region).

    Parameters
    ----------
    tile : str
        Roman tile ID, e.g. '20_20'.
    cpr_base : str
        Root of the Roman CPR files.
    data_base : str
        Root of the Rubin data tree.
    bands : sequence of str, optional
        Rubin bands to search.
    buffer_arcsec : float
        Extra border around the tile (matches the stitching buffer).

    Returns
    -------
    list of dict
        Same format as :func:`find_rubin_coadds`.
    """
    ra_min, ra_max, dec_min, dec_max = roman_tile_footprint(
        tile, cpr_base, buffer_arcsec=buffer_arcsec)
    logger.info(f'Roman tile {tile} footprint (buf={buffer_arcsec}"): '
                f'RA=[{ra_min:.4f},{ra_max:.4f}] Dec=[{dec_min:.4f},{dec_max:.4f}]')
    return find_rubin_coadds(ra_min, ra_max, dec_min, dec_max,
                             data_base=data_base, bands=bands)


def _read_wcs_for_cutout(astropy_wcs, cutout_shape):
    """Convert an astropy WCS to a Tractor ConstantFitsWcs."""
    h = astropy_wcs.to_header()
    ny, nx = cutout_shape
    if 'CD1_1' in h:
        cd11 = float(h['CD1_1'])
        cd12 = float(h.get('CD1_2', 0))
        cd21 = float(h.get('CD2_1', 0))
        cd22 = float(h['CD2_2'])
    else:
        cdelt1 = float(h.get('CDELT1', -1e-5))
        cdelt2 = float(h.get('CDELT2', 1e-5))
        pc11 = float(h.get('PC1_1', 1.0))
        pc12 = float(h.get('PC1_2', 0.0))
        pc21 = float(h.get('PC2_1', 0.0))
        pc22 = float(h.get('PC2_2', 1.0))
        cd11 = cdelt1 * pc11
        cd12 = cdelt1 * pc12
        cd21 = cdelt2 * pc21
        cd22 = cdelt2 * pc22
    tan = Tan(
        float(h.get('CRVAL1', 0)), float(h.get('CRVAL2', 0)),
        float(h.get('CRPIX1', 0)), float(h.get('CRPIX2', 0)),
        cd11, cd12, cd21, cd22,
        nx, ny,
    )
    return ConstantFitsWcs(tan)


def reconstruct_source(row, band, position_sigma_arcsec=None):
    """Rebuild a frozen-morphology Tractor source from a catalog row.

    Shape parameters are always frozen. Flux is always free.
    If ``position_sigma_arcsec`` is set, position is also free with a tight
    Gaussian prior centered on the catalog position (sigma in arcsec).
    """
    pos = RaDecPos(float(row['ra']), float(row['dec']))
    flux = Fluxes(**{band: 0.0})

    name = row['name']
    if isinstance(name, bytes):
        name = name.decode()

    if name == 'PointSource':
        src = PointSource(pos, flux)

    elif name == 'SimpleGalaxy':
        shape = EllipseESoft(np.log(0.1), 0., 0.)
        src = ExpGalaxy(pos, flux, shape)

    elif name == 'ExpGalaxy':
        shape = EllipseESoft(float(row['logre']),
                             float(row['ee1']),
                             float(row['ee2']))
        src = ExpGalaxy(pos, flux, shape)

    elif name == 'DevGalaxy':
        shape = EllipseESoft(float(row['logre']),
                             float(row['ee1']),
                             float(row['ee2']))
        src = DevGalaxy(pos, flux, shape)

    elif name == 'FixedCompositeGalaxy':
        fracdev = SoftenedFracDev(float(row['softfracdev']))
        shape_exp = EllipseESoft(float(row['logre_exp']),
                                  float(row['ee1_exp']),
                                  float(row['ee2_exp']))
        shape_dev = EllipseESoft(float(row['logre_dev']),
                                  float(row['ee1_dev']),
                                  float(row['ee2_dev']))
        src = FixedCompositeGalaxy(pos, flux, fracdev, shape_exp, shape_dev)
    else:
        src = PointSource(pos, flux)

    if position_sigma_arcsec is not None and position_sigma_arcsec > 0:
        src.freezeAllBut('brightness', 'pos')
        ra_val = float(row['ra'])
        dec_val = float(row['dec'])
        sigma_deg = float(position_sigma_arcsec) / 3600.0
        sigma_ra = sigma_deg / max(np.cos(np.radians(dec_val)), 1e-6)
        sigma_dec = sigma_deg
        try:
            src.addGaussianPrior('pos.ra', ra_val, sigma_ra)
            src.addGaussianPrior('pos.dec', dec_val, sigma_dec)
        except Exception:
            try:
                src.addGaussianPrior('ra', ra_val, sigma_ra)
                src.addGaussianPrior('dec', dec_val, sigma_dec)
            except Exception:
                src.pos.addGaussianPrior('ra', ra_val, sigma_ra)
                src.pos.addGaussianPrior('dec', dec_val, sigma_dec)
    else:
        src.freezeAllBut('brightness')
    return src


def _make_tractor_image(sci, wht, psf_model, wcs, band):
    """Build a Tractor Image from numpy arrays + PSF."""
    tractor_wcs = _read_wcs_for_cutout(wcs, sci.shape)
    return Image(
        data=sci.astype(np.float64),
        invvar=wht.astype(np.float64),
        psf=psf_model,
        wcs=tractor_wcs,
        photocal=FluxesPhotoCal(band),
        sky=ConstantSky(0),
    )


def load_psf(psf_input):
    """Load a PSF model from a FITS path, numpy array, or existing model.

    Returns a Tractor-compatible PSF object.
    """
    if isinstance(psf_input, str):
        data = fits.getdata(psf_input).astype(np.float32)
        data[~np.isfinite(data) | (data < 1e-31)] = 1e-31
        data /= data.sum()
        return HybridPixelizedPSF(PixelizedPSF(data))
    elif isinstance(psf_input, np.ndarray):
        data = psf_input.astype(np.float32).copy()
        data[~np.isfinite(data) | (data < 1e-31)] = 1e-31
        data /= data.sum()
        return HybridPixelizedPSF(PixelizedPSF(data))
    else:
        return psf_input


def forced_photometry(catalog_path, sci_path, wht_path, psf_input,
                      band, zeropoint=None,
                      cutout_size=200, max_steps=50, dlnp_crit=1e-6,
                      damping=1e-3, use_seg_groups=False,
                      position_sigma_arcsec=None):
    """Run forced photometry on a new image using slimfarmer catalog models.

    Parameters
    ----------
    catalog_path : str or Table
        Path to slimfarmer catalog.fits, or an astropy Table.
    sci_path : str
        Path to the new science image FITS.
    wht_path : str
        Path to the new weight (inverse-variance) map FITS.
    psf_input : str, ndarray, or Tractor PSF object
        Path to PSF stamp FITS, a numpy array, or a pre-built PSF model.
    band : str
        Band name for the new image (e.g. 'r', 'i', 'g').
    zeropoint : float, optional
        AB zeropoint of the new image. Stored in output but not used
        in the flux fitting (Tractor fits in image units).
    cutout_size : int
        Side length (pixels) of per-group cutouts. Should be large enough
        to contain the largest source + PSF wings.
    max_steps : int
        Maximum Tractor optimisation steps per group.
    dlnp_crit : float
        Convergence threshold for delta-log-probability.
    damping : float
        Levenberg-Marquardt damping parameter.
    use_seg_groups : bool
        If True, use the ``group_id`` column from the catalog (seg-map groups
        from the original Roman photometry run) instead of grid-based tiling.
    position_sigma_arcsec : float, optional
        If set, thaw source positions and apply a tight Gaussian prior with
        this sigma (in arcsec) centered on the catalog RA/Dec. Useful when
        there is astrometric slop between the Roman-derived catalog and the
        new image.

    Returns
    -------
    result : astropy.table.Table
        Catalog with columns: id, ra, dec, name, {band}_flux,
        {band}_flux_err, {band}_chi2, {band}_dof.
    """
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(name)s :: %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S')

    if isinstance(catalog_path, Table):
        cat = catalog_path
    else:
        cat = Table.read(catalog_path)

    sci_full = fits.getdata(sci_path).astype(np.float64)
    wht_full = fits.getdata(wht_path).astype(np.float64)
    wcs_full = WCS(fits.getheader(sci_path))
    psf_model = load_psf(psf_input)

    bad = ~np.isfinite(sci_full)
    sci_full[bad] = 0.
    wht_full[bad | ~np.isfinite(wht_full) | (wht_full < 0)] = 0.

    ny, nx = sci_full.shape

    valid = np.array([(n != b'Bad' and n != b'' and n != 'Bad' and n != '')
                       for n in cat['name']])
    cat_valid = cat[valid]
    logger.info(f'Loaded catalog: {len(cat_valid)} valid sources '
                f'({len(cat)} total, {(~valid).sum()} bad/empty)')

    # Convert source RA/Dec to pixel coords on the new image
    src_x, src_y = wcs_full.all_world2pix(
        np.asarray(cat_valid['ra'], dtype=float),
        np.asarray(cat_valid['dec'], dtype=float), 0)

    # Filter to sources that fall within the image
    in_image = ((src_x >= -cutout_size//2) & (src_x < nx + cutout_size//2) &
                (src_y >= -cutout_size//2) & (src_y < ny + cutout_size//2))
    cat_in = cat_valid[in_image]
    src_x = src_x[in_image]
    src_y = src_y[in_image]
    logger.info(f'{len(cat_in)} sources fall within the new image')

    if len(cat_in) == 0:
        logger.warning('No sources overlap the new image.')
        return Table()

    # Group sources for per-cutout fitting.
    group_map = {}
    if use_seg_groups and 'group_id' in cat_in.colnames:
        for i in range(len(cat_in)):
            gid = int(cat_in['group_id'][i])
            group_map.setdefault(gid, []).append(i)
        logger.info(f'Using {len(group_map)} seg-map groups from catalog')
    else:
        cell = cutout_size
        cell_x = (src_x / cell).astype(int)
        cell_y = (src_y / cell).astype(int)
        for i in range(len(cat_in)):
            key = (int(cell_x[i]), int(cell_y[i]))
            group_map.setdefault(key, []).append(i)
        logger.info(f'Formed {len(group_map)} spatial groups for fitting '
                    f'(cell size={cell} px)')

    # Output arrays
    out_flux = np.zeros(len(cat_in))
    out_flux_err = np.zeros(len(cat_in))
    out_chi2 = np.zeros(len(cat_in))
    out_dof = np.zeros(len(cat_in), dtype=int)

    from tqdm import tqdm as _tqdm
    for gid, members in _tqdm(group_map.items(), desc='Forced phot groups'):
        mx = src_x[members]
        my = src_y[members]
        cx = (mx.min() + mx.max()) / 2
        cy = (my.min() + my.max()) / 2
        span_x = max(mx.max() - mx.min() + cutout_size, cutout_size)
        span_y = max(my.max() - my.min() + cutout_size, cutout_size)
        size = (int(np.ceil(span_y)), int(np.ceil(span_x)))

        try:
            cut_sci = Cutout2D(sci_full, (cx, cy), size, wcs=wcs_full,
                               mode='partial', fill_value=0.)
            cut_wht = Cutout2D(wht_full, (cx, cy), size, wcs=wcs_full,
                               mode='partial', fill_value=0.)
        except Exception:
            for idx in members:
                out_flux[idx] = np.nan
                out_flux_err[idx] = np.nan
            continue

        tim = _make_tractor_image(cut_sci.data, cut_wht.data,
                                  psf_model, cut_sci.wcs, band)

        sources = []
        for idx in members:
            src = reconstruct_source(cat_in[idx], band,
                                     position_sigma_arcsec=position_sigma_arcsec)
            sources.append(src)

        tr = Tractor([tim], sources)
        tr.freezeParam('images')

        use_priors = position_sigma_arcsec is not None and position_sigma_arcsec > 0
        var = None
        for step in range(max_steps):
            try:
                dlnp, X, alpha, var = tr.optimize(
                    variance=True, damping=damping, priors=use_priors)
            except Exception:
                break
            if dlnp < dlnp_crit:
                break

        mod = tr.getModelImage(0)
        resid = cut_sci.data.astype(np.float64) - mod
        invvar = cut_wht.data.astype(np.float64)

        # Build global index of flux param per source (robust to frozen/free layout)
        var_offset = 0
        for i, idx in enumerate(members):
            src = sources[i]
            f = src.getBrightness().getFlux(band)
            out_flux[idx] = f
            n_free = src.numberOfParams()
            flux_local_idx = None
            try:
                pnames = src.getParamNames()
                for j, pn in enumerate(pnames):
                    if 'brightness' in pn.lower():
                        flux_local_idx = j
                        break
            except Exception:
                pass
            if flux_local_idx is None:
                flux_local_idx = n_free - 1  # brightness usually last
            gi = var_offset + flux_local_idx
            fvar = float(var[gi]) if (var is not None and gi < len(var)) else 0.
            out_flux_err[idx] = np.sqrt(abs(fvar)) if fvar > 0 else 0.
            var_offset += n_free

        valid_pix = invvar > 0
        chi2 = float(np.sum(resid[valid_pix] ** 2 * invvar[valid_pix]))
        n_params = sum(src.numberOfParams() for src in sources)
        dof = max(1, int(valid_pix.sum()) - n_params)
        for idx in members:
            out_chi2[idx] = chi2
            out_dof[idx] = dof

    # Compute per-source flux error scaled by chi2/dof (DES-style)
    s2 = np.where(out_dof > 0, out_chi2 / out_dof, 1.)
    out_flux_err_scaled = out_flux_err * np.sqrt(np.maximum(s2, 1.))

    result = Table()
    result['id'] = cat_in['id']
    result['ra'] = cat_in['ra']
    result['dec'] = cat_in['dec']
    result['name'] = cat_in['name']
    result[f'{band}_flux'] = out_flux
    result[f'{band}_flux_err'] = out_flux_err
    result[f'{band}_flux_err_scaled'] = out_flux_err_scaled
    result[f'{band}_chi2'] = out_chi2
    result[f'{band}_dof'] = out_dof.astype(int)
    if zeropoint is not None:
        result.meta[f'{band}_zeropoint'] = zeropoint

    logger.info(f'Forced photometry done: {len(result)} sources, '
                f'band={band}, median flux_err={np.nanmedian(out_flux_err):.4e}')
    return result
