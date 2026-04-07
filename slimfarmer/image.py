"""
FarmerImage: single-image (or multi-band) photometry pipeline.

Detection → grouping → model selection (detection band) → forced photometry (all bands).
"""

import logging
import sys
import numpy as np
import sep

import astropy.units as u
from astropy.io import fits
from astropy.table import Table, Column
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.wcs.utils import proj_plane_pixel_scales
from collections import OrderedDict

from tractor import PixelizedPSF, PixelizedPsfEx, HybridPixelizedPSF
import tqdm as _tqdm

#try:
#    from pathos.pools import ProcessPool
#    HAS_PATHOS = True
#except ImportError:
HAS_PATHOS = False
import multiprocessing

from ._group import _Group, _process_group, _process_group_second_pass
from .utils import (clean_catalog, dilate_and_group, get_detection_kernel,
                    get_params, segmap_to_dict)
from .config import Config


def _sys_byteorder():
    return '>' if sys.byteorder == 'big' else '<'


class FarmerImage:
    """
    Single-image (or multi-band) Tractor photometry pipeline.

    Parameters
    ----------
    bands : dict
        ``{band_name: {'science': path, 'psf': path, 'zeropoint': float,
                        'weight': path (optional)}}``
        All bands must be on the same pixel grid.
    detection_band : str, optional
        Band used for SEP source detection and model-type selection.
        Defaults to the first key in ``bands``.
    config : Config, optional
    """

    def __init__(self, bands, detection_band=None, config=None):
        self.logger = logging.getLogger('slimfarmer')
        self.config = config or Config()

        if detection_band is None:
            detection_band = list(bands.keys())[0]
        self.detection_band = detection_band
        self.bands = list(bands.keys())

        # ── Load all bands ────────────────────────────────────────────────────
        self.band_config = {}   # {band: {'science', 'weight', 'mask', 'psf_model', 'zeropoint'}}
        self.wcs = None
        self.pixel_scale = None

        for band, bconf in bands.items():
            sci_path = bconf['science']
            self.logger.info(f'Loading {band} from {sci_path}')
            with fits.open(sci_path) as hdul:
                ext = next((i for i, h in enumerate(hdul)
                            if h.data is not None and h.data.ndim >= 2), 0)
                sci = hdul[ext].data.astype(np.float64)
                header = hdul[ext].header
            if self.wcs is None:
                self.wcs = WCS(header)
                pscl = proj_plane_pixel_scales(self.wcs) * u.deg
                self.pixel_scale = pscl[0].to(u.arcsec)

            wt_path = bconf.get('weight')
            if wt_path is not None:
                with fits.open(wt_path) as hdul:
                    ext = next((i for i, h in enumerate(hdul)
                                if h.data is not None and h.data.ndim >= 2), 0)
                    wht = hdul[ext].data.astype(np.float64)
            else:
                _, _, rms = sigma_clipped_stats(sci[sci != 0])
                wht = np.where(rms > 0, np.ones_like(sci) / rms ** 2, 0.)
                self.logger.info(f'  {band}: no weight — generated from clipped RMS = {rms:.4g}')

            eff_gain_path = bconf.get('eff_gain')
            if eff_gain_path is not None:
                with fits.open(eff_gain_path) as hdul:
                    ext = next((i for i, h in enumerate(hdul)
                                if h.data is not None and h.data.ndim >= 2), 0)
                    eff_gain = hdul[ext].data.astype(np.float64)
            else:
                eff_gain = None

            noise_reals_path = bconf.get('noise_reals')

            psf_path = bconf['psf']
            if psf_path.endswith('.psf'):
                try:
                    psf_model = PixelizedPsfEx(fn=psf_path)
                except Exception:
                    img = fits.getdata(psf_path).astype(np.float32)
                    img[~np.isfinite(img) | (img < 1e-31)] = 1e-31
                    psf_model = HybridPixelizedPSF(PixelizedPSF(img))
            else:
                img = fits.getdata(psf_path).astype(np.float32)
                img[~np.isfinite(img) | (img < 1e-31)] = 1e-31
                psf_model = HybridPixelizedPSF(PixelizedPSF(img))
            if self.config.renorm_psf is not None:
                psf_model.img *= self.config.renorm_psf / float(np.nansum(psf_model.img))

            bad = ~np.isfinite(sci)
            sci[bad] = 0.
            wht[~np.isfinite(wht) | (wht < 0) | bad] = 0.

            # Precompute noise correlation r(dx,dy) once from full image
            noise_corr = None
            if noise_reals_path is not None:
                ny_f, nx_f = sci.shape
                # Estimate per-pixel sigma directly from the noise realizations.
                # This avoids unit mismatch between noise layers and weight map.
                with fits.open(noise_reals_path) as hdul:
                    nr_all = hdul[0].data.astype(np.float64)
                sigma_nr = np.std(nr_all, axis=0, ddof=1)
                # Smooth to reduce noise from few realizations
                from scipy.ndimage import uniform_filter
                sigma_nr = uniform_filter(sigma_nr, size=15)
                sigma_nr = np.where(sigma_nr > 0, sigma_nr, 1.0)

                n_real = nr_all.shape[0]
                R_2d = np.zeros((ny_f, nx_f), dtype=np.float64)
                for k in range(n_real):
                    n_norm = nr_all[k] / sigma_nr
                    R_2d += np.abs(np.fft.fft2(n_norm)) ** 2
                    del n_norm
                R_2d /= n_real
                noise_corr = np.fft.ifft2(R_2d).real / (ny_f * nx_f)
                del R_2d, nr_all, sigma_nr

            self.band_config[band] = {
                'science':          sci,
                'weight':           wht,
                'eff_gain':         eff_gain,
                'noise_corr':       noise_corr,
                'noise_reals_path': noise_reals_path,
                'mask':             (wht <= 0) | ~np.isfinite(sci),
                'psf_model':        psf_model,
                'zeropoint':        bconf['zeropoint'],
            }

        # Shared base mask from detection band
        self.mask = self.band_config[detection_band]['mask'].copy()

        # Outputs
        self.catalog = None
        self.segmap = None
        self.groupmap = None
        self.segmap_dict = None
        self.groupmap_dict = None
        self.model_catalog = OrderedDict()
        self.model_tracker = OrderedDict()

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self):
        """
        SEP source detection on the detection band + dilation grouping.

        Sets ``self.catalog``, ``self.segmap``, ``self.groupmap``,
        ``self.segmap_dict``, ``self.groupmap_dict``.
        """
        self.logger.info('Running SEP source detection...')
        cfg = self.config
        det = self.band_config[self.detection_band]
        img = det['science'].copy()

        if img.dtype.byteorder not in ('=', '|', _sys_byteorder()):
            img = img.astype(img.dtype.newbyteorder('='))

        var = None
        if cfg.use_detection_weight:
            wgt = det['weight'].copy()
            if wgt.dtype.byteorder not in ('=', '|', _sys_byteorder()):
                wgt = wgt.astype(wgt.dtype.newbyteorder('='))
            var = np.where(wgt > 0, 1. / wgt, 0.)

        if cfg.subtract_background:
            bkg = sep.Background(img, bw=cfg.back_bw, bh=cfg.back_bh,
                                 fw=cfg.back_fw, fh=cfg.back_fh)
            img = img - bkg.back()

        convfilt = None
        if cfg.filter_kernel is not None:
            convfilt = get_detection_kernel(cfg.filter_kernel)

        sep.set_extract_pixstack(cfg.pixstack_size)
        catalog, segmap = sep.extract(
            img, cfg.thresh, var=var, minarea=cfg.minarea,
            filter_kernel=convfilt, filter_type=cfg.filter_type,
            segmentation_map=True,
            clean=cfg.clean, clean_param=cfg.clean_param,
            deblend_nthresh=cfg.deblend_nthresh, deblend_cont=cfg.deblend_cont,
        )

        if len(catalog) == 0:
            raise RuntimeError('No sources detected! Check image and threshold.')

        self.logger.info(f'Detected {len(catalog)} sources.')
        catalog = Table(catalog)
        catalog.add_column(np.arange(1, len(catalog) + 1, dtype=np.int32), name='id', index=0)
        sky = self.wcs.all_pix2world(catalog['x'], catalog['y'], 0)
        catalog.add_column(sky[0] * u.deg, name='ra', index=1)
        catalog.add_column(sky[1] * u.deg, name='dec', index=2)

        radius_arcsec = cfg.dilation_radius.to(u.arcsec).value
        radius_px = round(radius_arcsec / self.pixel_scale.to(u.arcsec).value)
        self.logger.info(f'Grouping with dilation radius {radius_arcsec:.2f}" = {radius_px} px')

        group_ids, group_pops, groupmap = dilate_and_group(
            catalog, segmap, radius=radius_px, fill_holes=True)

        catalog.add_column(group_ids, name='group_id', index=3)
        catalog.add_column(group_pops, name='group_pop', index=4)
        catalog.add_column(np.ones(len(catalog), dtype=np.int32), name='brick_id', index=0)

        ny, nx = img.shape
        pad = cfg.paddingpixel
        near_boundary = ((catalog['x'] < pad) | (catalog['x'] >= nx - pad) |
                         (catalog['y'] < pad) | (catalog['y'] >= ny - pad))
        catalog['flag'][near_boundary] |= 0x0100

        self.catalog = catalog
        self.segmap = segmap
        self.groupmap = groupmap
        self.segmap_dict = segmap_to_dict(segmap)
        self.groupmap_dict = segmap_to_dict(groupmap)
        self.logger.info(f'Found {int(groupmap.max())} groups.')
        return catalog

    # ── Group spawning ────────────────────────────────────────────────────────

    def _spawn_group(self, group_id):
        """Build a _Group object with data cutouts for all bands."""
        cat = self.catalog
        subset = cat[cat['group_id'] == group_id]

        def _rejected(reason=''):
            g = _Group.__new__(_Group)
            g.rejected = True
            g.group_id = group_id
            g.source_ids = np.array(subset['id']) if len(subset) else np.array([], dtype=int)
            g.logger = logging.getLogger(f'slimfarmer.group_{group_id}')
            g.model_catalog = OrderedDict()
            g.model_tracker = OrderedDict()
            return g

        if len(subset) == 0:
            return _rejected()
        if len(subset) > self.config.group_size_limit:
            return _rejected()

        gy, gx = self.groupmap_dict.get(int(group_id), (np.array([]), np.array([])))
        if len(gy) == 0:
            return _rejected()

        ylo, yhi = int(np.min(gy)), int(np.max(gy))
        xlo, xhi = int(np.min(gx)), int(np.max(gx))
        yc = (ylo + yhi) / 2.
        xc = (xlo + xhi) / 2.
        position = self.wcs.pixel_to_world(xc, yc)

        pscl = self.pixel_scale.to(u.arcsec).value
        h_arcsec = ((yhi - ylo) + 1) * pscl * u.arcsec
        w_arcsec = ((xhi - xlo) + 1) * pscl * u.arcsec
        buf = self.config.group_buffer.to(u.arcsec)
        buffsize = (h_arcsec + 2 * buf, w_arcsec + 2 * buf)

        # Create cutouts for all bands
        try:
            det_sci_cut = Cutout2D(
                self.band_config[self.detection_band]['science'],
                position, buffsize, wcs=self.wcs, mode='partial', fill_value=0., copy=True)
        except (ValueError, IndexError):
            return _rejected()

        local_h, local_w = det_sci_cut.data.shape
        # Use center-based offset instead of origin_original: when the cutout extends
        # beyond the image boundary (mode='partial'), origin_original gives the first
        # *valid* pixel in the original image, but the padded array's local y=0 may
        # correspond to a negative row in the original image.  The center relationship
        # is always correct regardless of boundary padding.
        ox = int(round(det_sci_cut.center_original[0] - det_sci_cut.center_cutout[0]))
        oy = int(round(det_sci_cut.center_original[1] - det_sci_cut.center_cutout[1]))
        origin = (ox, oy)

        band_data = {}
        for band, bc in self.band_config.items():
            try:
                sci_cut = Cutout2D(bc['science'], position, buffsize,
                                   wcs=self.wcs, mode='partial', fill_value=0., copy=True)
                wht_cut = Cutout2D(bc['weight'], position, buffsize,
                                   wcs=self.wcs, mode='partial', fill_value=0., copy=True)
            except (ValueError, IndexError):
                return _rejected()
            eff_gain_cut = None
            if bc['eff_gain'] is not None:
                try:
                    eff_gain_cut = Cutout2D(bc['eff_gain'], position, buffsize,
                                            wcs=self.wcs, mode='partial',
                                            fill_value=0., copy=True).data
                except (ValueError, IndexError):
                    pass
            band_data[band] = {
                'sci':      sci_cut.data,
                'wht':      wht_cut.data,
                'eff_gain': eff_gain_cut,
                'psf':      bc['psf_model'],
                'zeropoint': bc['zeropoint'],
            }

        # Shared mask cutout
        try:
            msk_cut = Cutout2D(self.mask.astype(float), position, buffsize,
                               wcs=self.wcs, mode='partial', fill_value=1., copy=True)
        except (ValueError, IndexError):
            return _rejected()

        # Build LOCAL segmap / groupmap dicts (subtract cutout origin)
        seg_local = {}
        for sid_int in np.array(subset['id']):
            sid = int(sid_int)
            if sid not in self.segmap_dict:
                continue
            full_y, full_x = self.segmap_dict[sid]
            ly = full_y.astype(int) - oy
            lx = full_x.astype(int) - ox
            valid = (ly >= 0) & (ly < local_h) & (lx >= 0) & (lx < local_w)
            if np.any(valid):
                seg_local[sid] = (ly[valid], lx[valid])

        grp_local = {}
        for gid_int, (full_y, full_x) in self.groupmap_dict.items():
            ly = full_y.astype(int) - oy
            lx = full_x.astype(int) - ox
            valid = (ly >= 0) & (ly < local_h) & (lx >= 0) & (lx < local_w)
            if np.any(valid):
                grp_local[int(gid_int)] = (ly[valid], lx[valid])

        pscl_qty = proj_plane_pixel_scales(det_sci_cut.wcs) * u.deg

        return _Group(
            group_id=int(group_id),
            band_data=band_data,
            detection_band=self.detection_band,
            msk=msk_cut.data.astype(bool),
            catalog_subset=subset,
            segmap_local=seg_local,
            groupmap_local=grp_local,
            wcs=det_sci_cut.wcs,
            pixel_scales=pscl_qty,
            config=self.config,
            origin=origin,
        )

    # ── Absorb results ───────────────────────────────────────────────────────

    def _absorb(self, result):
        group_id, model_catalog, model_tracker = result
        self.model_catalog.update(model_catalog)
        for key, tracker in model_tracker.items():
            if key != 'group':
                self.model_tracker[key] = tracker

    # ── Neighbor lookup ───────────────────────────────────────────────────────

    def _get_nearby_models(self, group_id, radius_deg):
        """Return model_catalog entries for sources near this group."""
        subset = self.catalog[self.catalog['group_id'] == group_id]
        if len(subset) == 0 or not self.model_catalog:
            return {}
        ra_cen  = float(np.mean([float(r['ra'])  for r in subset]))
        dec_cen = float(np.mean([float(r['dec']) for r in subset]))
        cos_dec = np.cos(np.deg2rad(dec_cen))
        group_sids = set(int(s) for s in subset['id'])
        neighboring = {}
        for sid, model in self.model_catalog.items():
            if sid in group_sids:
                continue
            try:
                d_ra  = (model.pos.ra  - ra_cen) * cos_dec
                d_dec =  model.pos.dec - dec_cen
                if np.sqrt(d_ra**2 + d_dec**2) <= radius_deg:
                    neighboring[sid] = model
            except Exception:
                pass
        return neighboring

    # ── Process groups ────────────────────────────────────────────────────────

    def process_groups(self, group_ids=None):
        """
        Fit all (or specified) groups with Tractor.

        Parameters
        ----------
        group_ids : array-like, optional
            Subset of group IDs to process. Default: all.
        """
        if self.catalog is None:
            raise RuntimeError('Run detect() first.')

        if group_ids is None:
            group_ids = np.unique(self.catalog['group_id'])
        group_ids = np.atleast_1d(group_ids)

        self.logger.info(f'Processing {len(group_ids)} groups (ncpus={self.config.ncpus})...')

        if self.config.ncpus == 0 or len(group_ids) == 1:
            for gid in _tqdm.tqdm(group_ids, desc='Groups'):
                g = self._spawn_group(gid)
                self._absorb(_process_group(g))
        else:
            import os as _os
            for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'MKL_NUM_THREADS', 'BLAS_NUM_THREADS'):
                _os.environ.setdefault(_var, '1')
            import multiprocessing as mp
            saved = {}
            for band in self.band_config:
                saved[band] = {}
                for key in ('noise_corr', 'noise_reals_path'):
                    saved[band][key] = self.band_config[band].pop(key, None)
            groups = [self._spawn_group(gid) for gid in group_ids]
            pool = mp.Pool(processes=self.config.ncpus, maxtasksperchild=20)
            async_results = [pool.apply_async(_process_group, (g,)) for g in groups]
            del groups
            pool.close()
            pbar = _tqdm.tqdm(total=len(async_results), desc='Groups')
            for ar in async_results:
                try:
                    result = ar.get(timeout=self.config.timeout)
                    self._absorb(result)
                except Exception:
                    pass
                pbar.update(1)
            pbar.close()
            pool.terminate()
            pool.join()
            for band in self.band_config:
                for key, val in saved[band].items():
                    if val is not None:
                        self.band_config[band][key] = val

        self.logger.info(f'Finished — {len(self.model_catalog)} sources fit.')

        # ── Optional second pass: forced photometry with neighbor subtraction ──
        if self.config.neighbor_subtraction:
            radius_deg = self.config.neighbor_radius.to(u.deg).value
            self.logger.info(
                f'Neighbor-subtraction pass (radius={self.config.neighbor_radius})...')
            pass1_catalog = dict(self.model_catalog)  # snapshot of first-pass models
            if self.config.ncpus == 0 or len(group_ids) == 1:
                for gid in _tqdm.tqdm(group_ids, desc='Neighbors'):
                    g2 = self._spawn_group(gid)
                    neighboring = self._get_nearby_models(gid, radius_deg)
                    pass1 = {sid: pass1_catalog[sid]
                             for sid in g2.source_ids if sid in pass1_catalog}
                    result = _process_group_second_pass((g2, neighboring, pass1))
                    self._absorb(result)
            else:
                args_list = []
                for gid in group_ids:
                    g2 = self._spawn_group(gid)
                    neighboring = self._get_nearby_models(gid, radius_deg)
                    pass1 = {sid: pass1_catalog[sid]
                             for sid in g2.source_ids if sid in pass1_catalog}
                    args_list.append((g2, neighboring, pass1))
                with mp.Pool(processes=self.config.ncpus,
                             maxtasksperchild=50) as pool:
                    for result in _tqdm.tqdm(
                            pool.imap(_process_group_second_pass, args_list),
                            total=len(args_list), desc='Neighbors'):
                        self._absorb(result)
                del args_list
            self.logger.info(f'Neighbor pass done — {len(self.model_catalog)} sources.')

    # ── Post-fit correlated-noise kappa ─────────────────────────────────────

    def compute_kappa(self):
        """Compute kappa using cached h data.

        For noshot=True: uses noise_corr from init (sigma_bg, exact).
        For noshot=False: re-estimates r with sigma_total, one band at
        a time, freeing memory between bands.
        """
        import gc

        for band in self.bands:
            bc = self.band_config[band]
            noise_corr = bc.get('noise_corr')
            if noise_corr is None:
                continue
            ny, nx = noise_corr.shape

            # noise_corr from init already uses sigma estimated from
            # the realizations themselves, so no re-estimation needed.

            for source_id, model in _tqdm.tqdm(self.model_catalog.items()):
                if not hasattr(model, 'flux_err_noisereal_kappa'):
                    model.flux_err_noisereal_kappa = {}

                cache = getattr(model, '_kappa_cache', {}).get(band)
                if cache is None:
                    model.flux_err_noisereal_kappa[band] = 1.0
                    continue

                h_vals = cache['h_vals'].astype(np.float64)
                nzy = cache['nzy'].astype(np.intp)
                nzx = cache['nzx'].astype(np.intp)
                D = cache['D']

                hTRh = 0.0
                for i in range(len(h_vals)):
                    dy_i = (nzy[i] - nzy) % ny
                    dx_i = (nzx[i] - nzx) % nx
                    hTRh += h_vals[i] * np.dot(h_vals, noise_corr[dy_i, dx_i])

                kappa_sq = hTRh / D
                model.flux_err_noisereal_kappa[band] = np.sqrt(max(kappa_sq, 1.0))

            del noise_corr; gc.collect()

    # ── Build catalog ─────────────────────────────────────────────────────────

    def build_catalog(self):
        """
        Merge detection + photometry into a single astropy Table.

        Returns columns for the detection band (position, shape, flux) and
        flux-only columns for any additional bands.
        """
        if self.catalog is None:
            raise RuntimeError('Run detect() first.')

        catalog = self.catalog.copy()
        det_zp = self.band_config[self.detection_band]['zeropoint']

        for source_id, model in self.model_catalog.items():
            if not hasattr(model, 'statistics'):
                continue
            row_mask = catalog['id'] == source_id

            # Detection band: full params (position + shape + flux)
            params = get_params(model, self.detection_band, det_zp)

            # Additional bands: flux columns only
            for band in self.bands:
                if band == self.detection_band:
                    continue
                zp = self.band_config[band]['zeropoint']
                for k, v in get_params(model, band, zp).items():
                    if k.startswith(f'{band}_'):
                        params[k] = v

            for name, value in params.items():
                try:
                    unit = value.unit
                    value = float(value.value)
                except Exception:
                    unit = None
                dtype = 'S20' if isinstance(value, str) else type(value)
                if name not in catalog.colnames:
                    catalog.add_column(Column(length=len(catalog),
                                              name=name, dtype=dtype, unit=unit))
                catalog[name][row_mask] = value

        # Sources not fit by Tractor (rejected groups, optimisation failures)
        # fall back to PointSource so the name column is never empty.
        if 'name' in catalog.colnames:
            unfit = np.array([(n == b'' or n == '') for n in catalog['name']], dtype=bool)
            if unfit.any():
                catalog['name'][unfit] = 'Bad'

        return catalog

    def write_catalog(self, path, overwrite=True):
        """Build and write catalog to a FITS file."""
        cat = self.build_catalog()
        cat.write(path, overwrite=overwrite)
        self.logger.info(f'Catalog written → {path}  ({len(cat)} rows)')
        return cat

    def build_model_image(self, band=None):
        """
        Render the full-image model from the fitted model_catalog.

        Parameters
        ----------
        band : str, optional
            Band to render. Defaults to detection_band.

        Returns
        -------
        model_img : 2-D ndarray, same shape as the science image.
        """
        import copy
        from tractor import Image as TImage, Tractor as TTractor, FluxesPhotoCal, ConstantSky
        from .utils import read_wcs

        if band is None:
            band = self.detection_band
        bc = self.band_config[band]

        sources = [copy.deepcopy(m) for m in self.model_catalog.values()
                   if hasattr(m, 'pos')]
        if not sources:
            self.logger.warning('No fitted sources in model_catalog — returning zero model.')
            return np.zeros_like(bc['science'])

        timg = TImage(
            data=bc['science'].copy(),
            invvar=bc['weight'].copy(),
            psf=bc['psf_model'],
            wcs=read_wcs(self.wcs),
            photocal=FluxesPhotoCal(band),
            sky=ConstantSky(0),
        )
        return TTractor([timg], sources).getModelImage(0)


# ── Module-level helper for pathos lazy generator ─────────────────────────────

def _spawn_and_process(farmer, gid):
    return farmer._spawn_group(gid)


# ── Convenience top-level function ────────────────────────────────────────────

def run_photometry(science_path=None, psf_path=None, band=None, zeropoint=None,
                   weight_path=None, eff_gain_path=None, noise_reals_path=None,
                   bands=None,
                   detection_band=None, output_path=None, group_ids=None,
                   config=None, **config_kwargs):
    """
    One-call photometry pipeline.  Supports both single-band and multi-band modes.

    Single-band (backward-compatible)
    ----------------------------------
    run_photometry(science_path, psf_path, band, zeropoint, weight_path=...,
                   eff_gain_path=..., ...)

    Multi-band
    ----------
    run_photometry(
        bands={
            'F158': {'science': path, 'psf': path, 'zeropoint': 26.5,
                     'weight': path, 'eff_gain': path},
            'F106': {'science': path, 'psf': path, 'zeropoint': 26.3},
        },
        detection_band='F158',
    )

    Returns
    -------
    catalog : astropy.table.Table
    """
    if bands is None:
        if science_path is None or psf_path is None or band is None or zeropoint is None:
            raise ValueError(
                "Provide either a 'bands' dict or all of: "
                "science_path, psf_path, band, zeropoint")
        bands = {band: {'science': science_path, 'psf': psf_path,
                        'zeropoint': zeropoint, 'weight': weight_path,
                        'eff_gain': eff_gain_path,
                        'noise_reals': noise_reals_path}}

    if config is None:
        config = Config(**config_kwargs)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s :: %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
    )

    img = FarmerImage(bands, detection_band=detection_band, config=config)
    img.detect()
    img.process_groups(group_ids=group_ids)
    import gc, ctypes
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass
    logger = logging.getLogger('slimfarmer')
    logger.info('All groups done. Computing kappa...')
    img.compute_kappa()
    logger.info('Kappa done. Building catalog...')
    cat = img.build_catalog()
    if output_path is not None:
        cat.write(output_path, overwrite=True)
    if config.save_model_image:
        hdr = img.wcs.to_header()
        base = output_path.rsplit('.fits', 1)[0] if output_path else 'slimfarmer_output'
        logger = logging.getLogger('slimfarmer')
        for band in img.band_config:
            model_img = img.build_model_image(band=band)
            sci = img.band_config[band]['science']
            residual = (sci - model_img).astype(np.float32)
            suffix = f'_{band}' if len(img.band_config) > 1 else ''
            fits.writeto(base + suffix + '_model.fits',    model_img.astype(np.float32), header=hdr, overwrite=True)
            fits.writeto(base + suffix + '_residual.fits', residual,                     header=hdr, overwrite=True)
            logger.info(f'Model    → {base}{suffix}_model.fits')
            logger.info(f'Residual → {base}{suffix}_residual.fits')
    return cat, img
