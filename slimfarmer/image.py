"""
FarmerImage: single-image (or multi-band) photometry pipeline.

Detection → grouping → model selection (detection band) → forced photometry (all bands).
"""

import logging
import sys
import time
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
from .flags import FLAG_BOUNDARY, FLAG_TIMEOUT, FLAG_SINGLETON_FALLBACK


def _sys_byteorder():
    return '>' if sys.byteorder == 'big' else '<'


def _singleton_worker(singleton_group, timeout, result_queue):
    """Run ``_process_group`` on a one-source group and push the result.

    Run as a top-level target of :class:`multiprocessing.Process` so the
    parent can kill an individual hung singleton with ``p.terminate()``
    without losing the others. Pushes the ``_process_group`` return value
    (tuple) on success or ``None`` on exception.
    """
    try:
        result = _process_group(singleton_group, timeout=timeout)
    except Exception:
        result = None
    try:
        result_queue.put(result)
    except Exception:
        pass


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

            # noise_corr is built lazily by recompute_noise_corr_with_model()
            # after the initial fit pass. See doc/compute_kappa_implementation.md
            # §3 for why the pre-fit empirical-sigma estimate was removed.
            self.band_config[band] = {
                'science':          sci,
                'weight':           wht,
                'eff_gain':         eff_gain,
                'noise_corr':       None,
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
        sep.set_sub_object_limit(2048)
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

        if cfg.dilation_radius is None:
            self.logger.info('Grouping disabled: one source per group.')
            group_ids = np.asarray(catalog['id'], dtype=np.int32)
            group_pops = np.ones(len(catalog), dtype=np.int16)
            groupmap = segmap.copy()
        else:
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
        catalog['flag'][near_boundary] |= FLAG_BOUNDARY

        self.catalog = catalog
        self.segmap = segmap
        self.groupmap = groupmap
        self.segmap_dict = segmap_to_dict(segmap)
        self.groupmap_dict = segmap_to_dict(groupmap)
        self.logger.info(f'Found {int(groupmap.max())} groups.')
        return catalog

    # ── Stitching helper ──────────────────────────────────────────────────────

    def drop_isolated_groups_outside_pixbox(self, central_pixbox):
        """Drop sources whose group has no member touching `central_pixbox`.

        A source is considered "touching" the central box iff its SEP
        bounding box ``[xmin..xmax] x [ymin..ymax]`` intersects the
        ``(x0, y0, x1, y1)`` half-open rectangle ``central_pixbox``.

        Sources whose centroid is outside the central box but whose segmap
        bbox extends into it are *kept* (their flux still affects the fit
        of central-block neighbors). Pure-buffer-ring groups — i.e. groups
        where no member's bbox intersects the central box — are dropped
        entirely so they're not modeled.

        This must be called after :meth:`detect` and before
        :meth:`process_groups`.
        """
        if self.catalog is None:
            raise RuntimeError('Call detect() before drop_isolated_groups_outside_pixbox().')
        x0, y0, x1, y1 = central_pixbox
        cat = self.catalog
        if not all(c in cat.colnames for c in ('xmin', 'xmax', 'ymin', 'ymax')):
            raise RuntimeError('Catalog is missing SEP bbox columns; cannot filter.')
        bb_x0 = np.asarray(cat['xmin'])
        bb_x1 = np.asarray(cat['xmax']) + 1
        bb_y0 = np.asarray(cat['ymin'])
        bb_y1 = np.asarray(cat['ymax']) + 1
        touches_central = (bb_x1 > x0) & (bb_x0 < x1) & (bb_y1 > y0) & (bb_y0 < y1)

        gids = np.asarray(cat['group_id'])
        central_gids = set(int(g) for g in np.unique(gids[touches_central]))
        keep_mask = np.array([int(g) in central_gids for g in gids])

        n_before = len(cat)
        kept_ids  = set(int(i) for i in np.asarray(cat['id'])[keep_mask])
        self.catalog = cat[keep_mask]

        if self.segmap_dict is not None:
            self.segmap_dict = {k: v for k, v in self.segmap_dict.items()
                                if int(k) in kept_ids}
        if self.groupmap_dict is not None:
            self.groupmap_dict = {k: v for k, v in self.groupmap_dict.items()
                                  if int(k) in central_gids}

        self.logger.info(
            f'Stitching filter: kept {len(self.catalog)}/{n_before} sources '
            f'({len(central_gids)} groups touch central box {central_pixbox}); '
            f'dropped {n_before - len(self.catalog)} isolated buffer-ring detections.'
        )

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
            timeout = self.config.timeout
            for gid in _tqdm.tqdm(group_ids, desc='Groups'):
                g = self._spawn_group(gid)
                self._absorb(_process_group(g, timeout=timeout))
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
            timeout = self.config.timeout
            # Hard wall-clock ceiling per task. When a worker hangs in a C
            # extension (Ceres) and ignores the cooperative `timeout` deadline,
            # ar.get() would otherwise block forever — this cap lets us skip
            # the stuck task and move on.
            stuck_ceiling = self.config.stuck_ceiling
            groups = [self._spawn_group(gid) for gid in group_ids]
            pool = mp.Pool(processes=self.config.ncpus, maxtasksperchild=20)
            stuck_gids = []
            try:
                async_results = [pool.apply_async(_process_group, (g, timeout))
                                 for g in groups]
                del groups
                pool.close()
                pbar = _tqdm.tqdm(total=len(async_results), desc='Groups')
                n_stuck = 0
                for gid, ar in zip(group_ids, async_results):
                    try:
                        result = ar.get(timeout=stuck_ceiling)
                        self._absorb(result)
                    except mp.TimeoutError:
                        n_stuck += 1
                        stuck_gids.append(int(gid))
                    except Exception:
                        pass
                    pbar.update(1)
                pbar.close()
                if n_stuck:
                    self.logger.warning(
                        f'{n_stuck} group(s) exceeded {stuck_ceiling}s '
                        f'hard ceiling and were skipped (worker hang).')
            finally:
                # terminate first: workers may hang in C-extension finalizers,
                # and all results have already been collected above.
                pool.terminate()
                pool.join()

            # Singleton-fallback retry for hard-stuck groups: the soft-deadline
            # fallback inside _process_group can't fire for workers hung in a
            # C extension because the function never returns. Re-submit each
            # source of each stuck group as an independent one-source _Group
            # in a fresh pool with the same hard ceiling.
            if stuck_gids and getattr(self.config, 'singleton_fallback', True):
                from ._group import _make_singleton_group
                singletons = []  # list of (singleton_Group, parent_gid, sid)
                for gid in stuck_gids:
                    parent = self._spawn_group(gid)
                    if getattr(parent, 'rejected', False) or len(parent.source_ids) == 0:
                        continue
                    for sid in parent.source_ids:
                        s = _make_singleton_group(parent, int(sid))
                        if s is not None:
                            singletons.append((s, int(gid), int(sid)))
                if singletons:
                    ncpus_eff = max(1, int(self.config.ncpus) or 1)
                    self.logger.info(
                        f'Retrying {len(singletons)} sources from {len(stuck_gids)} '
                        f'stuck group(s) as singletons '
                        f'(per-source ceiling {stuck_ceiling}s, up to {ncpus_eff} in flight)...')
                    n_saved = 0
                    n_killed = 0
                    t_fallback_start = time.time()
                    pbar2 = _tqdm.tqdm(total=len(singletons), desc='Singletons')
                    # Rolling window: keep up to ncpus processes running at
                    # any time. Whenever one finishes (normally or via kill),
                    # immediately start the next queued singleton. Each
                    # singleton has its own per-source deadline; Process.
                    # terminate() kills Ceres-bound C code too.
                    active = []  # list of (process, queue, parent_gid, sid, t_start)
                    idx = 0
                    while idx < len(singletons) or active:
                        # Fill up to ncpus
                        while len(active) < ncpus_eff and idx < len(singletons):
                            s, parent_gid, sid = singletons[idx]
                            q = mp.Queue()
                            p = mp.Process(target=_singleton_worker,
                                           args=(s, timeout, q))
                            p.start()
                            active.append((p, q, parent_gid, sid, time.time()))
                            idx += 1
                        # Reap any finished or timed-out entries
                        now = time.time()
                        for entry in list(active):
                            p, q, parent_gid, sid, tstart = entry
                            if not p.is_alive():
                                active.remove(entry)
                                try:
                                    result = q.get_nowait()
                                except Exception:
                                    result = None
                                if result is None:
                                    pbar2.update(1)
                                    continue
                                self._absorb(result)
                                if sid in self.model_catalog:
                                    self.model_catalog[sid].group_id = parent_gid
                                if sid in self.model_tracker:
                                    stages = [k for k in self.model_tracker[sid]
                                              if isinstance(k, int)]
                                    if stages:
                                        last = self.model_tracker[sid][max(stages)]
                                        last['singleton_fallback'] = True
                                        last['timed_out'] = True
                                n_saved += 1
                                pbar2.update(1)
                            elif now - tstart > stuck_ceiling:
                                p.terminate()
                                p.join(timeout=5)
                                if p.is_alive():
                                    p.kill()
                                    p.join(timeout=5)
                                active.remove(entry)
                                n_killed += 1
                                pbar2.update(1)
                        if active:
                            time.sleep(0.5)
                    pbar2.close()
                    self.logger.info(
                        f'Singleton fallback recovered {n_saved}/'
                        f'{len(singletons)} sources '
                        f'({n_killed} killed on per-source ceiling, '
                        f'elapsed {time.time()-t_fallback_start:.0f}s).')
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

    # Crossover: N² < Npix·log2(Npix) ⟹ sparse wins. For a 2354² image
    # that's N < ~sqrt(2354² · 21) ≈ 10800. We use a conservative 3000
    # so the sparse path stays comfortably fast even with the Python loop.
    _KAPPA_SPARSE_CUTOFF = 3000

    @staticmethod
    def _kappa_for_source(cache, noise_corr, ny, nx, Rk=None,
                          sparse_cutoff=3000):
        """Compute kappa for one source — hybrid sparse / FFT.

        For N < ``sparse_cutoff`` nonzero pixels uses a direct O(N²)
        double-sum; for larger groups falls back to the O(Npix log Npix)
        FFT convolution path. ``Rk`` (precomputed ``rfft2(noise_corr)``)
        must be supplied when the FFT path might be needed.
        """
        h_vals = cache['h_vals'].astype(np.float64)
        nzy = cache['nzy'].astype(np.intp)
        nzx = cache['nzx'].astype(np.intp)
        D = cache['D']
        N = len(h_vals)
        if N == 0 or D <= 0:
            return 1.0

        if N < sparse_cutoff:
            hTRh = 0.0
            for i in range(N):
                dy_i = (nzy[i] - nzy) % ny
                dx_i = (nzx[i] - nzx) % nx
                hTRh += h_vals[i] * float(np.dot(h_vals, noise_corr[dy_i, dx_i]))
        else:
            H = np.zeros((ny, nx), dtype=np.float64)
            np.add.at(H, (nzy % ny, nzx % nx), h_vals)
            Hk = np.fft.rfft2(H)
            if Rk is None:
                Rk = np.fft.rfft2(noise_corr)
            conv = np.fft.irfft2(Rk * Hk, s=(ny, nx))
            hTRh = float(np.sum(H * conv))

        kappa_sq = hTRh / D
        return np.sqrt(max(kappa_sq, 1.0))

    def compute_kappa(self):
        """Compute kappa using cached h data.

        Uses a hybrid strategy per source: sparse O(N²) double-sum for
        small groups (N < ``_KAPPA_SPARSE_CUTOFF``), FFT convolution for
        large groups. Parallelised over sources via ThreadPoolExecutor
        when ``config.ncpus > 0``.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import gc

        ncpus = getattr(self.config, 'ncpus', 0)
        cutoff = self._KAPPA_SPARSE_CUTOFF

        for band in self.bands:
            bc = self.band_config[band]
            noise_corr = bc.get('noise_corr')
            if noise_corr is None:
                continue
            ny, nx = noise_corr.shape

            # Precompute Rk once per band for the FFT fallback path
            Rk = np.fft.rfft2(noise_corr)

            # Collect work items: (source_id, cache)
            work = []
            for source_id, model in self.model_catalog.items():
                if not hasattr(model, 'flux_err_noisereal_kappa'):
                    model.flux_err_noisereal_kappa = {}
                cache = getattr(model, '_kappa_cache', {}).get(band)
                if cache is None:
                    model.flux_err_noisereal_kappa[band] = 1.0
                else:
                    work.append((source_id, cache))

            if not work:
                continue

            n_fft = sum(1 for _, c in work if len(c['h_vals']) >= cutoff)
            self.logger.info(f'[{band}] kappa: {len(work)} sources '
                             f'({len(work) - n_fft} sparse, {n_fft} FFT)')

            if ncpus > 0:
                results = {}
                with ThreadPoolExecutor(max_workers=ncpus) as pool:
                    futures = {
                        pool.submit(self._kappa_for_source,
                                    cache, noise_corr, ny, nx,
                                    Rk=Rk, sparse_cutoff=cutoff): sid
                        for sid, cache in work
                    }
                    for fut in _tqdm.tqdm(as_completed(futures),
                                          total=len(futures),
                                          desc=f'Kappa {band}'):
                        sid = futures[fut]
                        results[sid] = fut.result()
                for sid, kappa in results.items():
                    self.model_catalog[sid].flux_err_noisereal_kappa[band] = kappa
            else:
                for sid, cache in _tqdm.tqdm(work, desc=f'Kappa {band}'):
                    kappa = self._kappa_for_source(
                        cache, noise_corr, ny, nx,
                        Rk=Rk, sparse_cutoff=cutoff)
                    self.model_catalog[sid].flux_err_noisereal_kappa[band] = kappa

            del noise_corr, Rk; gc.collect()

    def recompute_noise_corr_with_model(self, force_r00_to_one=True):
        """Rebuild ``noise_corr`` per band using fitted models + total variance.

        Background: the pre-fit ``noise_corr`` built in ``__init__`` normalises
        by the empirical per-pixel σ of the noise-realisation cube (smoothed).
        That's internally consistent but relies on a 15-px smoothing kernel
        and doesn't use the weight map or the fitted model at all. See
        ``doc/compute_kappa_implementation.md`` §3 for the diagnosis of why
        ``sqrt(w_bg) · n`` alone is non-stationary on IMCOM coadds.

        Post-fit, we can do better: build a total-variance weight that
        includes source shot noise from the fitted model,

            1/w_total(x) = 1/w_bg(x) + max(model(x), 0) / eff_gain(x)

        then compute ``noise_corr`` as the autocorrelation of
        ``sqrt(w_total) · n`` averaged over realisations. If the model and
        ``eff_gain`` are self-consistent with the noise realisation cube,
        ``n_bar = sqrt(w_total) · n`` has per-pixel variance ≈ 1 and the
        resulting ``noise_corr`` is a clean correlation coefficient function
        with ``r(0, 0) ≈ 1``.

        The method logs ``raw_r00`` per band *before* renormalisation. If it
        is close to 1, the post-fit formulation is self-consistent and the
        kappa values from the subsequent ``compute_kappa()`` pass are robust.
        If ``raw_r00`` drifts (say, outside 0.9–1.1), something in the model,
        ``eff_gain``, or the realisation cube doesn't line up and the result
        should be treated with suspicion.

        Intended call sequence:

            img.detect()
            img.process_groups()                   # initial fit pass
            img.recompute_noise_corr_with_model()  # refine noise_corr
            img.compute_kappa()                    # use the refined noise_corr

        Parameters
        ----------
        force_r00_to_one : bool
            If True, divide the rebuilt ``noise_corr`` by its central value
            so that ``r(0, 0) == 1`` exactly. The pre-normalisation value
            is logged either way.
        """
        import gc
        from astropy.io import fits

        if not self.model_catalog:
            self.logger.warning(
                'recompute_noise_corr_with_model: model_catalog is empty; '
                'call process_groups() first. Skipping.')
            return

        for band in self.bands:
            bc = self.band_config[band]
            noise_reals_path = bc.get('noise_reals_path')
            if noise_reals_path is None:
                self.logger.info(f'[{band}] no noise_reals_path — skipping rebuild')
                continue
            eff_gain = bc.get('eff_gain')
            if eff_gain is None:
                self.logger.info(
                    f'[{band}] no eff_gain map — cannot add shot noise; '
                    f'skipping rebuild')
                continue

            w_bg = bc['weight']
            bg_var = np.where(w_bg > 0, 1.0 / w_bg, np.inf)

            # Render the current best-fit model on this band's native grid
            try:
                model_img = self.build_model_image(band=band)
            except Exception as e:
                self.logger.warning(
                    f'[{band}] build_model_image failed ({e}); skipping rebuild')
                continue

            eff_g = np.where(eff_gain > 0, eff_gain, np.inf)
            poisson_var = np.maximum(model_img, 0.0) / eff_g
            total_var = bg_var + poisson_var
            w_total = np.where(np.isfinite(total_var) & (total_var > 0),
                               1.0 / total_var, 0.0)
            sqrt_w = np.sqrt(w_total)

            with fits.open(noise_reals_path) as hdul:
                nr_all = hdul[0].data.astype(np.float64)
            n_real, ny_f, nx_f = nr_all.shape

            # Power spectrum of sqrt(w_total) * n, averaged over realisations
            R_2d = np.zeros((ny_f, nx_f), dtype=np.float64)
            for k in range(n_real):
                n_norm = nr_all[k] * sqrt_w
                R_2d += np.abs(np.fft.fft2(n_norm)) ** 2
                del n_norm
            R_2d /= n_real

            noise_corr = np.fft.ifft2(R_2d).real / (ny_f * nx_f)

            # Diagnostic: how close is r(0, 0) to 1 before we force it?
            raw_r00 = float(noise_corr[0, 0])
            if force_r00_to_one:
                if raw_r00 > 0:
                    noise_corr = noise_corr / raw_r00
                else:
                    self.logger.warning(
                        f'[{band}] raw r(0,0) = {raw_r00:.4g} ≤ 0; '
                        f'keeping pre-normalisation noise_corr')

            bc['noise_corr'] = noise_corr
            msg = (f'[{band}] rebuilt noise_corr from w_total + model  '
                   f'(raw r00 = {raw_r00:.4f}'
                   + (', renormalised to 1' if force_r00_to_one else '')
                   + f', n_real = {n_real})')
            self.logger.info(msg)

            del nr_all, R_2d, noise_corr, w_bg, bg_var, w_total, sqrt_w
            del model_img, poisson_var, total_var
            gc.collect()

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

        # Flag sources whose group hit the timeout deadline. They have
        # valid models from the last completed stage but may not have
        # gone through full model selection or forced photometry.
        # Sources re-fit via the singleton fallback additionally carry
        # FLAG_SINGLETON_FALLBACK.
        for source_id, tracker in self.model_tracker.items():
            stages = [k for k in tracker if isinstance(k, int)]
            if not stages:
                continue
            last = tracker[max(stages)]
            row_mask = catalog['id'] == source_id
            if 'flag' not in catalog.colnames:
                continue
            if last.get('timed_out', False):
                catalog['flag'][row_mask] |= FLAG_TIMEOUT
            if last.get('singleton_fallback', False):
                catalog['flag'][row_mask] |= FLAG_SINGLETON_FALLBACK

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

        sources = []
        for m in self.model_catalog.values():
            if not hasattr(m, 'pos'):
                continue
            try:
                f = m.getBrightness().getFlux(band)
                if f is None:
                    continue
            except Exception:
                continue
            sources.append(copy.deepcopy(m))
        if not sources:
            self.logger.warning(f'No sources with valid {band} flux — returning zero model.')
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
                   central_pixbox=None,
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
    if central_pixbox is not None:
        img.drop_isolated_groups_outside_pixbox(central_pixbox)
    img.process_groups(group_ids=group_ids)
    import gc, ctypes
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass
    logger = logging.getLogger('slimfarmer')
    logger.info('All groups done. Refining noise_corr with fitted models...')
    img.recompute_noise_corr_with_model()
    logger.info('Computing kappa with refined noise_corr...')
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
