"""
_Group: in-memory group for Tractor model fitting.

Each group contains one or more nearby sources that are fit simultaneously.
Supports single-band (model selection) and multi-band (forced photometry).
"""

import os
import copy
import time
import logging
import numpy as np
from collections import OrderedDict

import astropy.units as u
from tractor import (Image, Tractor, FluxesPhotoCal, ConstantSky,
                     EllipseESoft, Fluxes)
from tractor.galaxy import ExpGalaxy, DevGalaxy, FixedCompositeGalaxy, SoftenedFracDev
from tractor.pointsource import PointSource
from tractor.constrained_optimizer import ConstrainedOptimizer as TractorOptimizer
#from tractor.gpu_optimizer import GpuOptimizer as TractorOptimizer
#from tractor.dense_optimizer import ConstrainedDenseOptimizer as TractorOptimizer
from tractor.wcs import RaDecPos

from scipy.ndimage import binary_dilation
from .utils import SimpleGalaxy, read_wcs, set_priors, create_circular_mask


class _Group:
    """
    Fits one group of nearby sources using Tractor.

    Parameters
    ----------
    group_id : int
    band_data : dict
        ``{band_name: {'sci': array, 'wht': array, 'psf': psf_model, 'zeropoint': float}}``
        All bands share the same pixel grid / WCS.
    detection_band : str
        Band used for source detection and model-type selection.
        Must be a key in ``band_data``.
    msk : 2-D bool array
        Base mask for this cutout (True = bad pixel / outside image).
    catalog_subset : astropy.table.Table
        Detection catalog rows for sources in this group.
    segmap_local : dict  {source_id: (y_arr, x_arr)}
    groupmap_local : dict  {group_id: (y_arr, x_arr)}
    wcs : astropy WCS  (local cutout WCS)
    pixel_scales : astropy Quantity array
    config : Config
    origin : (x_col_min, y_row_min)  0-indexed in the full image
    """

    def __init__(self, group_id, band_data, detection_band,
                 msk, catalog_subset, segmap_local, groupmap_local,
                 wcs, pixel_scales, config, origin):

        self.group_id = group_id
        self.type = 'group'
        self.logger = logging.getLogger(f'slimfarmer.group_{group_id}')
        self.rejected = False

        self.config = config
        self.detection_band = detection_band
        self.bands = list(band_data.keys())
        self.pixel_scales = pixel_scales
        self.origin = origin

        # Base mask (group boundary + bad pixels from detection band)
        self.msk = msk.astype(bool)

        # Per-band image data (cleaned)
        self.band_data = {}
        for band, bd in band_data.items():
            sci = bd['sci'].copy().astype(np.float64)
            wht = bd['wht'].copy().astype(np.float64)
            bad = ~np.isfinite(sci)
            sci[bad] = 0.
            wht[~np.isfinite(wht) | (wht < 0) | bad] = 0.
            eff_gain = bd.get('eff_gain')
            if eff_gain is not None:
                eff_gain = eff_gain.copy().astype(np.float64)
                eff_gain[~np.isfinite(eff_gain) | (eff_gain <= 0)] = np.inf
            self.band_data[band] = {
                'sci':      sci,
                'wht':      wht,
                'eff_gain': eff_gain,
                'psf':      bd['psf'],
                'zeropoint': bd['zeropoint'],
            }

        # Segmap / groupmap in local pixel coordinates
        self.segmap_local = segmap_local
        self.groupmap_local = groupmap_local

        # Catalog
        self.catalog = catalog_subset
        self.catalog_band = 'detection'
        self.catalog_imgtype = 'science'
        self.source_ids = np.array(catalog_subset['id'])
        self.catalogs = {self.catalog_band: {self.catalog_imgtype: catalog_subset}}

        # WCS
        self.wcs = wcs

        # Fitting state
        self.images = OrderedDict()
        self.engine = None
        self.model_catalog = OrderedDict()
        self.model_tracker = OrderedDict()
        self.stage = None
        self.solved = None
        self.variance = None
        self.existing_model_catalog = None

    # ── WCS ──────────────────────────────────────────────────────────────────

    def _make_tractor_wcs(self):
        """Build Tractor WCS with correct x0/y0 offset (mirrors the_farmer logic)."""
        tractor_wcs = read_wcs(self.wcs)
        try:
            src = self.catalog[0]
            ok, raw_x, raw_y = tractor_wcs.wcs.radec2pixelxy(
                float(src['ra']), float(src['dec']))
            cat_local_x = float(src['x']) - float(self.origin[0])
            cat_local_y = float(src['y']) - float(self.origin[1])
            x0 = (raw_x - 1) - cat_local_x
            y0 = (raw_y - 1) - cat_local_y
            tractor_wcs.setX0Y0(x0, y0)
        except Exception as e:
            self.logger.debug(f'Could not set WCS x0/y0: {e}')
        return tractor_wcs

    # ── Image staging ────────────────────────────────────────────────────────

    def _stage_images(self, bands=None):
        """
        Create Tractor Image objects for the specified bands.

        Parameters
        ----------
        bands : list of str, optional
            Bands to stage. Default: all bands in ``self.band_data``.
            The detection band should appear first so index-0 chi stats
            are computed for it during model selection.
        """
        if bands is None:
            bands = self.bands
        self.images = OrderedDict()
        tractor_wcs = self._make_tractor_wcs()

        for band in bands:
            if band not in self.band_data:
                continue
            bd = self.band_data[band]
            data = bd['sci'].copy()
            weight = bd['wht'].copy()
            masked = self.msk.astype(np.float32)

            # Zero-out pixels outside this group's fitting footprint.
            # The fitting footprint is the groupmap dilated by fit_dilation_radius,
            # which is larger than the grouping dilation to capture profile wings.
            if self.group_id in self.groupmap_local:
                gy, gx = self.groupmap_local[self.group_id]
                fit_mask = np.zeros(masked.shape, dtype=bool)
                fit_mask[gy, gx] = True
                pscl = self.pixel_scales[0].to(u.arcsec).value
                fit_r_px = round(self.config.fit_dilation_radius.to(u.arcsec).value / pscl)
                if fit_r_px > 0:
                    struct = create_circular_mask(2 * fit_r_px, 2 * fit_r_px, radius=fit_r_px)
                    fit_mask = binary_dilation(fit_mask, structure=struct)
                masked[~fit_mask] = 1.

            weight[~np.isfinite(data) | (masked == 1) | ~np.isfinite(masked)] = 0.
            data[~np.isfinite(data)] = 0.
            weight[~np.isfinite(weight)] = 0.

            if weight.sum() == 0:
                self.logger.debug(f'All weight pixels zero for band {band} — skipping.')
                continue

            self.images[band] = Image(
                data=data,
                invvar=weight,
                psf=bd['psf'],
                wcs=tractor_wcs,
                photocal=FluxesPhotoCal(band),
                sky=ConstantSky(0),
            )
            # Store background-only invvar for model-based Poisson update
            self.band_data[band]['invvar_bg'] = weight.copy()

    # ── Model staging ────────────────────────────────────────────────────────

    def _stage_models(self):
        """Create initial Tractor source models from the detection catalog."""
        staged_bands = list(self.images.keys())
        for src in self.catalog:
            source_id = int(src['id'])
            if source_id not in self.model_catalog:
                continue
            position = RaDecPos(float(src['ra']), float(src['dec']))

            # Initial flux: primary band gets aperture sum; all others start at 0.
            # Starting non-primary bands at 0 ensures background-only invvar on the
            # first optimisation step, giving an optimal matched-filter estimate and
            # avoiding Poisson-overshoot bias from noisy warm-started fluxes.
            # Primary = detection band if staged, else first staged band.
            primary_band = self.detection_band if self.detection_band in staged_bands else staged_bands[0]
            src_tracker = self.model_tracker.get(source_id, {})
            qflux = 0.0
            ref_band = self.detection_band if self.detection_band in staged_bands else (
                staged_bands[0] if staged_bands else None)
            if source_id in self.segmap_local and ref_band is not None:
                sy, sx = self.segmap_local[source_id]
                try:
                    qflux = float(np.nansum(self.images[ref_band].data[sy, sx]))
                except Exception:
                    qflux = 0.0

            flux = Fluxes(
                **{b: (qflux if b == primary_band else 0.0) for b in staged_bands},
                order=staged_bands,
            )

            tmpl = self.model_catalog[source_id]

            # Shape: warm-start from most recent stage with same model type; else use SEP moments
            prev_shape = None
            for prev_s in sorted([k for k in src_tracker if isinstance(k, int)], reverse=True):
                prev_model = src_tracker[prev_s].get('model')
                if (prev_model is not None
                        and hasattr(prev_model, 'shape')
                        and type(prev_model) is type(tmpl)):
                    try:
                        prev_shape = copy.deepcopy(prev_model.shape)
                        break
                    except Exception:
                        pass
            if prev_shape is not None:
                shape = prev_shape
            else:
                theta_val = float(src['theta'])
                if not np.isfinite(theta_val):
                    theta_val = 0.0
                pa = 90.0 - np.rad2deg(theta_val)
                pixscl = self.pixel_scales[0].to(u.arcsec).value
                a_pix = float(src['a'])
                b_pix = float(src['b'])
                if not np.isfinite(a_pix) or a_pix <= 0:
                    a_pix = 1.0
                if not np.isfinite(b_pix) or b_pix <= 0:
                    b_pix = a_pix
                guess_r = np.sqrt(a_pix * b_pix) * pixscl
                ba = min(b_pix / a_pix, 1.0)
                if self.config.fixed_reff is not None:
                    shape = EllipseESoft.fromRAbPhi(self.config.fixed_reff, ba, pa)
                else:
                    shape = EllipseESoft.fromRAbPhi(max(guess_r, 0.01), ba, pa)
            shape.lowers = [-3, -10.0, -10.0]
            shape.uppers = [1,   10.0,  10.0]
            if isinstance(tmpl, PointSource):
                model = PointSource(position, flux)
            elif isinstance(tmpl, SimpleGalaxy):
                model = SimpleGalaxy(position, flux)
            elif isinstance(tmpl, ExpGalaxy):
                model = ExpGalaxy(position, flux, shape)
            elif isinstance(tmpl, DevGalaxy):
                model = DevGalaxy(position, flux, shape)
            elif isinstance(tmpl, FixedCompositeGalaxy):
                model = FixedCompositeGalaxy(position, flux,
                                             SoftenedFracDev(0.5), shape, shape)
            else:
                model = PointSource(position, flux)

            model = set_priors(model, self.model_priors_active)
            model.variance = model.copy()
            model.statistics = {}
            self.model_catalog[source_id] = model

    # ── Optimisation ─────────────────────────────────────────────────────────

    def _update_invvar_with_model(self):
        """
        Update each Tractor image's invvar to include model-based Poisson noise.

        invvar = 1 / (bg_var + model_flux / eff_gain)

        where bg_var = 1 / invvar_bg (background-only variance) and
        eff_gain converts model flux to Poisson variance.  Called before each
        optimisation step so the noise model tracks the current best-fit model.
        If eff_gain is not available for a band, the invvar is left unchanged.
        """
        for i, (band, img) in enumerate(self.images.items()):
            bd = self.band_data[band]
            if bd.get('eff_gain') is None or 'invvar_bg' not in bd:
                continue
            model_img = self.engine.getModelImage(i)
            bg_var = np.where(bd['invvar_bg'] > 0, 1.0 / bd['invvar_bg'], np.inf)
            poisson_var = np.maximum(model_img, 0.) / bd['eff_gain']
            if not self.config.noshot:
                total_var = bg_var + poisson_var
            else:
                total_var = bg_var
            img.setInvvar(np.where(total_var > 0, 1.0 / total_var, 0.))

    def _optimize(self):
        """Run the Tractor optimiser; returns True on convergence."""
        import io, sys
        self.engine.optimizer = TractorOptimizer()
        _saved_stdout, _saved_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = io.StringIO()
        try:
            for i in range(self.config.max_steps):
                self._update_invvar_with_model()
                dlnp, X, alpha, var = self.engine.optimize(
                    variance=True, damping=self.config.damping, priors=True)
                if dlnp < self.config.dlnp_crit:
                    break
            if var is None:
                var = np.zeros(len(self.engine.getParams()))
            self.variance = var
            return True
        finally:
            sys.stdout, sys.stderr = _saved_stdout, _saved_stderr

    # ── Chi image and statistics ──────────────────────────────────────────────

    def _build_chi_image(self):
        """Chi = (data - model) * sqrt(invvar) for the first staged image."""
        if not self.images or self.engine is None:
            return None
        first_band = list(self.images.keys())[0]
        model_img = self.engine.getModelImage(0)
        chi = ((self.images[first_band].data - model_img)
               * np.sqrt(self.images[first_band].invvar))
        return chi

    def _measure_stats(self, stage):
        """Compute chi2 statistics per source and for the group."""
        chi = self._build_chi_image()
        if chi is None:
            return

        q_pc = (5, 16, 50, 84, 95)

        if self.group_id in self.groupmap_local:
            gy, gx = self.groupmap_local[self.group_id]
            group_chi = chi[gy, gx].flatten()
        else:
            group_chi = chi.flatten()

        group_chi2 = float(np.nansum(group_chi ** 2))
        first_img = list(self.images.values())[0]
        ndata = int(np.sum(first_img.invvar[gy, gx] > 0)) \
            if self.group_id in self.groupmap_local else 0
        try:
            nparam = self.engine.getCatalog().numberOfParams()
        except Exception:
            nparam = 0
        ndof = max(1, ndata - nparam)

        self.model_tracker['group'][stage] = {
            'chisq': group_chi2, 'rchisq': group_chi2 / ndof,
            'ndata': ndata, 'nparam': nparam, 'ndof': ndof,
            'total': {'rchisq': group_chi2 / ndof, 'chisq': group_chi2,
                      'ndata': ndata, 'nparam': nparam, 'ndof': ndof},
        }
        for pc, v in zip(q_pc, np.nanpercentile(group_chi, q=q_pc)):
            self.model_tracker['group'][stage][f'chi_pc{pc:02d}'] = v

        for src in self.catalog:
            sid = int(src['id'])
            if sid not in self.segmap_local:
                continue
            sy, sx = self.segmap_local[sid]
            src_chi = chi[sy, sx].flatten()
            chi2 = float(np.nansum(src_chi ** 2))
            ndata_s = len(sy)
            ndof_s = max(1, ndata_s - nparam)
            self.model_tracker[sid][stage] = {
                'chisq': chi2, 'rchisq': chi2 / ndof_s,
                'ndata': ndata_s, 'nparam': nparam, 'ndof': ndof_s,
                'total': {'rchisq': chi2 / ndof_s, 'chisq': chi2},
            }
            for pc, v in zip(q_pc, np.nanpercentile(src_chi, q=q_pc)):
                self.model_tracker[sid][stage][f'chi_pc{pc:02d}'] = v

    # ── Tracker initialisation ────────────────────────────────────────────────

    def _add_tracker(self, init_stage=0):
        if self.stage is None:
            self.stage = init_stage
            self.solved = np.zeros(len(self.source_ids), dtype=bool)
        if self.stage == init_stage:
            self.model_tracker['group'] = {}
        self.model_tracker['group'][self.stage] = {}
        for src in self.catalog:
            sid = int(src['id'])
            if sid not in self.model_catalog:
                self.model_catalog[sid] = PointSource(None, None)
                self.model_tracker[sid] = {}
            self.model_tracker[sid][self.stage] = {}

    def _reset_models(self):
        self.engine = None
        self.stage = None
        self.model_tracker = OrderedDict()
        self.model_catalog = OrderedDict()

    # ── Store / update models ─────────────────────────────────────────────────

    def _store_models(self):
        """Save fitted parameters back to model_catalog."""
        cat = self.engine.getCatalog()
        cat_var = copy.deepcopy(cat)
        cat_var.setParams(self.variance)

        # At forced-phot (stage 11), shape params are frozen so cat_var[i].shape
        # holds the *fitted value*, not a Fisher variance.  The model-selection
        # shape variances are still in cat[i].variance.shape (set by the final
        # determine_models _store_models call and preserved through _update_models
        # deepcopy).  Cache them here so we can restore them after the assignment.
        shape_var_cache = {}
        if self.stage == 11:
            for i, sid in enumerate(self.source_ids):
                if hasattr(cat[i], 'variance'):
                    shape_var_cache[sid] = self._get_shape_var(cat[i].variance)

        for i, sid in enumerate(self.source_ids):
            model = cat[i]
            variance = cat_var[i]
            model.variance = variance
            if sid in shape_var_cache:
                self._restore_shape_var(model.variance, shape_var_cache[sid])
            self.model_tracker[sid][self.stage]['model'] = model

            if self.solved is not None and (self.solved.all() or self.stage == 11):
                self.model_catalog[sid] = model
                self.model_catalog[sid].group_id = self.group_id
                self.model_catalog[sid].statistics = self.model_tracker[sid][self.stage]

    def _update_models(self):
        """Re-stage models from existing fitted shapes; update brightness for all bands."""
        staged_bands = list(self.images.keys())
        for src in self.catalog:
            sid = int(src['id'])
            if sid not in self.existing_model_catalog:
                # Keep PointSource from _stage_models; apply forced-phot priors
                if sid in self.model_catalog:
                    self.model_catalog[sid] = set_priors(
                        self.model_catalog[sid], self.model_priors_active)
                continue
            existing = self.existing_model_catalog[sid]

            # Build per-band flux dict; fallback to detection band for new bands
            flux_dict = {}
            for band in staged_bands:
                try:
                    fv = existing.getBrightness().getFlux(band)
                except Exception:
                #    try:
                #        fv = existing.getBrightness().getFlux(self.detection_band)
                #    except Exception:
                        fv = 0.0
                flux_dict[band] = fv

            flux   = Fluxes(**flux_dict, order=staged_bands)
            filler = Fluxes(**{b: 0.0 for b in staged_bands}, order=staged_bands)

            self.model_catalog[sid] = copy.deepcopy(existing)
            self.model_catalog[sid].brightness = flux
            self.model_catalog[sid].variance.brightness = filler
            self.model_catalog[sid] = set_priors(self.model_catalog[sid],
                                                  self.model_priors_active)

    # ── Decision tree ─────────────────────────────────────────────────────────

    def _decision_tree(self):
        """Chi2-based model selection — ported from blob.py decide_winners_chisq_opt1."""
        for i, sid in enumerate(self.source_ids):
            if self.solved[i]:
                continue
            s = self.stage

            if s == 1:
                self.model_catalog[sid] = SimpleGalaxy(None, None)

            elif s == 2:
                # blob.py level 0: PS wins if it beats SG by margin,
                # unless both exceed chisq_force_exp_dev (back-door for extended sources).
                ps_chi2 = self.model_tracker[sid][1]['total']['rchisq']
                sg_chi2 = self.model_tracker[sid][2]['total']['rchisq']
                ps_wins = (ps_chi2 - sg_chi2) < self.config.simplegalaxy_penalty
                force_exp_dev = (ps_chi2 > self.config.chisq_force_exp_dev and
                                 sg_chi2 > self.config.chisq_force_exp_dev)
                if ps_wins and not force_exp_dev:
                    self.model_catalog[sid] = PointSource(None, None)
                    self.solved[i] = True
                else:
                    self.model_catalog[sid] = ExpGalaxy(None, None, None)

            elif s == 3:
                self.model_catalog[sid] = DevGalaxy(None, None, None)

            elif s == 4:
                # blob.py level 1
                ps_chi2  = self.model_tracker[sid][1]['total']['rchisq']
                sg_chi2  = self.model_tracker[sid][2]['total']['rchisq']
                exp_chi2 = self.model_tracker[sid][3]['total']['rchisq']
                dev_chi2 = self.model_tracker[sid][4]['total']['rchisq']

                exp_beats_sg = exp_chi2 < sg_chi2
                dev_beats_sg = dev_chi2 < sg_chi2
                similar      = abs(exp_chi2 - dev_chi2) < self.config.exp_dev_similar_thresh
                force_comp   = (exp_chi2 > self.config.chisq_force_comp and
                                dev_chi2 > self.config.chisq_force_comp)

                # PS beats everything: go back (highest priority, overrides Exp/Dev)
                if (ps_chi2 < sg_chi2 + self.config.simplegalaxy_penalty and
                        ps_chi2 < exp_chi2 and ps_chi2 < dev_chi2):
                    self.model_catalog[sid] = PointSource(None, None)
                    self.solved[i] = True
                # SG beats both Exp and Dev AND is itself acceptable: go back to SimpleGalaxy
                elif (not exp_beats_sg and not dev_beats_sg and
                        sg_chi2 + self.config.simplegalaxy_penalty < ps_chi2 and
                        sg_chi2 < self.config.chisq_force_comp):
                    self.model_catalog[sid] = SimpleGalaxy(None, None)
                    self.solved[i] = True
                # Exp clearly beats SG and Dev
                elif exp_beats_sg and not similar and exp_chi2 < dev_chi2 and not force_comp:
                    self.model_catalog[sid] = ExpGalaxy(None, None, None)
                    self.solved[i] = True
                # Dev clearly beats SG and Exp
                elif dev_beats_sg and not similar and dev_chi2 < exp_chi2 and not force_comp:
                    self.model_catalog[sid] = DevGalaxy(None, None, None)
                    self.solved[i] = True
                # Both beat SG and are similar, or both still too bad: try Composite
                else:
                    self.model_catalog[sid] = FixedCompositeGalaxy(None, None, None, None, None)

            elif s == 5:
                # blob.py level 2: argmin over {Exp, Dev, Comp}, tie goes to Exp
                exp_chi2  = self.model_tracker[sid][3]['total']['rchisq']
                dev_chi2  = self.model_tracker[sid][4]['total']['rchisq']
                comp_chi2 = self.model_tracker[sid][5]['total']['rchisq']
                if comp_chi2 < exp_chi2 and comp_chi2 < dev_chi2:
                    self.model_catalog[sid] = FixedCompositeGalaxy(None, None, None, None, None)
                elif exp_chi2 <= dev_chi2:
                    self.model_catalog[sid] = ExpGalaxy(None, None, None)
                else:
                    self.model_catalog[sid] = DevGalaxy(None, None, None)
                self.solved[i] = True

    # ── Main fitting entry points ──────────────────────────────────────────────

    def determine_models(self):
        """Run model-type selection on the configured modeling bands (Stages 1–5+)."""
        self.model_priors_active = self.config.model_priors
        self._reset_models()
        self._add_tracker()
        bands_ordered = ([self.detection_band]
                         + [b for b in self.bands if b != self.detection_band])
        self._stage_images(bands=bands_ordered)
        if not self.images:
            return False
        tstart = time.time()
        while not self.solved.all():
            self.stage += 1
            self._add_tracker()
            self._stage_models()
            self.engine = Tractor(list(self.images.values()),
                                  [self.model_catalog[sid] for sid in self.source_ids
                                   if sid in self.model_catalog])
            self.engine.bands = list(self.images.keys())
            self.engine.freezeParam('images')
            ok = self._optimize()
            if not ok:
                return False
            self._measure_stats(self.stage)
            self._store_models()
            self._decision_tree()
        # Final convergence pass
        self.stage += 1
        self._add_tracker()
        self._stage_models()
        self.engine = Tractor(list(self.images.values()),
                              [self.model_catalog[sid] for sid in self.source_ids
                               if sid in self.model_catalog])
        self.engine.bands = list(self.images.keys())
        self.engine.freezeParam('images')
        ok = self._optimize()
        if not ok:
            return False
        self._measure_stats(self.stage)
        self._store_models()
        self.logger.debug(f'Modelling done ({time.time()-tstart:.2f}s)')
        return True

    def _compute_des_flux_err(self):
        """Compute DES-style flux error from pixel residuals (MOF afterburner approach).

        Mirrors mof/moflib.py:L1373 (esheldon/mof), restricted to each source's
        model footprint (model_i > 1e-3 × peak) and using the full invvar
        (background + model Poisson, from _update_invvar_with_model).

        Why footprint: the group fitting region is much larger than a MOF stamp,
        so background pixels (model≈0) dilute chi2/dof → 1 regardless of noise.
        Why full invvar (not invvar_bg): with invvar_bg, chi2/dof gives an
        unweighted average of σ_tot²/σ_bg² while the matched filter ψ²-weights
        those noisy center pixels, causing a systematic flux_err_des ~ 0.8×flux_err_tractor.
        With full invvar, chi2/dof ≈ 1 for a perfect fit → flux_err_des ≈ flux_err_tractor,
        and any model error / neighbour contamination raises chi2/dof > 1 →
        flux_err_des > flux_err_tractor.

        For each source i and band b:
            footprint = pixels where model_i > 1e-3 × max(model_i)
            chi2    = sum over footprint of (data - model_total)^2 * invvar
            msq_sum = sum over footprint of invvar * (model_i / flux_i)^2
            dof     = n_footprint_pixels - 1
            flux_err_des = sqrt(chi2 / msq_sum / dof)
        """
        if self.engine is None or not self.images:
            return

        nobj = len(self.source_ids)

        for ib, (band, img) in enumerate(self.images.items()):
            model_total = self.engine.getModelImage(ib)
            # Use img.invvar (background + model Poisson, updated by
            # _update_invvar_with_model).  For a perfect fit chi2/dof ≈ 1 and
            # flux_err_des ≈ flux_err_tractor; any model error, neighbour
            # contamination, or PSF mismatch drives chi2/dof > 1 so that
            # flux_err_des > flux_err_tractor.  Using invvar_bg instead causes
            # the unweighted chi2/dof to underestimate shot noise (shot noise is
            # ψ²-weighted in the MF but averaged uniformly in chi2), yielding a
            # systematic flux_err_des/flux_err_tractor ~ 0.8.
            invvar = img.invvar
            chi_image = (img.data - model_total) * np.sqrt(invvar)

            for sid in self.source_ids:
                if sid not in self.model_catalog:
                    continue
                model = self.model_catalog[sid]
                if not hasattr(model, 'flux_err_des'):
                    model.flux_err_des = {}

                try:
                    flux = model.getBrightness().getFlux(band)
                except Exception:
                    flux = 0.0

                if flux == 0.0:
                    model.flux_err_des[band] = 0.0
                    continue

                src_copy = copy.deepcopy(model)
                temp_engine = Tractor([img], [src_copy])
                temp_engine.bands = [band]
                model_img = temp_engine.getModelImage(0)

                # Restrict to the model footprint so that background-dominated
                # pixels (model ≈ 0) do not dilute chi2/dof toward 1 and wash
                # out the shot-noise signal.  In MOF each source has its own
                # small stamp so N_pix ≈ N_eff; here the group fitting region
                # is much larger, making the full-region chi2/dof ≈ 1 regardless
                # of shot noise → flux_err_des < flux_err_tractor (wrong).
                # Masking to the PSF footprint restores chi2/dof ≈ σ_tot²/σ_bg².
                model_peak = float(np.max(model_img))
                if model_peak <= 0:
                    model.flux_err_des[band] = 0.0
                    continue
                footprint = model_img > 1e-2 * model_peak

                invvar_foot = invvar[footprint]
                valid = invvar_foot > 0
                if not np.any(valid):
                    model.flux_err_des[band] = 0.0
                    continue

                chi2 = float(np.sum(chi_image[footprint][valid] ** 2))
                msq_sum = float(np.sum(invvar_foot[valid]
                                       * (model_img[footprint][valid] / flux) ** 2))
                ndof = max(1, int(np.sum(valid)) - 1)

                if msq_sum <= 0:
                    model.flux_err_des[band] = 0.0
                    continue

                arg = chi2 / msq_sum / ndof
                model.flux_err_des[band] = float(np.sqrt(arg)) if arg > 0 else 0.0

    def _get_shape_var(self, variance_obj):
        """Return {attr: deepcopy(attr)} for shape sub-objects in a variance model."""
        saved = {}
        for attr in ('shape', 'shapeExp', 'shapeDev'):
            obj = getattr(variance_obj, attr, None)
            if obj is not None:
                try:
                    saved[attr] = copy.deepcopy(obj)
                except Exception:
                    pass
        return saved

    def _restore_shape_var(self, variance_obj, saved):
        """Restore previously saved shape sub-objects into a variance model."""
        for attr, obj in saved.items():
            try:
                setattr(variance_obj, attr, obj)
            except Exception:
                pass

    def _logre_flux_deriv(self, model, img, band, delta):
        """Numerical central-difference dF_forced / d(logre).

        Perturbs logre by ±delta and computes the linear matched-filter flux
        estimate on the fitting footprint to get dF/d(logre).
        """
        from tractor.ellipses import EllipseESoft

        logre0 = float(model.shape.logre)
        ee1_0  = float(model.shape.ee1)
        ee2_0  = float(model.shape.ee2)

        def _linear_flux(logre_val):
            src = copy.deepcopy(model)
            # Construct a fresh (unfrozen) shape so rendering works regardless
            # of whether the original shape params are frozen.
            src.shape = EllipseESoft(logre_val, ee1_0, ee2_0)
            temp = Tractor([img], [src])
            temp.bands = [band]
            model_img = temp.getModelImage(0)
            try:
                flux = float(src.getBrightness().getFlux(band))
            except Exception:
                return None
            if abs(flux) < 1e-30:
                return None
            unit_t = model_img / flux
            denom = float(np.sum(img.invvar * unit_t ** 2))
            if denom <= 0:
                return None
            return float(np.sum(img.invvar * img.data * unit_t)) / denom

        F_plus  = _linear_flux(logre0 + delta)
        F_minus = _linear_flux(logre0 - delta)
        if F_plus is None or F_minus is None:
            return 0.0
        return (F_plus - F_minus) / (2.0 * delta)

    def _compute_flux_err_corr(self):
        """Propagate model-selection logre uncertainty into flux_err.

        For ExpGalaxy / DevGalaxy sources, estimates dF/d(logre) numerically
        and adds the size-propagation term in quadrature with the Fisher error:

            flux_err_corr = sqrt( flux_err² + (dF/d·logre × logre_err)² )

        Result stored as model.flux_err_corr[band].  For PointSource /
        SimpleGalaxy / FixedCompositeGalaxy, flux_err_corr = flux_err.
        """
        if self.engine is None or not self.images:
            return

        D_LOGRE = 0.05  # ±0.05 in log(arcsec) ≈ ±5 % change in r_e

        for ib, (band, img) in enumerate(self.images.items()):
            for sid in self.source_ids:
                if sid not in self.model_catalog:
                    continue
                model = self.model_catalog[sid]
                if not hasattr(model, 'flux_err_corr'):
                    model.flux_err_corr = {}

                try:
                    flux_var = model.variance.getBrightness().getFlux(band)
                    flux_err = float(np.sqrt(abs(flux_var))) if flux_var > 0 else 0.0
                except Exception:
                    model.flux_err_corr[band] = 0.0
                    continue

                if not isinstance(model, (ExpGalaxy, DevGalaxy)):
                    model.flux_err_corr[band] = flux_err
                    continue

                try:
                    logre_var = model.variance.shape.logre
                    logre_err = float(np.sqrt(abs(logre_var))) if logre_var > 0 else 0.0
                except Exception:
                    model.flux_err_corr[band] = flux_err
                    continue

                if logre_err == 0.0:
                    model.flux_err_corr[band] = flux_err
                    continue

                try:
                    dF = self._logre_flux_deriv(model, img, band, D_LOGRE)
                except Exception:
                    model.flux_err_corr[band] = flux_err
                    continue

                sigma_prop = abs(dF) * logre_err
                model.flux_err_corr[band] = float(
                    np.sqrt(flux_err ** 2 + sigma_prop ** 2))

    def _compute_flux_err_noshot(self):
        """Marginalized Fisher flux error using background-only invvar.

        Builds the full Fisher matrix over all thawed source parameters
        (flux, position, shape, …) using invvar_bg (background-only weights,
        shot noise excluded), adds Gaussian prior contributions, inverts,
        and extracts the marginalized flux variance.

        This correctly accounts for the noise-correlated template effect:
        when non-flux parameters are jointly fitted, the template correlates
        with the noise realization, inflating the effective flux scatter
        beyond the fixed-template Fisher bound.

        Result stored as model.flux_err_noshot_raw[band].
        """
        if self.engine is None or not self.images:
            return

        for ib, (band, img) in enumerate(self.images.items()):
            invvar_bg = self.band_data[band].get('invvar_bg')
            if invvar_bg is None:
                continue
            self._compute_marginal_fisher_flux_err(
                band, img, invvar_bg, 'flux_err_noshot_raw')

    def _compute_flux_err_shot(self):
        """Marginalized Fisher flux error using total invvar (bg + shot noise).

        Same as _compute_flux_err_noshot but uses the full inverse variance
        that includes model-based Poisson shot noise, giving the flux error
        expected when shot noise from the source is present.

        Result stored as model.flux_err_shot_raw[band].
        """
        if self.engine is None or not self.images:
            return

        for ib, (band, img) in enumerate(self.images.items()):
            bd = self.band_data[band]
            invvar_bg = bd.get('invvar_bg')
            if invvar_bg is None:
                continue
            eff_gain = bd.get('eff_gain')
            model_img = self.engine.getModelImage(ib)
            bg_var = np.where(invvar_bg > 0, 1.0 / invvar_bg, np.inf)
            poisson_var = (np.maximum(model_img, 0.0) / eff_gain
                          if eff_gain is not None
                          else np.zeros_like(bg_var))
            total_var = bg_var + poisson_var
            invvar_total = np.where(total_var > 0, 1.0 / total_var, 0.0)
            self._compute_marginal_fisher_flux_err(
                band, img, invvar_total, 'flux_err_shot_raw')

    def _compute_marginal_fisher_flux_err(self, band, img, invvar, attr_name):
        """Shared logic: build Fisher matrix with given invvar, extract flux err.

        Stores two values per source per band:
          model.<attr_name>[band]          — marginalized flux error
          model.<attr_name + '_fixed'>[band] — fixed-template (diagonal) flux error
        The ratio of these is used to scale flux_err_des.
        """
        fixed_attr = attr_name + '_fixed'
        for sid in self.source_ids:
            if sid not in self.model_catalog:
                continue
            model = self.model_catalog[sid]
            if not hasattr(model, attr_name):
                setattr(model, attr_name, {})
            if not hasattr(model, fixed_attr):
                setattr(model, fixed_attr, {})

            try:
                flux = float(model.getBrightness().getFlux(band))
            except Exception:
                flux = 0.0

            if flux == 0.0:
                getattr(model, attr_name)[band] = 0.0
                getattr(model, fixed_attr)[band] = 0.0
                continue

            src_copy = copy.deepcopy(model)
            temp_engine = Tractor([img], [src_copy])
            temp_engine.bands = [band]
            temp_engine.freezeParam('images')

            unit_t = temp_engine.getModelImage(0) / flux
            F_ff = float(np.sum(invvar * unit_t ** 2))
            fixed_err = 1.0 / np.sqrt(F_ff) if F_ff > 0 else 0.0
            getattr(model, fixed_attr)[band] = fixed_err

            p0 = np.array(src_copy.getParams())
            n_params = len(p0)
            if n_params == 0:
                getattr(model, attr_name)[band] = fixed_err
                continue

            step_sizes = np.array(src_copy.getStepSizes())
            derivs = np.zeros((n_params, invvar.size))

            for ip in range(n_params):
                dp = max(step_sizes[ip], 1e-10)
                pp = p0.copy(); pp[ip] += dp
                src_copy.setParams(pp)
                mp = temp_engine.getModelImage(0).ravel()
                pm = p0.copy(); pm[ip] -= dp
                src_copy.setParams(pm)
                mm = temp_engine.getModelImage(0).ravel()
                derivs[ip] = (mp - mm) / (2.0 * dp)
                src_copy.setParams(p0)

            w = invvar.ravel()
            fisher = np.zeros((n_params, n_params))
            for i in range(n_params):
                for j in range(i, n_params):
                    val = float(np.sum(w * derivs[i] * derivs[j]))
                    fisher[i, j] = val
                    fisher[j, i] = val

            if hasattr(src_copy, 'getLogPriorDerivatives'):
                try:
                    prior_derivs = src_copy.getLogPriorDerivatives()
                    for ip, (_, _, dd) in enumerate(prior_derivs):
                        fisher[ip, ip] += max(-dd, 0.0)
                except Exception:
                    pass

            flux_idx = None
            param_names = src_copy.getParamNames()
            for ip, pn in enumerate(param_names):
                if band in pn or pn.startswith('brightness'):
                    flux_idx = ip
                    break
            if flux_idx is None:
                flux_idx = 0

            try:
                cov = np.linalg.inv(fisher)
                margvar = max(cov[flux_idx, flux_idx], 0.0)
                getattr(model, attr_name)[band] = float(np.sqrt(margvar))
            except np.linalg.LinAlgError:
                getattr(model, attr_name)[band] = fixed_err

    def _subtract_neighbor_contributions(self, neighboring_models):
        """Subtract PSF-convolved neighbor model images from the staged image data."""
        neighbor_list = [copy.deepcopy(m) for sid, m in neighboring_models.items()
                         if sid not in self.source_ids]
        if not neighbor_list:
            return
        for band, img in self.images.items():
            temp_tractor = Tractor([img], neighbor_list)
            temp_tractor.bands = [band]
            temp_tractor.freezeParam('images')
            try:
                contrib = temp_tractor.getModelImage(0)
                img.data = img.data - contrib
            except Exception as e:
                self.logger.debug(f'Neighbor subtraction failed for band {band}: {e}')

    def force_models(self, neighboring_models=None):
        """
        Forced photometry: freeze shape parameters, re-fit flux in all bands.

        Detection band is staged first so chi statistics are computed for it.
        If neighboring_models is provided, their PSF-convolved contributions are
        subtracted from the image data before fitting (neighbor subtraction pass).
        """
        self.model_priors_active = self.config.phot_priors
        self.existing_model_catalog = copy.deepcopy(self.model_catalog)
        if not self.existing_model_catalog:
            return False
        self._reset_models()
        self._add_tracker(init_stage=10)
        # Stage detection band first, then other bands
        bands_ordered = ([self.detection_band]
                         + [b for b in self.bands if b != self.detection_band])
        self._stage_images(bands=bands_ordered)
        if not self.images:
            return False
        if neighboring_models:
            self._subtract_neighbor_contributions(neighboring_models)
        self._stage_models()
        self.stage = 11
        self._add_tracker()
        self._update_models()
        self.engine = Tractor(list(self.images.values()),
                              [self.model_catalog[sid] for sid in self.source_ids
                               if sid in self.model_catalog])
        self.engine.bands = list(self.images.keys())
        self.engine.freezeParam('images')
        self._measure_stats(10)
        ok = self._optimize()
        if not ok:
            return False
        self._measure_stats(self.stage)
        self._store_models()
        self._compute_des_flux_err()
        self._compute_flux_err_corr()
        self._compute_flux_err_noshot()
        self._compute_flux_err_shot()
        return True


# ── Module-level helpers (must be at top level for multiprocessing pickle) ─────

def _process_group(group):
    """Determine models + force photometry for a group; return serialisable result."""
    tstart = time.time()
    if not group.rejected:
        ok = group.determine_models()
        if ok:
            group.force_models()
    elapsed = time.time() - tstart
    for sid in group.source_ids:
        if sid in group.model_tracker:
            stages = [k for k in group.model_tracker[sid] if isinstance(k, int)]
            if stages:
                group.model_tracker[sid][max(stages)]['group_time'] = elapsed
    return group.group_id, group.model_catalog.copy(), group.model_tracker.copy()


def _process_group_second_pass(args):
    """Second-pass forced photometry with neighbor model subtraction."""
    group, neighboring_models, pass1_models = args
    if group.rejected or not pass1_models:
        return group.group_id, group.model_catalog.copy(), group.model_tracker.copy()
    # Restore pass-1 model_catalog so force_models can warm-start from it
    group.model_catalog = copy.deepcopy(pass1_models)
    tstart = time.time()
    ok = group.force_models(neighboring_models=neighboring_models)
    elapsed = time.time() - tstart
    for sid in group.source_ids:
        if sid in group.model_tracker:
            stages = [k for k in group.model_tracker[sid] if isinstance(k, int)]
            if stages:
                group.model_tracker[sid][max(stages)]['group_time'] = elapsed
    return group.group_id, group.model_catalog.copy(), group.model_tracker.copy()
