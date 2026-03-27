"""
track_source — replay model selection for a single catalog source,
               optionally match to a truth catalog and plot the region.

Usage
-----
>>> from slimfarmer.track import track_source
>>> result = track_source(
...     source_id=317,
...     science_path='slimfarmer_outputroman_image.fits',
...     weight_path='slimfarmer_outputroman_weight.fits',
...     psf_path='slimfarmer_outputPSF_F158.fits',
...     band='F158',
...     zeropoint=26.511267817492374,
...     truth_pos_path='galaxy_10307.parquet',    # optional: positions
...     truth_flux_path='galaxy_flux_10307.parquet',  # optional: fluxes
...     plot=True,
...     plot_out='source_317.png',                # optional
... )
"""

import copy
import numpy as np
from .config import Config
from .image import FarmerImage


# ── Flux unit conversion (IMCOM DN ↔ nanomaggy) ───────────────────────────────

def _get_flux_converters(science_path, band):
    """
    Return (obs_to_nm, truth_to_nm) callables for IMCOM DN and Roman-internal
    flux units respectively, using the image pixel scale and GalSim zeropoints.
    """
    from astropy.io import fits as _fits
    try:
        import galsim.roman as _roman
        import galsim as _gs
        bp      = _gs.roman.getBandpasses()
        zp_gs   = bp[band].zeropoint if band in bp else bp['H158'].zeropoint
        ca      = _roman.collecting_area
    except Exception:
        return None, None

    with _fits.open(science_path) as h:
        ext = next((i for i, hh in enumerate(h)
                    if hh.data is not None and hh.data.ndim >= 2), 0)
        oversamplepix = abs(h[ext].header['CDELT2']) * 3600.  # arcsec/px


    def obs_to_nm(flux_imcom):
        """IMCOM image DN → nanomaggy."""
        fn  = float(flux_imcom)# * gain / (pix_size / oversamplepix) ** 2
        mag = -2.5 * np.log10(abs(fn) + 1e-30) + zp_gs + 2.5 * np.log10(ca)
        return 10 ** ((mag - 22.5) / -2.5)

    def truth_to_nm(flux_roman):
        """Roman-internal flux unit → nanomaggy."""
        mag = -2.5 * np.log10(abs(float(flux_roman)) + 1e-30) + zp_gs
        return 10 ** ((mag - 22.5) / -2.5)

    return obs_to_nm, truth_to_nm

_STAGE_MODEL = {
    0:  '(init)',
    1:  'PointSource',
    2:  'SimpleGalaxy',
    3:  'ExpGalaxy',
    4:  'DevGalaxy',
    5:  'FixedCompositeGalaxy',
    6:  '(final convergence)',
    10: '(forced phot init)',
    11: '(forced phot)',
}


def _annotate_decision(stage, tracker, cfg):
    """Human-readable description of the stage-N decision (blob.py opt1 logic)."""
    sgp  = cfg.simplegalaxy_penalty
    edst = cfg.exp_dev_similar_thresh
    fefd = cfg.chisq_force_exp_dev
    fcp  = cfg.chisq_force_comp

    def r(s):
        return tracker.get(s, {}).get('total', {}).get('rchisq', float('nan'))

    ps = r(1); sg = r(2); exp = r(3); dev = r(4); comp = r(5)

    if stage == 2:
        ps_wins   = (ps - sg) < sgp
        force_edv = ps > fefd and sg > fefd
        if ps_wins and not force_edv:
            return f'→ PointSource wins (ps-sg={ps-sg:.3f} < {sgp})'
        elif force_edv:
            return f'→ both > chisq_force_exp_dev={fefd}, try ExpGalaxy'
        return f'→ ps-sg={ps-sg:.3f} ≥ {sgp}, try ExpGalaxy'

    if stage == 4:
        exp_beats_sg = exp < sg
        dev_beats_sg = dev < sg
        similar      = abs(exp - dev) < edst
        force_comp   = exp > fcp and dev > fcp
        if ps < sg + sgp and ps < exp and ps < dev:
            return f'→ PointSource wins overall (ps={ps:.3f})'
        if not exp_beats_sg and not dev_beats_sg and sg + sgp < ps and sg < fcp:
            return f'→ SimpleGalaxy best (sg={sg:.3f} < exp,dev; sg < {fcp})'
        if exp_beats_sg and not similar and exp < dev and not force_comp:
            return f'→ ExpGalaxy wins (exp={exp:.3f} < sg={sg:.3f}, dev={dev:.3f})'
        if dev_beats_sg and not similar and dev < exp and not force_comp:
            return f'→ DevGalaxy wins (dev={dev:.3f} < sg={sg:.3f}, exp={exp:.3f})'
        if force_comp:
            return f'→ both > chisq_force_comp={fcp}, try Composite'
        return f'→ try Composite (similar={similar})'

    if stage == 5:
        if comp < exp and comp < dev:
            return f'→ Composite wins (comp={comp:.3f} < exp={exp:.3f}, dev={dev:.3f})'
        elif exp <= dev:
            return f'→ ExpGalaxy wins (exp={exp:.3f} ≤ dev={dev:.3f})'
        return f'→ DevGalaxy wins (dev={dev:.3f} < exp={exp:.3f})'

    return ''


def _match_truth(source_ra, source_dec, truth_pos_path, radius_arcsec=10.0,
                 truth_flux_path=None, truth_flux_col=None, truth_to_nm=None):
    """
    Load truth_pos parquet and return nearby matches sorted by separation.

    Parameters
    ----------
    source_ra, source_dec : float
    truth_pos_path        : str   parquet with columns galaxy_id, ra, dec
    radius_arcsec         : float search radius
    truth_flux_path       : str or None   parquet with galaxy_id + flux column
    truth_flux_col        : str or None   flux column name (auto-detected if None)
    truth_to_nm           : callable or None   converts truth flux → nanomaggy

    Returns
    -------
    list of dicts with keys: galaxy_id, ra, dec, sep_arcsec[, flux_nm]
    """
    import pandas as pd
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    pos = pd.read_parquet(truth_pos_path, columns=['galaxy_id', 'ra', 'dec'])

    # Merge flux if provided
    flux_df = None
    if truth_flux_path is not None:
        flux_df = pd.read_parquet(truth_flux_path)
        if truth_flux_col is None:
            candidates = [c for c in flux_df.columns
                          if c.startswith('roman_flux') or c.startswith('flux')]
            truth_flux_col = candidates[0] if candidates else None
        if truth_flux_col and truth_flux_col in flux_df.columns:
            flux_df = flux_df[['galaxy_id', truth_flux_col]]
            pos = pos.merge(flux_df, on='galaxy_id', how='left')
        else:
            flux_df = None

    src_coord = SkyCoord(source_ra, source_dec, unit='deg')
    t_coords  = SkyCoord(pos['ra'].values, pos['dec'].values, unit='deg')
    seps      = src_coord.separation(t_coords).to(u.arcsec).value

    mask  = seps < radius_arcsec
    order = np.argsort(seps[mask])

    matches = []
    for i in np.where(mask)[0][order]:
        entry = {
            'galaxy_id':  int(pos['galaxy_id'].iloc[i]),
            'ra':         float(pos['ra'].iloc[i]),
            'dec':        float(pos['dec'].iloc[i]),
            'sep_arcsec': float(seps[i]),
            'flux_nm':    float('nan'),
        }
        if flux_df is not None and truth_flux_col in pos.columns:
            raw = pos[truth_flux_col].iloc[i]
            if truth_to_nm is not None and not np.isnan(float(raw)):
                entry['flux_nm'] = truth_to_nm(float(raw))
        matches.append(entry)
    return matches


def _plot_track(group, source_id, img, band, truth_matches=None, out=None):
    """
    4-panel plot (science / model / residual / chi2) centred on source_id's group.

    Parameters
    ----------
    group         : fitted _Group object (after determine_models + force_models)
    source_id     : int
    img           : FarmerImage (for full-image WCS and pixel data)
    band          : str
    truth_matches : list of dicts from _match_truth (optional)
    out           : str output path (None → return figure only)
    """
    import matplotlib.pyplot as plt
    from tractor import Image as TImage, Tractor, FluxesPhotoCal, ConstantSky

    sci_full = img.band_config[band]['science']
    wht_full = img.band_config[band]['weight']
    psf_model = img.band_config[band]['psf_model']

    # Cutout bounds: group members + 80 px padding
    members = img.catalog[
        np.isin(img.catalog['id'], group.source_ids)]
    buf = 80
    x_lo = max(0, int(np.min(members['x'])) - buf)
    x_hi = min(sci_full.shape[1], int(np.max(members['x'])) + buf)
    y_lo = max(0, int(np.min(members['y'])) - buf)
    y_hi = min(sci_full.shape[0], int(np.max(members['y'])) + buf)

    sci = sci_full[y_lo:y_hi, x_lo:x_hi].copy()
    wht = wht_full[y_lo:y_hi, x_lo:x_hi].copy()
    wht[~np.isfinite(wht) | (wht < 0)] = 0.
    sci[~np.isfinite(sci)] = 0.

    # Tractor WCS for the cutout
    from .utils import read_wcs
    cut_wcs    = img.wcs.slice((slice(y_lo, y_hi), slice(x_lo, x_hi)))
    tractor_wcs = read_wcs(cut_wcs)

    tractor_img = TImage(
        data=sci, invvar=wht,
        psf=psf_model,
        wcs=tractor_wcs,
        photocal=FluxesPhotoCal(band),
        sky=ConstantSky(0),
    )

    # Build model from fitted group sources
    sources = [copy.deepcopy(m) for m in group.model_catalog.values()
               if hasattr(m, 'pos')]
    if not sources:
        print('No fitted sources available for plot.')
        return None

    tractor   = Tractor([tractor_img], sources)
    model_img = tractor.getModelImage(0)

    residual = sci - model_img
    chi = np.where(wht > 0, residual * np.sqrt(wht), 0.)
    chi2 = chi ** 2

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.subplots_adjust(wspace=0.05)
    extent = [x_lo, x_hi, y_lo, y_hi]

    def asinh_norm(data, stretch=0.5):
        mask = np.isfinite(data) & (data != 0)
        if not mask.any():
            return data
        vmed = np.median(data[mask])
        vmax = np.percentile(np.abs(data[mask] - vmed), 99.5)
        return np.arcsinh((data - vmed) / max(vmax * stretch, 1e-30))

    sci_n = asinh_norm(sci)
    axes[0].imshow(sci_n, origin='lower', cmap='gray_r',
                   extent=extent, aspect='equal')
    axes[0].set_title('Science', fontsize=11)

    mod_n = asinh_norm(model_img)
    axes[1].imshow(mod_n, origin='lower', cmap='gray_r',
                   extent=extent, aspect='equal',
                   vmin=sci_n.min(), vmax=sci_n.max())
    axes[1].set_title('Model', fontsize=11)

    res_lim = np.percentile(np.abs(residual[wht > 0]), 99) if (wht > 0).any() else 1.
    im2 = axes[2].imshow(residual, origin='lower', cmap='RdBu_r',
                          extent=extent, aspect='equal',
                          vmin=-res_lim, vmax=res_lim)
    axes[2].set_title('Residual', fontsize=11)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='DN')

    chi2_disp = np.where(wht > 0, chi2, np.nan)
    im3 = axes[3].imshow(chi2_disp, origin='lower', cmap='hot_r',
                          extent=extent, aspect='equal', vmin=0, vmax=25)
    axes[3].set_title(r'$\chi^2$ per pixel', fontsize=11)
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel('x (pixel)')
    axes[0].set_ylabel('y (pixel)')
    for ax in axes[1:]:
        ax.set_yticklabels([])

    # Mark detected group sources (cyan cross; target in red)
    for sid, model in group.model_catalog.items():
        if not hasattr(model, 'pos'):
            continue
        try:
            px, py = img.wcs.all_world2pix(
                float(model.pos.ra), float(model.pos.dec), 0)
            color = 'red' if sid == source_id else 'cyan'
            size  = 150  if sid == source_id else 60
            for ax in axes:
                ax.scatter([px], [py], marker='+', c=color, s=size,
                           linewidths=1.5, zorder=5)
                ax.text(px + 2, py + 2, str(sid), color=color,
                        fontsize=7, fontweight='bold', zorder=6,
                        bbox=dict(facecolor='black', alpha=0.4,
                                  pad=1, edgecolor='none'))
        except Exception:
            pass

    # Mark truth sources — filled diamond, white edge, opaque label box
    if truth_matches:
        for k, tm in enumerate(truth_matches):
            try:
                tx, ty = img.wcs.all_world2pix(tm['ra'], tm['dec'], 0)
                label  = f"{tm['sep_arcsec']:.2f}\""
                for ax in axes:
                    ax.scatter([tx], [ty], marker='D', s=80,
                               c='lime', edgecolors='black',
                               linewidths=0.8, zorder=8)
                    ax.text(tx + 3, ty + 3, label,
                            color='lime', fontsize=7, fontweight='bold',
                            zorder=9,
                            bbox=dict(facecolor='black', alpha=0.6,
                                      pad=1.5, edgecolor='none'))
            except Exception:
                pass

    # Legend on the science panel only
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='+', color='cyan', linestyle='none',
               markersize=8, markeredgewidth=1.5, label='detected'),
        Line2D([0], [0], marker='+', color='red', linestyle='none',
               markersize=8, markeredgewidth=1.5, label='target'),
    ]
    if truth_matches:
        legend_elements.append(
            Line2D([0], [0], marker='D', color='lime', linestyle='none',
                   markersize=6, markeredgewidth=0.8, label='truth'))
    axes[0].legend(handles=legend_elements, fontsize=7,
                   loc='upper left', framealpha=0.7)

    final_model = type(group.model_catalog.get(source_id)).__name__
    fig.suptitle(
        f'Source {source_id}  |  model={final_model}  |  '
        f'group_id={group.group_id}  pop={len(group.source_ids)}  |  '
        f'region x=[{x_lo},{x_hi}) y=[{y_lo},{y_hi})',
        fontsize=9, y=1.01)

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved → {out}')
    plt.show()
    return fig


def track_source(source_id, science_path, weight_path, psf_path,
                 band, zeropoint, eff_gain_path=None, config=None,
                 truth_pos_path=None, truth_radius_arcsec=10.0,
                 truth_flux_path=None, truth_flux_col=None,
                 plot=False, plot_out=None):
    """
    Re-run model selection for *source_id*, print the decision tree trace,
    optionally match to a truth catalog and/or generate a region plot.

    Parameters
    ----------
    source_id            : int
    science_path,
    weight_path,
    psf_path             : str   — FITS file paths
    band                 : str   — e.g. 'F158'
    zeropoint            : float
    eff_gain_path        : str or None — effective gain map from prepare_images_from_cpr
    config               : Config or None
    truth_pos_path       : str or None
        Parquet with columns galaxy_id, ra, dec.
    truth_radius_arcsec  : float  (default 10.0)
    truth_flux_path      : str or None
        Parquet with galaxy_id + flux column.  When provided alongside
        truth_pos_path, observed and true fluxes are printed in nanomaggy.
    truth_flux_col       : str or None
        Flux column name in truth_flux_path (auto-detected if None,
        e.g. 'roman_flux_H158').
    plot                 : bool   — generate 4-panel plot
    plot_out             : str or None — save path for the plot

    Returns
    -------
    dict  with keys: source_id, group_id, group_pop,
                     stages, final_model, tracker, model,
                     truth_matches, obs_flux_nm, true_flux_nm, flux_ratio
    """
    if config is None:
        config = Config()

    obs_to_nm, truth_to_nm = _get_flux_converters(science_path, band)

    img = FarmerImage(
        bands={band: {'science':  science_path,
                      'weight':   weight_path,
                      'eff_gain': eff_gain_path,
                      'psf':      psf_path,
                      'zeropoint': zeropoint}},
        detection_band=band,
        config=config,
    )
    img.detect()

    row = img.catalog[img.catalog['id'] == source_id]
    if len(row) == 0:
        raise ValueError(
            f'Source ID {source_id} not found '
            f'(catalog has {len(img.catalog)} sources, '
            f'IDs {int(img.catalog["id"].min())}–{int(img.catalog["id"].max())})')

    group_id  = int(row['group_id'][0])
    group_pop = int(row['group_pop'][0])
    src_ra    = float(row['ra'][0])
    src_dec   = float(row['dec'][0])

    print(f'Source {source_id}  |  group_id={group_id}  group_pop={group_pop}')
    print(f'Position: x={float(row["x"][0]):.2f}  y={float(row["y"][0]):.2f}  '
          f'ra={src_ra:.6f}  dec={src_dec:.6f}')

    # ── Truth matching ────────────────────────────────────────────────────────
    truth_matches = []
    if truth_pos_path is not None:
        truth_matches = _match_truth(
            src_ra, src_dec, truth_pos_path, truth_radius_arcsec,
            truth_flux_path=truth_flux_path,
            truth_flux_col=truth_flux_col,
            truth_to_nm=truth_to_nm)
        if truth_matches:
            has_flux = not np.isnan(truth_matches[0].get('flux_nm', float('nan')))
            print(f'\nTruth matches within {truth_radius_arcsec}":')
            for tm in truth_matches:
                flux_str = (f'  flux={tm["flux_nm"]:.4f} nMgy'
                            if has_flux and not np.isnan(tm['flux_nm']) else '')
                print(f'  galaxy_id={tm["galaxy_id"]:>8}  '
                      f'ra={tm["ra"]:.6f}  dec={tm["dec"]:.6f}  '
                      f'sep={tm["sep_arcsec"]:.3f}"{flux_str}')
        else:
            print(f'\nNo truth source within {truth_radius_arcsec}" of this source.')

    # ── Group members ─────────────────────────────────────────────────────────
    members = img.catalog[img.catalog['group_id'] == group_id]
    if group_pop > 1:
        print(f'\nGroup members ({group_pop}):')
        for m in members:
            marker = ' ← target' if int(m['id']) == source_id else ''
            print(f'  id={int(m["id"])}  x={float(m["x"]):.1f}  '
                  f'y={float(m["y"]):.1f}{marker}')

    # ── Fit group ─────────────────────────────────────────────────────────────
    group = img._spawn_group(group_id)
    if group.rejected:
        print('\nGroup was REJECTED (e.g. all-zero weight)')
        return None

    group.determine_models()
    # Save model-selection tracker before force_models wipes it via _reset_models
    select_tracker = copy.deepcopy(group.model_tracker)
    group.force_models()

    # Merge: keep model-selection stages (1–6) from determine_models,
    # append forced-phot stages (10–11) from force_models
    tracker = select_tracker.get(source_id, {})
    for s, entry in group.model_tracker.get(source_id, {}).items():
        tracker[s] = entry

    final_model = type(group.model_catalog.get(source_id)).__name__
    stages_done = sorted(k for k in tracker if isinstance(k, int))

    # ── Print trace ───────────────────────────────────────────────────────────
    print(f'\n{"Stage":>5}  {"Model fitted":>22}  {"rchisq":>8}  Decision')
    print('─' * 75)
    stage_records = []
    for s in stages_done:
        if s == 0:
            continue
        model_name = _STAGE_MODEL.get(s, f'stage{s}')
        rchisq     = tracker[s].get('total', {}).get('rchisq', float('nan'))
        decision   = _annotate_decision(s, tracker, config) if s in (2, 4, 5) else ''
        print(f'{s:>5}  {model_name:>22}  {rchisq:>8.4f}  {decision}')
        stage_records.append({'stage': s, 'model': model_name,
                               'rchisq': rchisq, 'decision': decision})

    print('─' * 75)
    print(f'Final model: {final_model}')

    final = group.model_catalog.get(source_id)
    if final is not None and hasattr(final, 'shape'):
        try:
            shape = final.shape
            pnames = shape.getParamNames()
            pvals  = shape.getParams()
            pstr   = '  '.join(f'{n}={float(v):.3f}' for n, v in zip(pnames, pvals))
            print(f'Shape: {pstr}  reff={np.exp(float(shape.logre)):.3f}"')
        except Exception:
            pass

    # ── Flux comparison (nanomaggy) ───────────────────────────────────────────
    obs_flux_nm  = float('nan')
    true_flux_nm = float('nan')
    flux_ratio   = float('nan')

    if final is not None and obs_to_nm is not None:
        try:
            obs_flux_nm = obs_to_nm(final.brightness.getFlux(band))
        except Exception:
            pass

    if truth_matches and truth_to_nm is not None:
        nm = truth_matches[0].get('flux_nm', float('nan'))
        if not np.isnan(nm):
            true_flux_nm = nm

    if not np.isnan(obs_flux_nm) and not np.isnan(true_flux_nm) and true_flux_nm > 0:
        flux_ratio = obs_flux_nm / true_flux_nm

    if not np.isnan(obs_flux_nm) or not np.isnan(true_flux_nm):
        print(f'\nFlux (nMgy):  obs={obs_flux_nm:.5f}  '
              f'true={true_flux_nm:.5f}  ratio={flux_ratio:.4f}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = None
    if plot:
        fig = _plot_track(group, source_id, img, band,
                          truth_matches=truth_matches, out=plot_out)

    return {
        'source_id':     source_id,
        'group_id':      group_id,
        'group_pop':     group_pop,
        'stages':        stage_records,
        'final_model':   final_model,
        'tracker':       tracker,
        'model':         final,
        'truth_matches': truth_matches,
        'obs_flux_nm':   obs_flux_nm,
        'true_flux_nm':  true_flux_nm,
        'flux_ratio':    flux_ratio,
        'fig':           fig,
    }
