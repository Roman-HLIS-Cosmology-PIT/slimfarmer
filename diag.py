"""Source-level photometry diagnostics."""

import copy
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def diagnose_source(source_id, img, truth, obs_to_nm, truth_to_nm, oversamplepix,
                    band='F158', truth_flux_col='roman_flux_H158', plot=True):
    """
    Full photometry diagnostic for one detected source.

    Parameters
    ----------
    source_id     : int   — catalog id
    img           : FarmerImage (already img.detect()-ed)
    truth         : pd.DataFrame with ra, dec, truth_flux_col columns
    obs_to_nm     : callable, IMCOM DN → nMgy
    truth_to_nm   : callable, roman internal flux → nMgy
    oversamplepix : float, arcsec/px of the science image
    band          : str
    truth_flux_col: column name for truth flux
    plot          : bool

    Returns
    -------
    dict with keys: obs_nm, true_nm, ratio, rchisq, reff, model_name, fig
    """
    mask = img.catalog['id'] == source_id
    if not np.any(mask):
        raise ValueError(f'source {source_id} not in catalog')
    row = img.catalog[mask][0]
    cx, cy = float(row['x']), float(row['y'])

    print(f'=== Source {source_id} ===')
    print(f'  x={cx:.2f}  y={cy:.2f}  '
          f'ra={float(row["ra"]):.7f}  dec={float(row["dec"]):.7f}')
    print(f'  SEP a={float(row["a"])*oversamplepix:.3f}"  '
          f'b={float(row["b"])*oversamplepix:.3f}"  '
          f'group_id={int(row["group_id"])}  pop={int(row["group_pop"])}')

    sci = img.band_config[band]['science']
    wht = img.band_config[band]['weight']
    seg_y, seg_x = img.segmap_dict.get(source_id, (np.array([]), np.array([])))
    n_segpix  = len(seg_y)
    seg_sum   = float(np.sum(sci[seg_y, seg_x])) if n_segpix else 0.0
    print(f'  segmap pixels={n_segpix}  '
          f'segmap sum={seg_sum:.2f} DN → {obs_to_nm(seg_sum):.5f} nMgy')

    yy, xx = np.mgrid[0:sci.shape[0], 0:sci.shape[1]]
    dist_full = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    print(f'\n  Circular aperture sums:')
    ap_results = {}
    for r_as in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
        r_px = r_as / oversamplepix
        s = float(np.sum(sci[dist_full <= r_px]))
        ap_results[r_as] = s
        print(f'    r={r_as:.1f}" ({r_px:.1f}px): {s:.2f} DN → {obs_to_nm(s):.5f} nMgy')

    tru_c = SkyCoord(truth['ra'].values, truth['dec'].values, unit='deg')
    src_c = SkyCoord(float(row['ra']), float(row['dec']), unit='deg')
    seps  = src_c.separation(tru_c).to(u.arcsec).value
    nearby = np.argsort(seps)[:6]

    print(f'\n  Truth galaxies within 5":')
    for ni in nearby:
        if seps[ni] > 5.0:
            break
        f_nm = truth_to_nm(float(truth[truth_flux_col].iloc[ni]))
        cols = truth.columns.tolist()
        dhlr = float(truth['diskHalfLightRadiusArcsec'].iloc[ni]) if 'diskHalfLightRadiusArcsec' in cols else float('nan')
        print(f'    sep={seps[ni]:.3f}"  gal_id={truth["galaxy_id"].iloc[ni]}  '
              f'flux={f_nm:.5f} nMgy  dhlr={dhlr:.3f}"')

    true_nm = truth_to_nm(float(truth[truth_flux_col].iloc[nearby[0]]))

    gid   = int(row['group_id'])
    group = img._spawn_group(gid)
    group.determine_models()
    group.force_models()
    final = group.model_catalog.get(source_id)

    obs_nm = float('nan'); rchisq = float('nan'); reff = float('nan')
    model_name = 'None'
    if final is not None:
        obs_nm = obs_to_nm(float(final.brightness.getFlux(band)))
        model_name = type(final).__name__
        if hasattr(final, 'shape'):
            try:
                reff = 10 ** float(final.shape.logre)
            except Exception:
                pass
        t = group.model_tracker.get(source_id, {}).get(11, {})
        rchisq = t.get('total', {}).get('rchisq', float('nan'))
        print(f'\n  Tractor model: {model_name}')
        print(f'  fitted flux:   {obs_nm:.5f} nMgy')
        if not np.isnan(reff):
            print(f'  reff:          {reff:.4f}"')
        print(f'  rchisq:        {rchisq:.4f}')

    ratio = obs_nm / true_nm if (not np.isnan(obs_nm) and true_nm > 0) else float('nan')
    print(f'\n  Truth flux:    {true_nm:.5f} nMgy')
    print(f'  obs/true:      {ratio:.4f}')

    # Build full-image model and chi arrays from the group cutout
    model_full = np.zeros_like(sci)
    chi_full   = np.zeros_like(sci)
    if group.engine is not None and group.images:
        try:
            model_local = group.engine.getModelImage(0)
            first_img   = list(group.images.values())[0]
            chi_local   = ((first_img.data - model_local)
                           * np.sqrt(np.maximum(first_img.invvar, 0)))
            gox = int(group.origin[0])
            goy = int(group.origin[1])
            lh, lw = model_local.shape
            fy0, fy1 = goy, goy + lh
            fx0, fx1 = gox, gox + lw
            ly0 = max(0, -fy0); fy0 = max(0, fy0)
            lx0 = max(0, -fx0); fx0 = max(0, fx0)
            fy1 = min(sci.shape[0], fy1); ly1 = ly0 + (fy1 - fy0)
            fx1 = min(sci.shape[1], fx1); lx1 = lx0 + (fx1 - fx0)
            model_full[fy0:fy1, fx0:fx1] = model_local[ly0:ly1, lx0:lx1]
            chi_full[fy0:fy1, fx0:fx1]   = chi_local[ly0:ly1, lx0:lx1]
        except Exception:
            pass

    fig = None
    if plot:
        fig = _plot_source(source_id, img, sci, wht, cx, cy, dist_full,
                           seg_y, seg_x, row, seps, nearby, truth,
                           obs_to_nm, oversamplepix, final,
                           model_full, chi_full,
                           obs_nm, true_nm, ratio, band)

    return dict(obs_nm=obs_nm, true_nm=true_nm, ratio=ratio,
                rchisq=rchisq, reff=reff, model_name=model_name, fig=fig)


def _plot_source(source_id, img, sci, wht, cx, cy, dist_full,
                 seg_y, seg_x, row, seps, nearby, truth,
                 obs_to_nm, oversamplepix, final,
                 model_full, chi_full,
                 obs_nm, true_nm, ratio, band):
    buf = 60
    ix, iy = int(round(cx)), int(round(cy))
    y0, y1 = max(0, iy - buf), min(sci.shape[0], iy + buf)
    x0, x1 = max(0, ix - buf), min(sci.shape[1], ix + buf)
    sci_cut   = sci[y0:y1, x0:x1]
    wht_cut   = wht[y0:y1, x0:x1]
    model_cut = model_full[y0:y1, x0:x1]
    chi_cut   = chi_full[y0:y1, x0:x1]

    seg_mask = np.zeros(sci.shape, dtype=bool)
    if len(seg_y):
        seg_mask[seg_y, seg_x] = True
    seg_cut = seg_mask[y0:y1, x0:x1]

    def anorm(d):
        m = np.isfinite(d) & (d != 0)
        if not m.any():
            return d
        med = np.median(d[m])
        vm  = np.percentile(np.abs(d[m] - med), 99.5)
        return np.arcsinh((d - med) / max(vm * 0.5, 1e-30))

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    ext = [x0, x1, y0, y1]

    # Panel 0: Science
    axes[0].imshow(anorm(sci_cut), origin='lower', cmap='gray_r', extent=ext, aspect='equal')
    axes[0].contour(seg_cut, levels=[0.5], colors=['cyan'], linewidths=0.8,
                    extent=ext, origin='lower')
    axes[0].scatter([cx], [cy], marker='+', c='red', s=150, linewidths=1.5, zorder=5)
    for r_as in [0.3, 0.7, 1.0]:
        axes[0].add_patch(plt.Circle((cx, cy), r_as / oversamplepix,
                                     color='orange', fill=False, lw=0.8, ls='--'))
    for ni in nearby:
        if seps[ni] > 4.0:
            break
        try:
            tx, ty = img.wcs.all_world2pix(
                float(truth['ra'].iloc[ni]), float(truth['dec'].iloc[ni]), 0)
            axes[0].scatter([tx], [ty], marker='D', s=60, c='lime',
                            edgecolors='k', linewidths=0.6, zorder=6)
            axes[0].text(tx + 2, ty + 2, f'{seps[ni]:.2f}"',
                         color='lime', fontsize=7, fontweight='bold',
                         bbox=dict(facecolor='k', alpha=0.5, pad=1, edgecolor='none'))
        except Exception:
            pass
    axes[0].set_title('Science  (cyan=segmap)', fontsize=10)

    # Panel 1: Weight map (actual values)
    wv = wht_cut.copy()
    wv[wv <= 0] = np.nan
    wmed = np.nanmedian(wv) if np.any(np.isfinite(wv)) else 1.0
    im1 = axes[1].imshow(wv, origin='lower', cmap='viridis', extent=ext, aspect='equal',
                         norm=mcolors.Normalize(vmin=np.nanmin(wv), vmax=np.nanmax(wv)))
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(f'Weight  (med={wmed:.0f})', fontsize=10)

    # Panel 2: Model
    axes[2].imshow(anorm(model_cut), origin='lower', cmap='gray_r', extent=ext, aspect='equal')
    axes[2].contour(seg_cut, levels=[0.5], colors=['cyan'], linewidths=0.8,
                    extent=ext, origin='lower')
    axes[2].scatter([cx], [cy], marker='+', c='red', s=150, linewidths=1.5, zorder=5)
    model_label = type(final).__name__ if final is not None else '?'
    axes[2].set_title(f'Model  ({model_label})', fontsize=10)

    # Panel 3: Chi  (data - model) / noise
    chi_lim = max(np.abs(chi_cut).max(), 1e-3)
    im3 = axes[3].imshow(chi_cut, origin='lower', cmap='RdBu_r', extent=ext, aspect='equal',
                         vmin=-5, vmax=5)
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].contour(seg_cut, levels=[0.5], colors=['cyan'], linewidths=0.8,
                    extent=ext, origin='lower')
    axes[3].scatter([cx], [cy], marker='+', c='lime', s=150, linewidths=1.5, zorder=5)
    axes[3].set_title('Chi  (data−model)/σ', fontsize=10)

    # Panel 4: Radial profile
    r_px_bins = np.arange(0, 80, 1.0)
    profile = []
    for r in r_px_bins:
        ann = (dist_full >= r) & (dist_full < r + 1) & (wht > 0)
        profile.append(float(np.mean(sci[ann])) if ann.any() else 0.)
    axes[4].semilogy(r_px_bins * oversamplepix, np.abs(profile) + 1e-6,
                     'k-', lw=1.2, label='data SB')
    if model_full.any():
        mprofile = []
        for r in r_px_bins:
            ann = (dist_full >= r) & (dist_full < r + 1) & (wht > 0)
            mprofile.append(float(np.mean(model_full[ann])) if ann.any() else 0.)
        axes[4].semilogy(r_px_bins * oversamplepix, np.abs(mprofile) + 1e-6,
                         'b--', lw=1.2, label='model SB')
    axes[4].axvline(x=float(row['a']) * oversamplepix, color='cyan', ls='--', lw=1,
                    label=f'SEP a={float(row["a"])*oversamplepix:.2f}"')
    if final is not None and hasattr(final, 'shape'):
        try:
            re = 10 ** float(final.shape.logre)
            axes[4].axvline(x=re, color='red', ls='--', lw=1, label=f'reff={re:.3f}"')
        except Exception:
            pass
    axes[4].set_xlabel('radius (arcsec)')
    axes[4].set_ylabel('mean SB (DN/px)')
    axes[4].set_title('Radial profile', fontsize=10)
    axes[4].legend(fontsize=8)

    fig.suptitle(f'Source {source_id} | {model_label} | obs/true={ratio:.3f}', fontsize=10)
    fig.tight_layout()
    return fig
