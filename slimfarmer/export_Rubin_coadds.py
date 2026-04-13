#!/usr/bin/env python
"""
Export LSST coadd images and PSF kernels for all tract/patch pairs
overlapping the Roman mosaic footprint.

Output structure:
  <output_dir>/
    <band>/
      coadd_<tract>_<patch>.fits    (image + variance + mask as multi-HDU)
      psf_<tract>_<patch>.fits      (PSF kernel image)

Usage:
    python export_roman_coadds.py [--output-dir OUTPUT_DIR] [--image-only] [--bands u,g,r,i,z,y]
"""
import argparse
import os
import time
import numpy as np

import lsst.daf.butler as dafButler
import lsst.geom as geom
from astropy.io import fits


def find_overlapping_patches(skymap, ra_center, dec_center, half_size_deg):
    """Find all tract/patch pairs overlapping the mosaic footprint."""
    cos_dec = np.cos(np.radians(dec_center))
    grid_spacing_deg = 5.0 / 60.0  # 5 arcmin

    ra_grid = np.arange(
        ra_center - half_size_deg / cos_dec,
        ra_center + half_size_deg / cos_dec + grid_spacing_deg / cos_dec,
        grid_spacing_deg / cos_dec,
    )
    dec_grid = np.arange(
        dec_center - half_size_deg,
        dec_center + half_size_deg + grid_spacing_deg,
        grid_spacing_deg,
    )

    tract_patch_set = set()
    for ra in ra_grid:
        for dec in dec_grid:
            point = geom.SpherePoint(ra, dec, geom.degrees)
            for tract_info in skymap.findAllTracts(point):
                tract_id = tract_info.getId()
                patch_info = tract_info.findPatch(point)
                patch_idx = patch_info.getSequentialIndex()
                tract_patch_set.add((tract_id, patch_idx))

    return sorted(tract_patch_set)


def export_coadd(butler, ref, output_dir, band, image_only=False):
    """Export a single coadd image and PSF kernel to FITS."""
    tract_id = ref.dataId['tract']
    patch_idx = ref.dataId['patch']

    band_dir = os.path.join(output_dir, band)
    os.makedirs(band_dir, exist_ok=True)

    coadd_path = os.path.join(band_dir, f'coadd_{tract_id}_{patch_idx}.fits')
    psf_path = os.path.join(band_dir, f'psf_{tract_id}_{patch_idx}.fits')

    # Skip if already exported
    if os.path.exists(coadd_path) and os.path.exists(psf_path):
        return 'skipped'

    coadd = butler.get(ref)

    # Build FITS HDU list
    hdu_list = [fits.PrimaryHDU()]

    # Image HDU
    img_hdu = fits.ImageHDU(coadd.image.array, name='IMAGE')
    img_hdu.header['TRACT'] = tract_id
    img_hdu.header['PATCH'] = patch_idx
    img_hdu.header['BAND'] = band

    # Add WCS from the LSST SkyWcs
    wcs_meta = coadd.getWcs().getFitsMetadata()
    for key in wcs_meta.names():
        try:
            img_hdu.header[key] = wcs_meta.getScalar(key)
        except Exception:
            pass

    # Add bbox info for coordinate reconstruction
    bbox = coadd.getBBox()
    img_hdu.header['LTV1'] = -bbox.getMinX()
    img_hdu.header['LTV2'] = -bbox.getMinY()
    img_hdu.header['CRVAL1A'] = bbox.getMinX()
    img_hdu.header['CRVAL2A'] = bbox.getMinY()
    hdu_list.append(img_hdu)

    if not image_only:
        # Variance HDU
        var_hdu = fits.ImageHDU(coadd.variance.array, name='VARIANCE')
        hdu_list.append(var_hdu)

        # Mask HDU
        mask_hdu = fits.ImageHDU(coadd.mask.array, name='MASK')
        hdu_list.append(mask_hdu)

    hdul = fits.HDUList(hdu_list)
    hdul.writeto(coadd_path, overwrite=True)

    # PSF kernel at patch center
    psf = coadd.getPsf()
    center_point = geom.Point2D(
        (bbox.getMinX() + bbox.getMaxX()) / 2.0,
        (bbox.getMinY() + bbox.getMaxY()) / 2.0,
    )
    psf_kernel = psf.computeKernelImage(center_point)

    # Also compute and store FWHM
    sigma = psf.computeShape(center_point).getDeterminantRadius()
    SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
    pixel_scale = 0.19976576462766338  # arcsec/pixel
    fwhm_arcsec = sigma * SIGMA_TO_FWHM * pixel_scale

    psf_hdu = fits.PrimaryHDU(psf_kernel.array)
    psf_hdu.header['TRACT'] = tract_id
    psf_hdu.header['PATCH'] = patch_idx
    psf_hdu.header['BAND'] = band
    psf_hdu.header['SIGMA'] = (sigma, 'PSF sigma in pixels')
    psf_hdu.header['FWHMPIX'] = (sigma * SIGMA_TO_FWHM, 'PSF FWHM in pixels')
    psf_hdu.header['FWHMARC'] = (fwhm_arcsec, 'PSF FWHM in arcsec')
    psf_hdu.header['PIXSCALE'] = (pixel_scale, 'Pixel scale in arcsec/pixel')
    fits.HDUList([psf_hdu]).writeto(psf_path, overwrite=True)

    return 'exported'


def main():
    parser = argparse.ArgumentParser(description='Export LSST coadds overlapping Roman mosaic')
    parser.add_argument('--output-dir', default=os.path.expanduser('~/code/LSSTRoman/coadds'),
                        help='Output directory (default: ~/code/LSSTRoman/coadds)')
    parser.add_argument('--image-only', action='store_true',
                        help='Export image only (skip variance and mask planes)')
    parser.add_argument('--bands', default='u,g,r,i,z,y',
                        help='Comma-separated list of bands (default: u,g,r,i,z,y)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just print what would be exported and estimate storage')
    args = parser.parse_args()

    bands = args.bands.split(',')

    # Roman mosaic 1 parameters
    ra_center = 9.55
    dec_center = -44.1
    mosaic_size = 5.0 / 3.0 * 40  # arcmin
    half_size_deg = mosaic_size / 2.0 / 60.0

    print(f'Roman mosaic: center=({ra_center}, {dec_center}), '
          f'size={mosaic_size:.1f} arcmin')
    print(f'Bands: {bands}')
    print(f'Output: {args.output_dir}')
    print(f'Image only: {args.image_only}')
    print()

    # Connect to butler
    repo = '/global/cfs/cdirs/lsst/production/gen3/roman-desc-sims/repo'
    butler = dafButler.Butler(repo)
    skymap = butler.get('skyMap', skymap='DC2_cells_v1', collections='skymaps')

    # Find overlapping patches
    patches = find_overlapping_patches(skymap, ra_center, dec_center, half_size_deg)
    print(f'Found {len(patches)} tract/patch pairs')

    # Build collections list
    collections = [f'u/descdm/step3b_wfd_{i:03d}_w_2024_22' for i in range(67)]

    # Query all refs
    all_refs = {}
    for band in bands:
        band_refs = []
        for tract_id, patch_idx in patches:
            refs = list(butler.registry.queryDatasets(
                'deepCoadd_calexp',
                collections=collections,
                dataId={'band': band, 'skymap': 'DC2_cells_v1',
                        'tract': tract_id, 'patch': patch_idx},
            ))
            band_refs.extend(refs)
        all_refs[band] = band_refs
        print(f'  Band {band}: {len(band_refs)} datasets')

    total_refs = sum(len(v) for v in all_refs.values())
    print(f'\nTotal: {total_refs} coadds to export')

    # Storage estimate
    # Image: 3400x3400 float32 = 46.2 MB, Variance: same, Mask: 3400x3400 int32 = 46.2 MB
    # PSF: 35x35 float64 = 9.8 KB
    img_size = 3400 * 3400 * 4  # float32
    var_size = 3400 * 3400 * 4  # float32
    mask_size = 3400 * 3400 * 4  # int32
    psf_size = 35 * 35 * 8  # float64

    per_coadd = img_size + psf_size
    if not args.image_only:
        per_coadd += var_size + mask_size

    total_bytes = total_refs * per_coadd
    print(f'Estimated storage: {total_bytes / 1e9:.1f} GB')
    if args.image_only:
        print(f'  ({total_refs} x {img_size/1e6:.1f} MB image + PSF)')
    else:
        print(f'  ({total_refs} x {(img_size+var_size+mask_size)/1e6:.1f} MB image+var+mask + PSF)')

    if args.dry_run:
        print('\nDry run complete. Add --no flag to actually export.')
        return

    # Export
    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()
    n_exported = 0
    n_skipped = 0

    for band in bands:
        for i, ref in enumerate(all_refs[band]):
            tract_id = ref.dataId['tract']
            patch_idx = ref.dataId['patch']

            status = export_coadd(butler, ref, args.output_dir, band,
                                  image_only=args.image_only)
            if status == 'exported':
                n_exported += 1
            else:
                n_skipped += 1

            elapsed = time.time() - t_start
            total_done = n_exported + n_skipped
            rate = total_done / elapsed if elapsed > 0 else 0
            remaining = (total_refs - total_done) / rate if rate > 0 else 0
            print(f'\r  [{band}] {i+1}/{len(all_refs[band])} '
                  f'tract={tract_id} patch={patch_idx} ({status}) '
                  f'[{total_done}/{total_refs}, ~{remaining:.0f}s remaining]',
                  end='', flush=True)
        print()

    elapsed = time.time() - t_start
    print(f'\nDone: exported {n_exported}, skipped {n_skipped} '
          f'in {elapsed:.0f}s ({elapsed/60:.1f} min)')

    # Print actual disk usage
    total_size = 0
    for root, dirs, files in os.walk(args.output_dir):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    print(f'Actual disk usage: {total_size / 1e9:.1f} GB')


if __name__ == '__main__':
    main()
