"""
slimfarmer — streamlined single/multi-band Tractor photometry.

Keeps the full The Farmer fitting logic (detection → grouping → model selection
→ forced photometry) but removes mosaics, bricks, HDF5 storage, and the
config-file system.

Single-band usage
-----------------
>>> import slimfarmer
>>> cat = slimfarmer.run_photometry(
...     science_path='roman_image.fits',
...     weight_path='roman_weight.fits',
...     psf_path='psf_H158.fits',
...     band='F158',
...     zeropoint=26.5,
...     output_path='catalog.fits',
... )

Multi-band usage
----------------
>>> cat = slimfarmer.run_photometry(
...     bands={
...         'F158': {'science': 'F158.fits', 'weight': 'F158_wht.fits',
...                  'psf': 'F158_psf.fits', 'zeropoint': 26.5},
...         'F106': {'science': 'F106.fits', 'psf': 'F106_psf.fits', 'zeropoint': 26.3},
...     },
...     detection_band='F158',
... )
"""

from .config import Config
from .image import FarmerImage, run_photometry
from .utils import SimpleGalaxy, prepare_images_from_cpr, prepare_stitched_block
from .track import track_source, _get_flux_converters
from .diag import diagnose_source
from .forced import (forced_photometry, reconstruct_source, load_psf,
                     find_rubin_coadds_for_tile, find_rubin_coadds,
                     rubin_coadd_wcs, roman_tile_footprint,
                     load_skymap_info, source_in_exclusive_region,
                     patch_inner_bbox_local, nearest_tract)

__version__ = '1.1.0'
__all__ = ['Config', 'FarmerImage', 'run_photometry', 'SimpleGalaxy',
           'prepare_images_from_cpr', 'prepare_stitched_block',
           'track_source', '_get_flux_converters',
           'diagnose_source',
           'forced_photometry', 'reconstruct_source', 'load_psf']
