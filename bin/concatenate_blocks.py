#!/usr/bin/env python
"""Concatenate per-block catalog.fits and forced_phot_rubin_*.fits into a
single survey catalog, dropping sources whose ``flag`` field has any bit in
the ``--flag_bit`` mask set (default ``0x0300``).

Flag bits
---------
- ``0x0100`` — IMCOM overlap skirt (34-px border). Each flagged source has
  an un-flagged duplicate in the neighboring block's main region, so
  dropping it is lossless deduplication.
- ``0x0200`` — Group hit the per-group timeout deadline during fitting.
  The source has a valid best-so-far model (from the last completed
  model-selection stage) but may not have gone through full model selection
  or forced photometry. Dropped by default; pass ``--flag_bit 0x0100`` to
  keep them.

Caveat: at the survey boundary the outermost 34-px skirt has no neighbor to
provide an un-flagged copy, so those rows are simply dropped. They are
typically IMCOM padding artifacts.

Forced photometry columns (from forced_phot_rubin_{block}.fits) are joined
onto the Roman catalog via a left join on ``id``, so sources without a Rubin
measurement get masked/NaN Rubin columns. If the forced-phot file doesn't
exist for a block, that block still contributes its Roman-only columns.

If a Single-Object-Fitting variant ``forced_phot_rubin_{block}_sof.fits``
also exists, its band-level columns are joined in with a ``_sof`` suffix
(e.g. ``r_flux`` → ``r_flux_sof``) so MOF and SOF fluxes coexist in the
final survey catalog.

Usage
-----
    python concatenate_blocks.py \\
        --work_base /home/chto/code/Roman/data/openUV24/output_slimfarmer_multi_realization \\
        --out       /home/chto/code/Roman/data/openUV24/catalog_combined.fits
"""
import argparse
import os
import re
import sys

import numpy as np
from astropy.table import Table, vstack, join

BLOCK_RE = re.compile(r'^\d{2}_\d{2}$')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--work_base', required=True,
                   help='Directory containing per-block subdirs (each with catalog.fits)')
    p.add_argument('--out', required=True,
                   help='Output FITS path for the combined catalog')
    p.add_argument('--flag_bit', type=lambda s: int(s, 0), default=0x0300,
                   help='Flag bitmask: drop sources with any of these bits set. '
                        'Default 0x0300 = 0x0100 (overlap skirt) | 0x0200 (timeout)')
    p.add_argument('--catalog_name', default='catalog.fits',
                   help='Per-block catalog filename (default: catalog.fits)')
    return p.parse_args()


def discover_blocks(work_base, catalog_name):
    blocks = []
    for entry in sorted(os.listdir(work_base)):
        if not BLOCK_RE.match(entry):
            continue
        cat_path = os.path.join(work_base, entry, catalog_name)
        if os.path.isfile(cat_path):
            blocks.append((entry, cat_path))
    return blocks


def _join_forced(t, forced_path, suffix=''):
    """Join forced-photometry columns from ``forced_path`` onto ``t``.

    ``suffix`` is appended to each band-level column name ('' for MOF,
    '_sof' for SOF), so the two sets can coexist in the final catalog.
    Returns the (possibly-joined) table and a bool indicating success.
    """
    try:
        fp = Table.read(forced_path)
    except Exception as exc:
        print(f'  WARN: could not read {forced_path}: {exc}', file=sys.stderr)
        return t, False
    fp.meta.clear()
    # Columns already in the base catalog are skipped (id, ra, dec, name).
    # For SOF we check the suffixed name, so `r_flux` -> `r_flux_sof` is kept
    # even though `r_flux` (from MOF) is already in the table.
    skip = set(t.colnames)
    join_cols = ['id']
    for c in list(fp.colnames):
        if c == 'id':
            continue
        new_name = f'{c}{suffix}'
        if new_name in skip:
            continue
        if new_name == c:
            join_cols.append(c)
        else:
            fp.rename_column(c, new_name)
            join_cols.append(new_name)
    if len(join_cols) <= 1:
        return t, False
    return join(t, fp[join_cols], keys='id', join_type='left'), True


def load_all(blocks, work_base, flag_bit):
    parts = []
    n_total = 0
    n_dropped = 0
    n_forced = 0
    n_forced_sof = 0
    for block_id, cat_path in blocks:
        try:
            t = Table.read(cat_path)
        except Exception as exc:
            print(f'  WARN: could not read {cat_path}: {exc}', file=sys.stderr)
            continue
        t.meta.clear()
        n_total += len(t)

        flagged = (np.asarray(t['flag']) & flag_bit) != 0
        n_dropped += int(flagged.sum())
        t = t[~flagged]

        # MOF forced-phot columns (default names)
        mof_path = os.path.join(
            work_base, block_id, f'forced_phot_rubin_{block_id}.fits')
        if os.path.isfile(mof_path):
            t, ok = _join_forced(t, mof_path, suffix='')
            if ok:
                n_forced += 1

        # SOF forced-phot columns (band-level columns get a _sof suffix)
        sof_path = os.path.join(
            work_base, block_id, f'forced_phot_rubin_{block_id}_sof.fits')
        if os.path.isfile(sof_path):
            t, ok = _join_forced(t, sof_path, suffix='_sof')
            if ok:
                n_forced_sof += 1

        if 'block_id' not in t.colnames:
            t['block_id'] = np.full(len(t), block_id, dtype='U6')
        parts.append(t)

    if not parts:
        raise RuntimeError('No block catalogs loaded.')
    print(f'  loaded {len(parts)} block catalogs '
          f'({n_forced} with Rubin MOF, {n_forced_sof} with Rubin SOF)')
    print(f'  total rows read              : {n_total}')
    print(f'  dropped (flag & 0x{flag_bit:04x}) : {n_dropped}')
    print(f'  total rows kept              : {n_total - n_dropped}')
    return vstack(parts, join_type='outer')


def main():
    args = parse_args()

    print(f'Scanning {args.work_base}')
    blocks = discover_blocks(args.work_base, args.catalog_name)
    if not blocks:
        sys.exit(f'No per-block {args.catalog_name} files found in {args.work_base}')
    print(f'  found {len(blocks)} block catalogs')

    print('Loading and dropping flagged rows...')
    cat = load_all(blocks, args.work_base, args.flag_bit)

    print(f'Writing {args.out}  ({len(cat)} rows, {len(cat.colnames)} columns)')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cat.write(args.out, overwrite=True)
    print('Done.')


if __name__ == '__main__':
    main()
