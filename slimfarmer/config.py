"""Pipeline configuration for slimfarmer."""

import astropy.units as u


class Config:
    """All pipeline parameters with sensible defaults for Roman H158."""

    # Detection (SEP)
    thresh = 3.0
    minarea = 5
    back_bw = 32;  back_bh = 32
    back_fw = 2;  back_fh = 2
    filter_kernel = 'gauss_3.0_7x7.conv'
    filter_type = 'matched'
    deblend_nthresh =  2**5
    deblend_cont = 0.001
    clean = True
    clean_param = 1.0
    pixstack_size = 20_000_000
    use_detection_weight = True
    # Grouping. Set to None to disable grouping entirely (one source per group,
    # even if their SEP segmaps touch). Set to 0*u.arcsec to label by raw
    # segmap adjacency without dilation.
    dilation_radius = 0.2 * u.arcsec
    group_buffer = 0.01 * u.arcsec
    group_size_limit = 100
    fit_dilation_radius = 0.2 * u.arcsec  # expand fitting region beyond groupmap to capture profile wings
    timeout=1000
    # Hard wall-clock ceiling per task (seconds). When a worker hangs inside a
    # C extension and ignores the cooperative `timeout` deadline, the parent
    # gives up on that task after this many seconds and moves on.
    stuck_ceiling = 600
    paddingpixel = 34
    # When a group hits the timeout, retry each source as an independent
    # single-source fit. Disable to keep the old behavior of accepting the
    # partial joint-fit result from the last completed stage.
    singleton_fallback = True
    # Dominant-source prefit: if a group contains one source whose SEP segmap
    # is much larger than any sibling's, solo-fit that source first, then run
    # the joint fit with the dominant source's parameters frozen. This prevents
    # neighbors from stealing the dominant source's wing flux during joint
    # optimization (a common failure on bright extended galaxies).
    dominant_prefit = True
    dominant_npix_ratio = 3.0  # dominant.npix must exceed ratio × 2nd-largest.npix
    dominant_npix_min = 500    # absolute minimum npix for "dominant" qualification

    # Background
    subtract_background = False
    backtype = 'variable'

    # Tractor optimisation
    max_steps = 300
    damping = 1E-3
    dlnp_crit = 1e-10
    ignore_failures = True
    renorm_psf = 1.0       # normalise PSF stamp to this value; 1=unbiased

    # Model selection
    modeling_bands = None  # list of bands used for model-type selection; None = detection band only
    model_priors = {'pos': 0.1*u.arcsec, 'reff': 0.5*u.arcsec, 'shape': 'none', 'fracDev': 'none'}
    phot_priors  = {'pos': 0.001*u.arcsec, 'reff': 'freeze', 'shape': 'freeze', 'fracDev': 'freeze'}
    fixed_reff   = None    # astropy Quantity (arcsec) or None; when set, forces logre = log(value)
                           # and freezes it in all stages, overriding model_priors['reff']
    noshot=False
    sufficient_thresh      = 0.3    # rchisq < 1 means model is good enough
    simplegalaxy_penalty   = 0.1    # PS must beat SG by this margin to be preferred (blob.py PS_SG_THRESH1)
    exp_dev_similar_thresh = 0.1    # |exp_rchisq - dev_rchisq| < this → try Composite (blob.py EXP_DEV_THRESH)
    chisq_force_exp_dev    = 0.15   # if both PS and SG exceed this, force Exp/Dev regardless (blob.py CHISQ_FORCE_EXP_DEV)
    chisq_force_comp       = 0.15    # if both Exp and Dev exceed this, force Composite (blob.py CHISQ_FORCE_COMP)

    # Neighbor subtraction (second forced-photometry pass)
    neighbor_subtraction = False     # subtract neighboring group models before forced phot
    neighbor_radius = 5.0 * u.arcsec  # include sources within this radius as neighbors

    # 3x3 stitching (boundary handling for IMCOM blocks)
    buffer_arcsec = 6.0   # width of neighbor strip to include around the central block
                          # (0 disables stitching → fall back to per-block processing)
    block_size_px = 2108
    # Note: the IMCOM overlap-per-side is `paddingpixel` (defined above).
    # block_overlap_px is intentionally not a separate field — they must agree.

    # Canonical IMCOM Roman pixel scale (arcsec/pixel). Used when overlaying
    # Roman-derived shape parameters (a, b in Roman pixels) on images with
    # a different pixel scale, e.g. Rubin forced-photometry diagnostics.
    roman_pixel_scale_arcsec = 0.049019607843138

    # Output
    save_model_image = True  # save <output>_model.fits and <output>_residual.fits after run_photometry

    # Parallelism
    ncpus = 0          # 0 = serial; >0 = multiprocessing

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f'Unknown config key: {k}')
            setattr(self, k, v)
