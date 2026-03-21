"""Pipeline configuration for slimfarmer."""

import astropy.units as u


class Config:
    """All pipeline parameters with sensible defaults for Roman H158."""

    # Detection (SEP)
    thresh = 5.0
    minarea = 8
    back_bw = 32;  back_bh = 32
    back_fw = 2;  back_fh = 2
    filter_kernel = 'gauss_2.0_5x5.conv'
    filter_type = 'matched'
    deblend_nthresh =  2**8
    deblend_cont = 1E-10
    clean = True
    clean_param = 1.0
    pixstack_size = 1_000_000
    use_detection_weight = True

    # Grouping
    dilation_radius = 0.2 * u.arcsec
    group_buffer = 2.0 * u.arcsec
    group_size_limit = 10
    fit_dilation_radius = 0.2 * u.arcsec  # expand fitting region beyond groupmap to capture profile wings

    # Background
    subtract_background = False
    backtype = 'variable'

    # Tractor optimisation
    max_steps = 100
    damping = 1E-6
    dlnp_crit = 1e-3
    ignore_failures = True
    renorm_psf = 1.0       # normalise PSF stamp to this value; 1=unbiased

    # Model selection
    model_priors = {'pos': 0.10 * u.arcsec, 'reff': 'none', 'shape': 'none', 'fracDev': 'none'}
    phot_priors  = {'pos': 0.001  * u.arcsec, 'reff': 'freeze', 'shape': 'freeze', 'fracDev': 'freeze'}
    sufficient_thresh      = 0.3    # rchisq < 1 means model is good enough
    simplegalaxy_penalty   = 0.1    # PS must beat SG by this margin to be preferred (blob.py PS_SG_THRESH1)
    exp_dev_similar_thresh = 0.1    # |exp_rchisq - dev_rchisq| < this → try Composite (blob.py EXP_DEV_THRESH)
    chisq_force_exp_dev    = 0.15   # if both PS and SG exceed this, force Exp/Dev regardless (blob.py CHISQ_FORCE_EXP_DEV)
    chisq_force_comp       = 0.15    # if both Exp and Dev exceed this, force Composite (blob.py CHISQ_FORCE_COMP)

    # Neighbor subtraction (second forced-photometry pass)
    neighbor_subtraction = False     # subtract neighboring group models before forced phot
    neighbor_radius = 5.0 * u.arcsec  # include sources within this radius as neighbors

    # Parallelism
    ncpus = 0          # 0 = serial; >0 = multiprocessing

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f'Unknown config key: {k}')
            setattr(self, k, v)
