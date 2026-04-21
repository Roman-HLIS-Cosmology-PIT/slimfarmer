"""Bit-flag values for the catalog ``flag`` column.

The low byte (0x00FF) is populated by SEP during detection; slimfarmer does
not set those bits itself but they survive into the output catalog. SEP
constants are listed here for reference — see
https://sep.readthedocs.io/en/latest/api/sep.extract.html.

The high bytes are populated by slimfarmer at later pipeline stages.
"""

# ── SEP detection flags (low byte, set by sep.extract) ────────────────────
SEP_OBJ_MERGED       = 0x0001
SEP_OBJ_TRUNC        = 0x0002
SEP_OBJ_DOVERFLOW    = 0x0004
SEP_OBJ_SINGU        = 0x0008
SEP_APER_TRUNC       = 0x0010
SEP_APER_HASMASKED   = 0x0020
SEP_APER_ALLMASKED   = 0x0040
SEP_APER_NONPOSITIVE = 0x0080

# ── Slimfarmer pipeline flags (high bytes) ────────────────────────────────
FLAG_BOUNDARY           = 0x0100  # near image boundary or IMCOM overlap skirt
FLAG_TIMEOUT            = 0x0200  # group hit timeout (soft deadline or hard ceiling)
FLAG_SINGLETON_FALLBACK = 0x0400  # re-fit as a singleton after group timeout
