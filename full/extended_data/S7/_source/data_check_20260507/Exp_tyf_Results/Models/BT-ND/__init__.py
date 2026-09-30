__all__ = []


try:
    from .eeg_meg_prior import EEGMEGPriorEncoder
    __all__.append("EEGMEGPriorEncoder")
except ImportError:
    pass
try:
    from .fmri_prior import FMRIPriorEncoder
    __all__.append("FMRIPriorEncoder")
except ImportError:
    pass
try:
    from .eeg_meg_prior_v2 import EEGMAEEncoder, PATCH_SIZE
    __all__ += ["EEGMAEEncoder", "PATCH_SIZE"]
except ImportError:
    pass


from .eeg_meg_prior_eigenval import (
    EEGMEGPriorEigenval, EEGMEGMAEDecoderEigenval,
    compute_E_lambda_from_phi,
)
from .fmri_prior_eigenval import (
    FMRIPriorEigenval, FMRIMAEDecoderEigenval,
    compute_fmri_group_eigval,
)
__all__ += [
    "EEGMEGPriorEigenval", "EEGMEGMAEDecoderEigenval", "compute_E_lambda_from_phi",
    "FMRIPriorEigenval", "FMRIMAEDecoderEigenval", "compute_fmri_group_eigval",
]


from .eeg_meg_prior_eigenval2 import (
    EEGMEGPriorEigenval2, EEGMEGMAEDecoderEigenval2,
)
__all__ += ["EEGMEGPriorEigenval2", "EEGMEGMAEDecoderEigenval2"]
