from __future__ import annotations

import logging

from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from collections.abc import Mapping
from autorad.config import config

log = logging.getLogger(__name__)


def create_oversampling_model(method: str|dict, random_state: int = config.SEED):
    if method is None:
        return None
    if isinstance(method, str):
        return _checks(method, random_state=random_state)
    elif isinstance(method, Mapping):
        kwarg_dict = {k:v for k, v in method.items() if k!='_method_'}
        return _checks(method['_method_'], random_state=random_state, **kwarg_dict)
    else:
        raise TypeError(f"method is not a recognised datatype, got {type(method)}")

def _checks(method_name, **kwargs):
    if method_name == "ADASYN":
        return ADASYN(**kwargs)
    elif method_name == "SMOTE":
        return SMOTE(**kwargs)
    elif method_name == "BorderlineSMOTE":
        return BorderlineSMOTE(kind="borderline-1", **kwargs)
    elif method_name=="SMOTETomek":
        return SMOTETomek(**kwargs)
    elif method_name=="SMOTEENN":
        return SMOTEENN(**kwargs)
    raise ValueError(f"Unknown oversampling method: {method_name}")

class OversamplerWrapper:
    def __init__(self, oversampler, random_state=config.SEED):
        self.oversampler = oversampler
        self.oversampler.__init__(random_state=random_state)

    def fit(self, X, y):
        return self.oversampler.fit(X, y)

    def fit_resample(self, X, y):
        return self.oversampler.fit_resample(X, y)

    def transform(self, X):
        log.debug(f"{self.oversampler} does nothing on .transform()...")
        return X
