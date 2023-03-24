import pandas as pd
from category_encoders import hashing
from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose import CORE_MTYPES


class AttachAreaConsumerType(BaseTransformer):
    # TODO: Double check these tags. I just copied them from the Id / IdentityTransformer
    _tags = {
        "capability:inverse_transform": True,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": CORE_MTYPES,  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": True,
        # does transform return have the same time index as input X
        "handles-missing-data": True,  # can estimator handle missing data?
    }

    def _transform(self, X, y=None):
        X["area_exog"] = X.index.get_level_values(0)
        X["consumer_type_exog"] = X.index.get_level_values(1)

        return X

    def _inverse_transform(self, X, y=None):
        X = X.drop(columns=["area_exog", "consumer_type_exog"])

        return X


class HashingEncoder(BaseTransformer):
    # TODO: Double check these tags. I just copied them from the Id / IdentityTransformer
    _tags = {
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": CORE_MTYPES,  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": True,
        # does transform return have the same time index as input X
        "handles-missing-data": True,  # can estimator handle missing data?
    }

    def __init__(self, **kwargs):
        super().__init__()

        self.hashing_encoder = hashing.HashingEncoder(**kwargs)

    def _fit(self, X, y=None):
        self.hashing_encoder.fit(X)

    def _transform(self, X: pd.DataFrame, y=None):
        return self.hashing_encoder.transform(X)
