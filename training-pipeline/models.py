import lightgbm as lgb
from category_encoders import hashing
from sktime.forecasting.compose import make_reduction, ForecastingPipeline
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.summarize import WindowSummarizer


def build_model(config: dict):
    lag = config.pop(
        "forecaster_transformers__window_summarizer__lag_feature__lag",
        list(range(1, 72 + 1)),
    )
    mean = config.pop(
        "forecaster_transformers__window_summarizer__lag_feature__mean",
        [[1, 24], [1, 48], [1, 72]],
    )
    std = config.pop(
        "forecaster_transformers__window_summarizer__lag_feature__std",
        [[1, 24], [1, 48], [1, 72]],
    )
    n_jobs = config.pop("forecaster_transformers__window_summarizer__n_jobs", 1)
    window_summarizer = WindowSummarizer(
        **{"lag_feature": {"lag": lag, "mean": mean, "std": std}},
        n_jobs=n_jobs,
    )

    regressor = lgb.LGBMRegressor()
    forecaster = make_reduction(
        regressor,
        transformers=[window_summarizer],
        strategy="recursive",
        pooling="global",
        window_length=None,
    )

    pipe = ForecastingPipeline(
        steps=[
            (
                "daily_season",
                DateTimeFeatures(
                    manual_selection=["day_of_week", "hour_of_day"],
                    keep_original_columns=True,
                ),
                (
                    "encode_categorical",
                    TabularToSeriesAdaptor(
                        hashing.HashingEncoder(
                            return_df=True,
                            cols=[
                                "area",
                                "consumer_type",
                                "day_of_week",
                                "hour_of_day",
                            ],
                        )
                    ),
                ),
            ),
            ("forecaster", forecaster),
        ]
    )
    pipe = pipe.set_params(**config)

    return pipe


def build_baseline_model():
    forecaster = NaiveForecaster(sp=24)

    return forecaster
