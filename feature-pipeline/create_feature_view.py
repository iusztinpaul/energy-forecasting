import datetime
import os

import hopsworks
from dotenv import load_dotenv

load_dotenv()

FS_API_KEY = os.getenv("FS_API_KEY")


def main(target: str = "energy_consumption", **kwargs):
    project = hopsworks.login(api_key_value=FS_API_KEY, project="energy_consumption")
    fs = project.get_feature_store()
    energy_consumption_fg = fs.get_feature_group('energy_consumption_denmark', version=1)

    # Create feature view.
    ds_query = energy_consumption_fg.select_all()
    # TODO: Write transformation functions.
    # standard_scaler = fs.get_transformation_function(name='label_encoder')
    # transformation_functions = {
    #     "consumer_type": standard_scaler,
    #     "area": standard_scaler
    # }

    feature_view = fs.create_feature_view(
        name="energy_consumption_denmark_view",
        description="Energy consumption for Denmark forecasting model.",
        query=ds_query,
        labels=[]
        # labels=[target],
        # transformation_functions=transformation_functions,
    )

    # Create the train, validation and test splits.
    # TODO: Find a better way to compute the splits.
    # Data has a delay of 15 days. Thus, we have to shift our window with 15 days.
    days_delay = kwargs.get("days_delay", 15)
    # This is the actual time window of data we will use to train our model on.
    days_export = kwargs.get("days_export", 30)
    export_end = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    export_start = export_end - datetime.timedelta(days=days_delay + days_export)

    train_start, train_end = export_start, export_end - datetime.timedelta(days=7, minutes=1)
    test_start, test_end = export_end - datetime.timedelta(days=7), export_end

    # TODO: Try to make the splits within Hopsworks. But the last time I tried it, it thrown an error.
    feature_view.create_training_data(
        description='Energy consumption training dataset',
        data_format='csv',
        # train_start=train_start,
        # train_end=train_end,
        # test_start=test_start,
        # test_end=test_end,
        write_options={'wait_for_job': False},
        coalesce=True,
    )


if __name__ == "__main__":
    main()
