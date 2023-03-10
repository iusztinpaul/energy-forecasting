from great_expectations.core import ExpectationSuite, ExpectationConfiguration

# Create an expectation suite
expectation_suite_energy_consumption = ExpectationSuite(
    expectation_suite_name="energy_consumption_suite"
)

# Set expected columns
expectation_suite_energy_consumption.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_table_columns_to_match_ordered_list",
        kwargs={"column_list": ["datetime_utc", "area", "consumer_type", "energy_consumption"]}
    )
)

# # Datetime UTC
# # TODO: Build custom validation to ensure datetime_utc is type datetime
# expectation_suite_energy_consumption.add_expectation(
#     ExpectationConfiguration(
#         expectation_type="expect_column_values_to_be_type_datetime", 
#         kwargs={
#             "column": "datetime_utc",
#             "strftime_format": "%Y-%m-%dT%H:%M:%S"
#         }
#     )
# )

# Area
expectation_suite_energy_consumption.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={
            "column": "area",
            "value_set": (0, 1, 2)
        }
    )
)

expectation_suite_energy_consumption.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_of_type",
        kwargs={
            "column": "area",
            "type_": "int8"
        }
    )
)

# Consumer type
expectation_suite_energy_consumption.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_min_to_be_between",
        kwargs={
            "column": "consumer_type",
            "min_value": 0,
            "strict_min": True
        }
    )
)

expectation_suite_energy_consumption.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_of_type",
        kwargs={
            "column": "consumer_type",
            "type_": "int32"
        }
    )
)

# Energy consumption
expectation_suite_energy_consumption.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_min_to_be_between",
        kwargs={
            "column": "energy_consumption",
            "min_value": 0,
            "strict_min": False
        }
    )
)

expectation_suite_energy_consumption.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_of_type",
        kwargs={
            "column": "energy_consumption",
            "type_": "float64"
        }
    )
)