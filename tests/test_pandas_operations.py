import pandas as pd
import pytest
from dataframe_operations.pandas import PandasDataFrameOperations


@pytest.fixture
def operations():
    return PandasDataFrameOperations.setup(1)


def test_filter(operations: PandasDataFrameOperations):
    users_df = pd.DataFrame(
        {
            "is_deleted": [False, True, False, False, False],
            "address": ["Box 123", "Box 456", "Box 789", "Sample Address", "Box 789"],
            "balance": [5000, 15000, 20000, 30000, 40000],
            "cards": [
                [{"provider": "Visa"}],
                [{"provider": "Mastercard"}],
                [
                    {"provider": "Mastercard"},
                    {"provider": "Visa"},
                    {"provider": "Visa"},
                ],
                [{"provider": "Mastercard"}],
                [{"provider": "Visa"}, {"provider": "Visa"}, {"provider": "Visa"}],
            ],
        }
    )

    # Apply the filter method
    result = operations.filter(users_df)

    # Define the expected result
    expected_result = pd.DataFrame(
        {
            "is_deleted": [False],
            "address": ["Box 789"],
            "balance": [20000],
            "cards": [
                [
                    {"provider": "Mastercard"},
                    {"provider": "Visa"},
                    {"provider": "Visa"},
                ],
            ],
        }
    )

    pd.testing.assert_frame_equal(
        operations.to_pandas(result).reset_index(drop=True),
        expected_result.reset_index(drop=True),
        check_index_type=False,
    )
