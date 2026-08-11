import pandas as pd

from compare_segmentation_techniques import build_feature_schema


def test_feature_schema_restricts_segmentation_to_requested_columns():
    dev_df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "score": [0.1, 0.8, 0.2, 0.9],
            "ead": [100, 200, 150, 250],
            "age": [25, 35, 45, 55],
            "income": [30000, 50000, 70000, 90000],
            "region": ["North", "South", "North", "South"],
            "occupation": ["Engineer", "Doctor", "Engineer", "Doctor"],
            "other_feature": [1, 2, 3, 4],
        }
    )

    schema = build_feature_schema(dev_df)

    assert schema.numeric_cols == ["age", "income"]
    assert schema.categorical_cols == ["region", "occupation"]
    assert schema.exclude_cols == ["ead", "other_feature", "score", "target"]
