import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

def evaluate_model(
    model,
    model_name,
    X_test,
    y_test
):
    predictions = model.predict(
        X_test
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    with open(
        "outputs/model_report.txt",
        "w"
    ) as file:

        file.write(
            f"Best Model: {model_name}\n"
        )

        file.write(
            f"RMSE: {rmse:.2f}\n"
        )

        file.write(
            f"R2 Score: {r2:.4f}\n"
        )

    if hasattr(
        model,
        "feature_importances_"
    ):
        importance = pd.DataFrame(
            {
                "Feature":
                X_test.columns,

                "Importance":
                model.feature_importances_
            }
        )

        importance.sort_values(
            by="Importance",
            ascending=False
        ).to_csv(
            "outputs/feature_importance.csv",
            index=False
        )

    print("\nMODEL EVALUATION")
    print(
        f"Best Model : {model_name}"
    )
    print(
        f"RMSE : {rmse:.2f}"
    )
    print(
        f"R2 Score : {r2:.4f}"
    )