import os
import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def train_model(
    X_train,
    y_train,
    X_test,
    y_test
):
    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    models = {
        "Linear Regression":
        LinearRegression(),

        "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42
        )
    }

    comparison = []

    best_model = None
    best_name = ""
    best_score = float("-inf")

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        score = r2_score(
            y_test,
            predictions
        )

        comparison.append(
            [name, score]
        )

        print(
            f"{name} R2 Score: {score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    pd.DataFrame(
        comparison,
        columns=[
            "Model",
            "R2 Score"
        ]
    ).to_csv(
        "outputs/model_comparison.csv",
        index=False
    )

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    return (
        best_model,
        best_name
    )