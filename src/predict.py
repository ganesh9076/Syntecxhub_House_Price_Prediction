import pandas as pd

def save_predictions(
    model,
    X_test
):
    predictions = model.predict(
        X_test
    )

    results = pd.DataFrame(
        {
            "Predicted Price":
            predictions
        }
    )

    results.to_csv(
        "outputs/sample_predictions.csv",
        index=False
    )

    print(
        "\nPredictions saved successfully."
    )

    return predictions