import pandas as pd

def custom_prediction(
    model,
    input_data,
    X_train
):
    user_df = pd.DataFrame(
        [input_data]
    )

    user_df = pd.get_dummies(
        user_df
    )

    user_df = user_df.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    prediction = model.predict(
        user_df
    )[0]

    print("\n" + "=" * 50)
    print(
        f"Predicted House Price: ₹ {prediction:,.2f}"
    )
    print("=" * 50)