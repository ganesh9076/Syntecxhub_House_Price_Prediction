import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data():
    df = pd.read_csv("Data/Housing.csv")

    df = df.dropna()

    target_column = "price"

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        df
)