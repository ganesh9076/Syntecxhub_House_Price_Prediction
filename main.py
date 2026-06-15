from src.data_preprocessing import preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.predict import save_predictions
from src.visualize import generate_plots
from src.custom_predict import custom_prediction

def main():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        df
    ) = preprocess_data()

    (
        model,
        model_name
    ) = train_model(
        X_train,
        y_train,
        X_test,
        y_test
    )

    predictions = save_predictions(
        model,
        X_test
    )

    evaluate_model(
        model,
        model_name,
        X_test,
        y_test
    )

    generate_plots(
        df,
        y_test,
        predictions
    )

    print("\n" + "=" * 50)
    print("HOUSE PRICE PREDICTION SYSTEM")
    print("=" * 50)
    print(f"Best Model: {model_name}")
    print("Model Saved Successfully")
    print("Reports Generated Successfully")
    print("=" * 50)

    print("\nCUSTOM HOUSE PRICE PREDICTION")

    area = float(input("Area: "))
    bedrooms = int(input("Bedrooms: "))
    bathrooms = int(input("Bathrooms: "))
    stories = int(input("Stories: "))

    mainroad = input("Main Road (yes/no): ")
    guestroom = input("Guest Room (yes/no): ")
    basement = input("Basement (yes/no): ")
    hotwaterheating = input("Hot Water Heating (yes/no): ")
    airconditioning = input("Air Conditioning (yes/no): ")

    parking = int(input("Parking Spaces: "))

    prefarea = input("Preferred Area (yes/no): ")

    furnishingstatus = input(
        "Furnishing Status (furnished/semi-furnished/unfurnished): "
    )

    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }

    custom_prediction(
        model,
        input_data,
        X_train
    )

if __name__ == "__main__":
    main()