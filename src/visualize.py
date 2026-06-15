import os
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(
    df,
    y_test,
    predictions
):
    os.makedirs(
        "outputs",
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 8)
    )

    sns.heatmap(
        df.corr(
            numeric_only=True
        ),
        annot=True
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/correlation_heatmap.png"
    )

    plt.close()

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_test,
        predictions
    )

    plt.xlabel(
        "Actual Price"
    )

    plt.ylabel(
        "Predicted Price"
    )

    plt.title(
        "Actual vs Predicted"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/actual_vs_predicted.png"
    )

    plt.close()