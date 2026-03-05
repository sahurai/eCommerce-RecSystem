from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import typer
from ecommerce_recsystem.config import FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def plot_sales_funnel(df: pd.DataFrame, save_path: Path = None) -> None:
    """
    Core function to plot a sales funnel from the dataset.
    Can be imported into Jupyter Notebooks or run via the CLI.
    """
    if 'event_type' not in df.columns:
        logger.error("The dataframe must contain an 'event_type' column.")
        raise ValueError("Missing 'event_type' column")

    logger.info("Calculating event distribution...")
    event_counts = df['event_type'].value_counts()

    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(x=event_counts.index, y=event_counts.values, palette="viridis")

    plt.title("Distribution of Event Types (Sales Funnel)", fontsize=14)
    plt.ylabel("Number of Events (Log Scale)", fontsize=12)
    plt.xlabel("Event Type", fontsize=12)
    plt.yscale('log')

    # Add text labels on top of bars
    for i, count in enumerate(event_counts.values):
        plt.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save or show the plot
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.success(f"Plot successfully saved to: {save_path}")
    else:
        plt.show()

    plt.close()

    # Log conversion rates
    total_views = event_counts.get('view', 0)
    total_carts = event_counts.get('cart', 0)
    total_purchases = event_counts.get('purchase', 0)

    logger.info("--- Conversion Metrics ---")
    if total_views > 0:
        logger.info(f"View -> Cart: {(total_carts / total_views) * 100:.2f}%")
    if total_carts > 0:
        logger.info(f"Cart -> Purchase: {(total_purchases / total_carts) * 100:.2f}%")


@app.command()
def main(
        # By default, we point to the raw data for the EDA plot
        # Change RAW_DATA_DIR to PROCESSED_DATA_DIR if you only defined PROCESSED_DATA_DIR in config.py
        input_path: Path = typer.Option(
            RAW_DATA_DIR / "2019-Oct.csv",
            help="Path to the raw CSV dataset"
        ),
        output_path: Path = typer.Option(
            FIGURES_DIR / "sales_funnel.png",
            help="Path to save the generated plot"
        ),
        nrows: int = typer.Option(
            5000000,
            help="Number of rows to read (to prevent OOM errors)"
        ),
):
    """
    CLI command to generate and save the sales funnel plot.
    """
    logger.info(f"Starting plot generation process...")
    logger.info(f"Reading first {nrows} rows from {input_path}")

    try:
        df = pd.read_csv(input_path, nrows=nrows)
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}. Please check your path.")
        raise typer.Exit(code=1)

    plot_sales_funnel(df, save_path=output_path)


if __name__ == "__main__":
    app()