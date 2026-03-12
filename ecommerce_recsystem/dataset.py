import numpy as np
import pandas as pd
import typer
from pathlib import Path
from loguru import logger
from typing import Tuple

# Placeholders for your configs
from ecommerce_recsystem.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def process_implicit_data(df: pd.DataFrame, min_interactions: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Core function to clean and preprocess raw eCommerce data for Implicit CF.
    Returns two DataFrames:
        1. Warm users (ready for matrix factorization)
        2. Cold users (for fallback/popularity models)
    """
    logger.info("Handling missing categorical values...")
    df['category_code'] = df.get('category_code', pd.Series(dtype='object')).fillna('unknown_category')
    df['brand'] = df.get('brand', pd.Series(dtype='object')).fillna('unknown_brand')

    # Drop missing user_id/product_id.
    initial_len = len(df)
    df = df.dropna(subset=['user_id', 'product_id'])
    logger.info(f"Dropped {initial_len - len(df)} rows with missing user_id or product_id.")

    # 1. SMART COLD-START: Split instead of dropping
    logger.info(f"Splitting users into 'warm' (>= {min_interactions} interactions) and 'cold'...")
    user_activity = df['user_id'].value_counts()
    warm_user_ids = user_activity[user_activity >= min_interactions].index

    # Create a boolean mask to separate users
    df['is_warm'] = df['user_id'].isin(warm_user_ids)

    df_warm = df[df['is_warm']].copy()
    df_cold = df[~df['is_warm']].copy()

    logger.info(f"Warm users: {df_warm['user_id'].nunique()} | Cold users: {df_cold['user_id'].nunique()}")

    # 2. CONFIDENCE CALCULATION (for Implicit CF)
    logger.info("Mapping event types to implicit weights...")
    event_weights = {
        'view': 1,
        'cart': 3,
        'purchase': 5
    }
    df_warm['implicit_weight'] = df_warm['event_type'].map(event_weights).fillna(1)  # fallback if unknown

    logger.info("Aggregating interactions (summing weights for user-item pairs)...")
    interaction_df = df_warm.groupby(['user_id', 'product_id'])['implicit_weight'].sum().reset_index()

    # Log-smoothing to handle extreme outliers (bots/power users).
    # Note: NO [0, 1] normalization. ALS prefers unnormalized smoothed counts.
    logger.info("Applying log-smoothing to raw weights for the confidence matrix...")
    interaction_df['confidence'] = np.log1p(interaction_df['implicit_weight'])

    # Drop the raw column, keep the smoothed confidence
    interaction_df = interaction_df.drop(columns=['implicit_weight'])

    # 3. INDEXING FOR CF
    logger.info("Creating contiguous indices (cat.codes) for matrix factorization...")
    interaction_df['user_index'] = interaction_df['user_id'].astype("category").cat.codes
    interaction_df['item_index'] = interaction_df['product_id'].astype("category").cat.codes

    return interaction_df, df_cold


@app.command()
def main(
        input_path: Path = typer.Option(
            RAW_DATA_DIR / "2019-Oct.csv",
            help="Path to the raw CSV dataset"
        ),
        output_warm_path: Path = typer.Option(
            PROCESSED_DATA_DIR / "warm_interactions.csv",
            help="Path to save processed CF data"
        ),
        output_cold_path: Path = typer.Option(
            PROCESSED_DATA_DIR / "cold_users_data.csv",
            help="Path to save cold user data for fallback strategies"
        ),
        nrows: int = typer.Option(
            5000000,
            help="Number of rows to process (useful for memory limits)"
        ),
        min_interactions: int = typer.Option(
            5,
            help="Minimum number of interactions a user must have to be considered 'warm'"
        )
):
    """
    CLI tool to process raw eCommerce data into an implicit interaction matrix format,
    while gracefully handling cold-start users.
    """
    logger.info(f"Starting data processing pipeline...")
    logger.info(f"Reading first {nrows} rows from {input_path}...")

    try:
        df = pd.read_csv(input_path, nrows=nrows)
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}. Check your path or config.")
        raise typer.Exit(code=1)

    # Process the data using our core function
    warm_df, cold_df = process_implicit_data(df, min_interactions=min_interactions)

    # Ensure output directories exist
    output_warm_path.parent.mkdir(parents=True, exist_ok=True)
    output_cold_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the datasets
    logger.info(f"Saving WARM dataset ({len(warm_df)} interactions) -> {output_warm_path}")
    warm_df.to_csv(output_warm_path, index=False)

    logger.info(f"Saving COLD dataset ({len(cold_df)} raw interactions) -> {output_cold_path}")
    cold_df.to_csv(output_cold_path, index=False)

    logger.success("Dataset processing complete! Warm data is ready for ALS modeling.")


if __name__ == "__main__":
    app()