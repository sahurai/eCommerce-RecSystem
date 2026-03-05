from pathlib import Path
import pandas as pd
from loguru import logger
import typer
from ecommerce_recsystem.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def process_implicit_data(df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    """
    Core function to clean and preprocess raw eCommerce data for Implicit CF.
    - Fills missing categorical values.
    - Drops missing user_id/product_id.
    - Mitigates Cold-Start by filtering inactive users.
    - Maps events to implicit confidence weights.
    - Creates contiguous integer indices for matrix factorization.
    """
    logger.info("Handling missing values...")
    df['category_code'] = df.get('category_code', pd.Series()).fillna('unknown_category')
    df['brand'] = df.get('brand', pd.Series()).fillna('unknown_brand')

    # Drop critical NaNs
    initial_len = len(df)
    df = df.dropna(subset=['user_id', 'product_id'])
    logger.info(f"Dropped {initial_len - len(df)} rows with missing user_id or product_id.")

    logger.info(f"Filtering users with less than {min_interactions} interactions...")
    user_activity = df.groupby('user_id').size()
    active_users = user_activity[user_activity >= min_interactions].index

    original_users = df['user_id'].nunique()
    df_filtered = df[df['user_id'].isin(active_users)].copy()
    remaining_users = df_filtered['user_id'].nunique()
    logger.info(f"Filtered out {original_users - remaining_users} inactive users.")

    logger.info("Mapping event types to implicit weights...")
    event_weights = {
        'view': 1,
        'cart': 3,
        'purchase': 5
    }
    df_filtered['implicit_weight'] = df_filtered['event_type'].map(event_weights)

    logger.info("Aggregating interactions (summing weights for user-item pairs)...")
    interaction_df = df_filtered.groupby(['user_id', 'product_id'])['implicit_weight'].sum().reset_index()
    interaction_df.rename(columns={'implicit_weight': 'confidence'}, inplace=True)

    logger.info("Creating contiguous indices (cat.codes) for matrix factorization...")
    interaction_df['user_index'] = interaction_df['user_id'].astype("category").cat.codes
    interaction_df['item_index'] = interaction_df['product_id'].astype("category").cat.codes

    return interaction_df


@app.command()
def main(
        # Setting smart default paths based on your config
        input_path: Path = typer.Option(
            RAW_DATA_DIR / "2019-Oct.csv",
            help="Path to the raw CSV dataset"
        ),
        output_path: Path = typer.Option(
            PROCESSED_DATA_DIR / "implicit_interactions.csv",
            help="Path to save the processed dataset"
        ),
        nrows: int = typer.Option(
            5000000,
            help="Number of rows to process (useful for memory limits)"
        ),
        min_interactions: int = typer.Option(
            5,
            help="Minimum number of interactions a user must have to be kept"
        )
):
    """
    CLI tool to process raw eCommerce data into an implicit interaction matrix format.
    """
    logger.info(f"Starting data processing pipeline...")
    logger.info(f"Reading first {nrows} rows from {input_path}...")

    try:
        df = pd.read_csv(input_path, nrows=nrows)
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}. Check your path or config.")
        raise typer.Exit(code=1)

    # Process the data using our core function
    processed_df = process_implicit_data(df, min_interactions=min_interactions)

    # Save to processed folder
    logger.info(f"Saving processed dataset ({processed_df.shape[0]} interactions) to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    logger.success("Dataset processing complete! Data is ready for modeling.")


if __name__ == "__main__":
    app()