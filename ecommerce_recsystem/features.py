from pathlib import Path
import pandas as pd
import scipy.sparse as sparse
from loguru import logger
import typer

from ecommerce_recsystem.config import PROCESSED_DATA_DIR

app = typer.Typer()


def get_sparse_matrices(interaction_df: pd.DataFrame):
    """
    Core function to convert a processed interaction dataframe into sparse CSR matrices.
    Returns both User-Item and Item-User matrices (required by Implicit CF models).
    """
    logger.info("Extracting matrix dimensions...")
    n_users = interaction_df['user_index'].nunique()
    n_items = interaction_df['item_index'].nunique()

    logger.info(f"Creating sparse matrices for {n_users} users and {n_items} items...")

    # 1. User-Item Matrix
    sparse_user_item = sparse.csr_matrix(
        (interaction_df['confidence'], (interaction_df['user_index'], interaction_df['item_index'])),
        shape=(n_users, n_items)
    )

    # 2. Item-User Matrix (Implicit ALS specifically needs this orientation for training)
    sparse_item_user = sparse.csr_matrix(
        (interaction_df['confidence'], (interaction_df['item_index'], interaction_df['user_index'])),
        shape=(n_items, n_users)
    )

    # Calculate Sparsity
    matrix_size = sparse_user_item.shape[0] * sparse_user_item.shape[1]
    num_interactions = sparse_user_item.nnz
    sparsity = 100 * (1 - (num_interactions / matrix_size))

    logger.info(f"Matrix Sparsity: {sparsity:.4f}%")

    return sparse_user_item, sparse_item_user


@app.command()
def main(
        # We point to the processed CSV created by dataset.py
        input_path: Path = typer.Option(
            PROCESSED_DATA_DIR / "implicit_interactions_sample.csv",
            help="Path to the processed interactions CSV"
        ),
        # We output to a directory because we are saving two .npz files
        output_dir: Path = typer.Option(
            PROCESSED_DATA_DIR,
            help="Directory to save the generated sparse matrix features"
        ),
):
    """
    CLI command to generate sparse interaction matrices from processed data.
    """
    logger.info(f"Generating features (sparse matrices) from {input_path}...")

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}. Please run dataset.py first!")
        raise typer.Exit(code=1)

    # 1. Generate matrices using our core function
    sparse_user_item, sparse_item_user = get_sparse_matrices(df)

    # 2. Define output paths
    output_dir.mkdir(parents=True, exist_ok=True)
    user_item_path = output_dir / "sparse_user_item.npz"
    item_user_path = output_dir / "sparse_item_user.npz"

    # 3. Save matrices in optimized .npz format
    logger.info("Saving matrices to disk in .npz format...")
    sparse.save_npz(user_item_path, sparse_user_item)
    sparse.save_npz(item_user_path, sparse_item_user)

    logger.success(f"Features generation complete! Files saved to {output_dir}")


if __name__ == "__main__":
    app()