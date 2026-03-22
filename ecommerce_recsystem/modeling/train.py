import gc
import itertools
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

from implicit.als import AlternatingLeastSquares
from implicit.evaluation import train_test_split
from loguru import logger
import pandas as pd
import scipy.sparse as sparse
from tqdm import tqdm
import typer

from ecommerce_recsystem.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def train_test_split_implicit(
    sparse_user_item: sparse.csr_matrix,
    train_percentage: float = 0.8,
    random_state: int = 42,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Split a user-item interaction matrix into train/test sets.
    Randomly holds out (1 - train_percentage) of interactions per user.
    """
    logger.info(
        f"Splitting interactions ({sparse_user_item.nnz} nnz) "
        f"into {train_percentage:.0%} train / {1 - train_percentage:.0%} test..."
    )

    train, test = train_test_split(
        sparse_user_item.tocoo(), train_percentage=train_percentage, random_state=random_state
    )

    logger.info(f"Train set: {train.nnz} interactions | Test set: {test.nnz} interactions")
    return train, test


def train_als_model(
    train_user_item: sparse.csr_matrix,
    factors: int = 128,
    regularization: float = 0.01,
    alpha: float = 1.0,
    iterations: int = 20,
    random_state: int = 42,
) -> AlternatingLeastSquares:
    """
    Train an ALS model on the given user-item matrix.
    Returns the fitted model.
    """
    logger.info(
        f"Training ALS: factors={factors}, reg={regularization}, "
        f"alpha={alpha}, iterations={iterations}"
    )

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
        calculate_training_loss=True,
        random_state=random_state,
    )

    start = time.time()
    model.fit(train_user_item, show_progress=True)
    elapsed = time.time() - start

    logger.info(f"Training completed in {elapsed:.1f}s")
    return model


def tune_als_hyperparameters(
    train_csr: sparse.csr_matrix,
    test_csr: sparse.csr_matrix,
    param_grid: Dict[str, List[Any]],
    K: int = 10,
) -> Tuple[pd.DataFrame, AlternatingLeastSquares]:
    """
    Grid search over param_grid. Evaluates each combo with MAP@K.
    Returns (results_df sorted by MAP@K desc, best_model).
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    logger.info(f"Tuning ALS: {len(combos)} combinations to evaluate")

    results = []
    best_score = -1.0
    best_model = None

    for combo in tqdm(combos, desc="Tuning"):
        params = dict(zip(keys, combo))
        from ecommerce_recsystem.modeling.predict import evaluate_model

        model = train_als_model(train_csr, **params)

        metrics = evaluate_model(model, train_csr, test_csr, K=K)
        score = metrics[f"MAP@{K}"]

        results.append({**params, f"MAP@{K}": score})
        logger.info(f"  {params} -> MAP@{K} = {score:.6f}")

        if score > best_score:
            best_score = score
            best_model = model
        else:
            del model
            gc.collect()

    results_df = pd.DataFrame(results).sort_values(f"MAP@{K}", ascending=False)
    logger.info(f"Best MAP@{K}: {best_score:.6f}")
    return results_df, best_model


def save_model(model: AlternatingLeastSquares, model_path: Path) -> None:
    """Save a trained ALS model to disk."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")


def load_model(model_path: Path) -> AlternatingLeastSquares:
    """Load a trained ALS model from disk."""
    model = AlternatingLeastSquares.load(str(model_path))
    logger.info(f"Model loaded from {model_path}")
    return model


def save_id_mappings(interaction_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Extract and save user_id <-> user_index and product_id <-> item_index mappings.
    These are essential for translating model outputs back to real IDs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_map = (
        interaction_df[["user_id", "user_index"]]
        .drop_duplicates()
        .sort_values("user_index")
    )
    item_map = (
        interaction_df[["product_id", "item_index"]]
        .drop_duplicates()
        .sort_values("item_index")
    )

    user_path = output_dir / "user_id_mapping.csv"
    item_path = output_dir / "item_id_mapping.csv"

    user_map.to_csv(user_path, index=False)
    item_map.to_csv(item_path, index=False)

    logger.info(
        f"Saved {len(user_map)} user mappings to {user_path} "
        f"and {len(item_map)} item mappings to {item_path}"
    )


@app.command()
def main(
    input_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "warm_interactions_sample.csv",
        help="Path to the processed interactions CSV",
    ),
    model_path: Path = typer.Option(
        MODELS_DIR / "als_model.npz",
        help="Path to save the trained model",
    ),
    factors: int = typer.Option(128, help="Number of latent factors"),
    regularization: float = typer.Option(0.01, help="L2 regularization"),
    alpha: float = typer.Option(1.0, help="Confidence scaling factor"),
    iterations: int = typer.Option(20, help="Number of ALS iterations"),
    train_percentage: float = typer.Option(0.8, help="Train split percentage"),
):
    """Train an ALS model on processed interaction data."""
    from ecommerce_recsystem.features import get_sparse_matrices

    logger.info(f"Loading interactions from {input_path}...")
    df = pd.read_csv(input_path)

    sparse_user_item, _ = get_sparse_matrices(df)
    train_csr, test_csr = train_test_split_implicit(
        sparse_user_item, train_percentage=train_percentage
    )

    model = train_als_model(
        train_csr,
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
    )

    save_model(model, model_path)
    save_id_mappings(df, MODELS_DIR)
    logger.success("Training pipeline complete.")


if __name__ == "__main__":
    app()
