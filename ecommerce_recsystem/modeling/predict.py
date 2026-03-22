from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from loguru import logger
from tqdm import tqdm
import typer

from ecommerce_recsystem.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def _build_test_lookup(test_user_item: sparse.csr_matrix) -> dict:
    """Build a dict mapping user_index -> set of test item indices."""
    test_coo = test_user_item.tocoo()
    lookup = defaultdict(set)
    for u, i in zip(test_coo.row, test_coo.col):
        lookup[u].add(i)
    return lookup


def evaluate_model(
    model: AlternatingLeastSquares,
    train_user_item: sparse.csr_matrix,
    test_user_item: sparse.csr_matrix,
    K: int = 10,
    batch_size: int = 10000,
) -> Dict[str, float]:
    """
    Evaluate a trained model using ranking metrics computed via batch recommend.
    Bypasses implicit.evaluation (incompatible with scipy>=1.14).
    Returns dict with Precision@K, MAP@K, NDCG@K.
    """
    logger.info(f"Evaluating model at K={K}...")

    test_lookup = _build_test_lookup(test_user_item)
    test_users = sorted(test_lookup.keys())

    if not test_users:
        logger.warning("No users with test interactions found.")
        return {f"Precision@{K}": 0.0, f"MAP@{K}": 0.0, f"NDCG@{K}": 0.0}

    precisions = []
    avg_precisions = []
    ndcgs = []

    # Precompute ideal DCG denominators: 1/log2(j+2) for j in 0..K-1
    discount = 1.0 / np.log2(np.arange(2, K + 2))

    for start in tqdm(range(0, len(test_users), batch_size), desc=f"Eval@{K}"):
        batch_uids = np.array(test_users[start : start + batch_size])
        rec_ids, _ = model.recommend(
            batch_uids, train_user_item[batch_uids], N=K, filter_already_liked_items=True
        )

        for i, uid in enumerate(batch_uids):
            test_items = test_lookup[uid]
            recs = rec_ids[i]

            # Boolean hit vector
            hits = np.array([1 if r in test_items else 0 for r in recs])

            # Precision@K
            precisions.append(hits.sum() / K)

            # MAP@K: average precision
            cumhits = np.cumsum(hits)
            precision_at_j = cumhits / np.arange(1, K + 1)
            ap = (precision_at_j * hits).sum() / min(len(test_items), K)
            avg_precisions.append(ap)

            # NDCG@K
            dcg = (hits * discount).sum()
            n_relevant = min(len(test_items), K)
            idcg = discount[:n_relevant].sum()
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    metrics = {
        f"Precision@{K}": float(np.mean(precisions)),
        f"MAP@{K}": float(np.mean(avg_precisions)),
        f"NDCG@{K}": float(np.mean(ndcgs)),
    }

    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.6f}")

    return metrics


def evaluate_at_multiple_k(
    model: AlternatingLeastSquares,
    train_user_item: sparse.csr_matrix,
    test_user_item: sparse.csr_matrix,
    k_values: List[int] | None = None,
) -> pd.DataFrame:
    """
    Evaluate model at multiple K values. Returns a DataFrame with metrics as columns.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    rows = []
    for k in k_values:
        metrics = evaluate_model(model, train_user_item, test_user_item, K=k)
        # Normalize keys: "Precision@5" -> "Precision@K" so columns align across K values
        normalized = {name.split("@")[0]: val for name, val in metrics.items()}
        rows.append({"K": k, **normalized})

    return pd.DataFrame(rows).set_index("K")


def recommend_for_user(
    model: AlternatingLeastSquares,
    userid: int,
    user_items: sparse.csr_matrix,
    N: int = 10,
    filter_already_liked: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate top-N recommendations for a single user.
    Returns (item_indices, scores).
    """
    ids, scores = model.recommend(
        userid, user_items[userid], N=N, filter_already_liked_items=filter_already_liked
    )
    return ids, scores


def get_similar_items(
    model: AlternatingLeastSquares,
    itemid: int,
    N: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find N most similar items to the given item."""
    ids, scores = model.similar_items(itemid, N=N)
    return ids, scores


def map_recommendations_to_ids(
    rec_item_indices: np.ndarray,
    scores: np.ndarray,
    item_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map model's integer item indices back to original product_ids.
    item_mapping must have columns: product_id, item_index.
    """
    result = pd.DataFrame({"item_index": rec_item_indices, "score": scores})
    result = result.merge(item_mapping, on="item_index", how="left")
    return result[["product_id", "item_index", "score"]]


@app.command()
def main(
    model_path: Path = typer.Option(
        MODELS_DIR / "als_model.npz",
        help="Path to the trained model",
    ),
    input_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "warm_interactions_sample.csv",
        help="Path to the processed interactions CSV",
    ),
    n_sample_users: int = typer.Option(5, help="Number of sample users to recommend for"),
):
    """Evaluate a trained model and generate sample recommendations."""
    from ecommerce_recsystem.features import get_sparse_matrices
    from ecommerce_recsystem.modeling.train import load_model, train_test_split_implicit

    logger.info("Loading model and data...")
    model = load_model(model_path)
    df = pd.read_csv(input_path)

    sparse_user_item, _ = get_sparse_matrices(df)
    train_csr, test_csr = train_test_split_implicit(sparse_user_item)

    # Evaluate
    metrics_df = evaluate_at_multiple_k(model, train_csr, test_csr)
    logger.info(f"\n{metrics_df.to_string()}")

    # Sample recommendations
    item_mapping = df[["product_id", "item_index"]].drop_duplicates()
    sample_users = np.random.choice(sparse_user_item.shape[0], n_sample_users, replace=False)

    for uid in sample_users:
        ids, scores = recommend_for_user(model, uid, sparse_user_item, N=10)
        recs = map_recommendations_to_ids(ids, scores, item_mapping)
        logger.info(f"\nUser {uid} recommendations:\n{recs.to_string(index=False)}")

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
