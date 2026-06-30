import numpy as np
import torch
from pathlib import Path
from scipy.spatial.distance import cdist
from priors.models_torch import AdaLNResNetCrossAttn

class PoserRetrievalHarness:
    def __init__(self, prior_model_path, lda_basis_path, device="cpu", model_kwargs=None):
        self.device = device
        
        # Load LDA
        lda_data = np.load(lda_basis_path)
        self.lda_basis = lda_data["lda_basis"]
        self.pooled_mean = lda_data["pooled_mean"]
        
        if model_kwargs is None:
            model_kwargs = dict(d_in=64, d_out=64, d_hidden=1024, n_blocks=12, d_cond=1024, n_heads=8, n_queries=4)
            
        self.model = AdaLNResNetCrossAttn(**model_kwargs)
        
        ckpt = torch.load(prior_model_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.index_vectors = None
        self.index_labels = None

    @torch.no_grad()
    def predict_pin(self, t5_embedding: np.ndarray, mask: np.ndarray = None,
                    dt: float = 0.1, max_tokens: int = 256) -> np.ndarray:
        """Predict identity pin(s) from full T5 sequence(s) via the FM ODE.

        Replicates the EXACT conditioning preprocessing used at training
        (scripts/pipeline/train_exp1_g2.py): keep only mask-valid tokens, cap to
        max_tokens, zero-pad back to max_tokens. The model has no internal
        key-padding mask, so feeding raw 512-token sequences (with T5's non-zero
        padding embeddings) is out-of-distribution and corrupts conditioning.

        Args:
            t5_embedding: (B, S, 1024) full T5 hidden states (S typically 512).
            mask: (B, S) bool array of valid (non-padding) tokens. If None, all
                  tokens are treated as valid (legacy behaviour — only correct
                  when the caller has already masked/capped).
            dt: Euler step size (training used 0.1 → 10 steps).
            max_tokens: token cap matching training MAX_TOKENS (256).
        """
        t5 = np.asarray(t5_embedding, dtype=np.float32)
        if t5.ndim == 2:
            t5 = t5[None, ...]
        B, S, D = t5.shape

        # Build masked + capped + zero-padded sequences, exactly as training did.
        cond_np = np.zeros((B, max_tokens, D), dtype=np.float32)
        for i in range(B):
            if mask is not None:
                valid = t5[i][mask[i].astype(bool)]
            else:
                valid = t5[i]
            valid = valid[:max_tokens]
            cond_np[i, :len(valid)] = valid

        cond = torch.from_numpy(cond_np).float().to(self.device)
        D_OUT = self.model.proj_out.weight.shape[0]

        x = torch.randn(B, D_OUT, device=self.device)
        for k in range(int(round(1.0 / dt))):
            t = torch.full((B, 1), k * dt, device=self.device)
            v = self.model(x, t, cond)
            x = x + v * dt

        return x.cpu().numpy()

    def build_index(self, vectors: np.ndarray, labels: list):
        """Store vectors for retrieval. vectors shape (N, 64)"""
        assert len(vectors) == len(labels)
        self.index_vectors = vectors
        self.index_labels = labels
        self.index_std = np.std(vectors, axis=0)

    def retrieve(self, query_pin: np.ndarray, k: int = 10, metric: str = 'euclidean'):
        """Retrieve top k matches for query_pin of shape (1, 64)"""
        if self.index_vectors is None:
            raise ValueError("Index not built.")
            
        dists = cdist(query_pin, self.index_vectors, metric=metric)[0]
        top_indices = np.argsort(dists)[:k]
        
        results = []
        for idx in top_indices:
            results.append({
                "label": self.index_labels[idx],
                "distance": dists[idx]
            })
        return results


    def manipulate_pin(self, pin: np.ndarray, axis: int, step_sigma: float) -> np.ndarray:
        if self.index_vectors is None:
            raise ValueError("Index not built. Cannot determine population sigma.")
        new_pin = pin.copy()
        new_pin[:, axis] += step_sigma * self.index_std[axis]
        return new_pin


def evaluate_recall(queries: np.ndarray, index_vectors: np.ndarray, query_labels: list, index_labels: list, k: int) -> float:
    from scipy.spatial.distance import cdist
    dists = cdist(queries, index_vectors, metric='euclidean')
    hits = 0
    for i, q_label in enumerate(query_labels):
        top_indices = np.argsort(dists[i])[:k]
        top_labels = [index_labels[idx] for idx in top_indices]
        if q_label in top_labels:
            hits += 1
    return hits / len(query_labels) if len(query_labels) > 0 else 0.0

def generate_random_null(queries: np.ndarray) -> np.ndarray:
    """Generate random Gaussian vectors with matched norm per row."""
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    random_dirs = np.random.randn(*queries.shape).astype(np.float32)
    random_norms = np.linalg.norm(random_dirs, axis=1, keepdims=True)
    return (random_dirs / random_norms) * norms

