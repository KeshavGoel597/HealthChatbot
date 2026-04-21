"""
SapBERT embedding index for CUI semantic search.

Loads pre-computed CUI embeddings (from DR.KNOWS) and the SapBERT encoder.
Builds a FAISS index for fast cosine similarity search over 407k CUIs.
"""

import json
import pickle
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from app.services.torch_runtime import detect_torch_runtime

SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


class EmbeddingIndex:
    """Singleton-style index: load once at startup, query instantly."""

    def __init__(self, embedding_path: str, vocab_path: str | None = None):
        """
        Args:
            embedding_path: Path to GraphModel_SNOMED_CUI_Embedding.pkl
                            Dict[str, ndarray(1, 768)]
            vocab_path:     Path to CUI_Vocab.json (pickle format).
                            Dict[str, list[[aui_id, preferred_text], ...]]
                            If None, auto-detects next to embedding_path.
        """
        # --- Load CUI embeddings ---
        path = Path(embedding_path)
        if not path.exists():
            raise FileNotFoundError(f"CUI embedding file not found: {path}")

        with open(path, "rb") as f:
            raw: dict[str, np.ndarray] = pickle.load(f)

        self.cui_list: list[str] = list(raw.keys())
        matrix = np.stack([raw[c].squeeze() for c in self.cui_list]).astype(np.float32)

        # L2-normalize so dot product = cosine similarity
        faiss.normalize_L2(matrix)

        # Build FAISS inner-product index
        dim = matrix.shape[1]  # 768
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(matrix)

        # --- Load CUI name vocabulary ---
        self.cui_names: dict[str, str] = {}
        vocab = self._resolve_vocab_path(path, vocab_path)
        if vocab and vocab.exists():
            with open(vocab, "rb") as f:
                raw_vocab = pickle.load(f)
            # Extract shortest preferred text per CUI as the display name
            for cui, entries in raw_vocab.items():
                if entries:
                    # Pick the shortest, most readable name
                    self.cui_names[cui] = min((e[1] for e in entries), key=len)
            print(f"[EmbeddingIndex] Loaded {len(self.cui_names)} CUI names.")

        # --- Load SapBERT encoder ---
        backend_name, device, _, _ = detect_torch_runtime()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(SAPBERT_MODEL)
        self.model = AutoModel.from_pretrained(SAPBERT_MODEL)
        self.model.to(self.device)
        self.model.eval()

        print(
            f"[EmbeddingIndex] Loaded {len(self.cui_list)} CUI embeddings, "
            f"SapBERT on {backend_name.upper()}, FAISS index ready."
        )

    @staticmethod
    def _resolve_vocab_path(embedding_path: Path, vocab_path: str | None) -> Path | None:
        """Find CUI_Vocab.json next to the embedding file if not specified."""
        if vocab_path:
            return Path(vocab_path)
        candidate = embedding_path.parent / "CUI_Vocab.json"
        return candidate if candidate.exists() else None

    def get_name(self, cui: str) -> str:
        """Return human-readable name for a CUI, or the CUI itself if unknown."""
        return self.cui_names.get(cui, cui)

    def encode(self, text: str) -> np.ndarray:
        """Encode text to a 768-dim vector using SapBERT.

        SapBERT is trained on lowercased UMLS concept names, so input
        is lowercased before encoding for consistent results.

        Args:
            text: Natural language query (e.g. "my head hurts")

        Returns:
            L2-normalized float32 vector of shape (768,)
        """
        text = text.lower().strip()
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            output = self.model(**tokens)
        vec = output.pooler_output.squeeze().detach().cpu().numpy().astype(np.float32)

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Search the FAISS index for nearest CUIs.

        Args:
            query_vec: L2-normalized 768-dim vector
            top_k: Number of results to return

        Returns:
            List of (cui, cosine_similarity_score) tuples, descending by score.
        """
        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.cui_list[idx], float(score)))
        return results
