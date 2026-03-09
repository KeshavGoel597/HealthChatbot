"""
Graph-Augmented Semantic RAG for Electronic Medical Records (EMR).

Pipeline:
  1. Entity Extraction + Query Encoding (SapBERT)
  2. CUI Retrieval (keyword + FAISS + SapBERT re-ranking)
  3. Graph Expansion (SNOMED knowledge graph, BFS with ISA penalty)
  4. Path Ranking (semantic + relation + length + taxonomy penalty)
  5. EMR Parsing (Ruby-style JSON → Python dict)
  6. EMR Concept Matching (SapBERT + graph intersection, hard filter)
  7. Temporal Filtering (evidence persistence windows)
  8. Evidence Ranking (text_sim + graph_support + concept_overlap + lab_abnormality + recency)
  9. Prompt Assembly (structured prompt for downstream LLM)

Improvements (v2):
  - Medical entity extraction from natural language queries
  - Stronger semantic CUI filtering (legal/admin/death/organism)
  - ISA relations penalised in expansion and ranking
  - Evidence must have graph connection (hard filter)
  - Concept overlap score in evidence ranking
  - Aggressive caching (graph, indices, embeddings, pipeline results)
  - Per-query performance target < 0.5 s (warm cache)
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 768

# Phase 2
TOP_K_SEED_CUIS = 10

# Phase 3
MAX_HOPS = 2
MAX_NEIGHBORS_PER_NODE = 10
MAX_NODES_TOTAL = 100
MAX_PATHS = 100
# Max fraction of edges in a single BFS expansion that may be ISA
MAX_ISA_FRACTION = 0.3
# Maximum ISA-only consecutive hops before we stop following that branch
MAX_ISA_CONSECUTIVE = 1

# Clinical relations (high value)
CLINICAL_RELATIONS = {
    "cause_of",
    "due to",
    "causative agent of",
    "has definitional manifestation",
    "definitional manifestation of",
    "has pathological process",
    "pathological process of",
    "has associated finding",
    "associated finding of",
    "has focus",
    "focus of",
    "has direct morphology",
    "associated morphology of",
    "has associated morphology",
    "interprets",
    "is interpreted by",
    "has disposition",
    "disposition of",
    "has finding context",
    "has occurrence",
    "same as",
    "possibly equivalent to",
    "manifestation_of",
}

# Taxonomy relation – allowed but heavily penalised
TAXONOMY_RELATIONS = {"isa"}

# Combined allowed set
ALLOWED_RELATIONS = CLINICAL_RELATIONS | TAXONOMY_RELATIONS

WEAK_RELATIONS = {
    "has finding site",
    "finding site of",
    "has procedure site",
    "procedure site of",
    "has part",
    "part of",
    "has entire anatomy structure",
    "direct procedure site of",
    "has direct procedure site",
    "location_of",
    "finding_site",
}

# Phase 4 – relation weights for scoring (ISA heavily down-weighted)
RELATION_WEIGHT_MAP = {
    "cause_of": 1.0,
    "due to": 1.0,
    "causative agent of": 0.95,
    "has definitional manifestation": 0.9,
    "definitional manifestation of": 0.9,
    "has pathological process": 0.85,
    "pathological process of": 0.85,
    "has associated finding": 0.8,
    "associated finding of": 0.8,
    "has focus": 0.8,
    "focus of": 0.8,
    "same as": 0.65,
    "possibly equivalent to": 0.6,
    "has associated morphology": 0.55,
    "associated morphology of": 0.55,
    "has direct morphology": 0.55,
    "interprets": 0.5,
    "is interpreted by": 0.5,
    "has disposition": 0.5,
    "disposition of": 0.5,
    "has finding context": 0.45,
    "has occurrence": 0.4,
    "isa": 0.15,                    # heavily penalised (was 0.7)
}

TOP_K_PATHS = 10

# Phase 7 – temporal persistence (days). None = infinite.
PERSISTENCE_WINDOWS: Dict[str, Optional[int]] = {
    "symptom": 30,
    "lab": 14,
    "procedure": 365,
    "diagnosis": 365,
    "comorbidity": None,
    "medicine": 365,
}

# Phase 8 – evidence scoring weights (v2: added concept_overlap)
EVIDENCE_WEIGHTS = {
    "text_sim": 0.30,
    "graph_support": 0.25,
    "concept_overlap": 0.20,
    "lab_abnormality": 0.15,
    "recency": 0.10,
}
TOP_K_EVIDENCE = 10

# Lab abnormality thresholds: test_name_lower → (comparator, threshold)
LAB_THRESHOLDS: Dict[str, Tuple[str, float]] = {
    "total wbc count": (">", 11000),
    "wbc": (">", 11000),
    "platelet count": ("<", 150),
    "platelet": ("<", 150),
    "rbs": (">", 200),
    "sodium": ("<", 135),
    "potassium": ("<", 3.5),
    "hemoglobin": ("<", 12.0),
    "creatinine": (">", 1.2),
    "sgpt": (">", 40),
    "sgot": (">", 40),
    "alanine transaminase": (">", 40),
    "aspartate aminotransferase": (">", 40),
    "alkaline phosphatase": (">", 120),
    "bilirubin": (">", 1.2),
    "urea": (">", 40),
    "blood urea nitrogen": (">", 20),
    "esr": (">", 20),
}

# Semantic-type keywords used to DISCARD irrelevant CUIs
_DISCARD_KEYWORDS = {
    # Molecular biology
    "gene", "genes", "amino acid", "nucleotide", "protein", "rna", "dna",
    "genome", "allele", "locus", "codon",
    # Organisms & microbiology
    "organism", "bacteria", "virus", "fungus", "parasite",
    "cell", "cells", "cell structure",
    # Chemistry
    "chemical", "element", "ion", "compound", "reagent",
    # Legal / administrative / classification
    "mha", "mental health act", "act 1983", "act 2005",
    "legal", "administrative", "classification",
    "death certification", "cause of death",
    "social context", "regime", "government",
    # Geographic / ethnic
    "geographic", "country", "ethnic", "racial",
    "population group",
    # Veterinary
    "veterinary", "animal",
}

# Semantic-type keywords used to KEEP relevant CUIs
_KEEP_KEYWORDS = {
    "disease", "disorder", "finding", "symptom", "procedure",
    "drug", "tablet", "capsule", "injection", "vaccine",
    "syndrome", "infection", "condition", "diagnosis",
    "medication", "therapy", "treatment",
    "clinical finding", "observable entity",
    "body structure", "morphologic abnormality",
    "pharmaceutical", "substance",
    "headache", "pain", "fever", "cough", "nausea",
}

# Stopwords for entity extraction
_QUERY_STOPWORDS = {
    "i", "am", "is", "are", "was", "were", "be", "been", "being",
    "a", "an", "the", "my", "me", "we", "our", "it", "its",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "but", "and", "or", "nor", "not", "no", "so", "if", "then",
    "than", "too", "very", "just", "only", "also", "still",
    "from", "since", "for", "with", "without", "about", "between",
    "through", "during", "before", "after", "above", "below",
    "to", "at", "by", "on", "in", "of", "up", "out", "off",
    "that", "this", "these", "those", "which", "who", "whom",
    "what", "where", "when", "how", "why",
    "suffer", "suffering", "suffered", "feel", "feeling",
    "experience", "experiencing", "experienced",
    "taken", "take", "taking", "took", "doesn't", "don't",
    "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "won't", "couldn't", "shouldn't",
    "morning", "evening", "night", "day", "today", "yesterday",
    "week", "month", "year", "ago", "last", "past",
    "improve", "improves", "improved", "improving",
    "help", "helps", "helped", "helping",
    "get", "gets", "got", "getting", "better", "worse",
    "lot", "much", "many", "some", "any", "every", "all",
    "patient", "doctor", "hospital",
    "doesn", "don", "didn", "isn", "aren", "wasn", "weren",
    "hasn", "haven", "won", "couldn", "shouldn", "wouldn",
    "tell", "know", "show", "give", "need", "want", "like",
    "please", "thanks", "thank", "hello", "hi",
    "really", "quite", "rather", "seem", "seems",
}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1a — Medical Entity Extraction
# ═══════════════════════════════════════════════════════════════════════════
def extract_medical_entities(query: str) -> List[str]:
    """
    Extract medical entity phrases from a natural language query.

    Uses lightweight NP-chunking heuristics (no dependency on SciSpacy)
    to pull out clinically relevant noun phrases while discarding
    conversational filler.

    Returns a list of cleaned entity strings, e.g.
    "I am suffering from a severe headache since the morning"
      → ["severe headache"]
    """
    # Lowercase, remove punctuation except hyphens
    q = query.lower()
    q = re.sub(r"[^\w\s\-]", " ", q)
    tokens = q.split()

    # Remove stopwords to get candidate medical tokens
    medical_tokens = [t for t in tokens if t not in _QUERY_STOPWORDS and len(t) > 1]

    if not medical_tokens:
        # Fallback: return the original query cleaned up
        return [re.sub(r"\s+", " ", query.strip())]

    # Reconstruct contiguous phrases from original token order
    phrases: List[str] = []
    current_phrase: List[str] = []
    for tok in tokens:
        if tok in medical_tokens:
            current_phrase.append(tok)
        else:
            if current_phrase:
                phrases.append(" ".join(current_phrase))
                current_phrase = []
    if current_phrase:
        phrases.append(" ".join(current_phrase))

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique if unique else [re.sub(r"\s+", " ", query.strip())]


# ═══════════════════════════════════════════════════════════════════════════
# Helper — SapBERT encoder (lazy-loaded singleton)
# ═══════════════════════════════════════════════════════════════════════════
class _SapBERTEncoder:
    """Lazy-loaded SapBERT encoder singleton."""

    _instance: Optional["_SapBERTEncoder"] = None

    def __init__(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        import torch

        model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self._torch = torch

    @classmethod
    def get(cls) -> "_SapBERTEncoder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts → (N, 768) float32 numpy array, L2-normalised."""
        torch = self._torch
        toks = self.tokenizer(
            texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
        )
        with torch.no_grad():
            out = self.model(**toks)
        # CLS pooling
        vecs = out.last_hidden_state[:, 0, :].numpy().astype(np.float32)
        # L2 normalise
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vecs = vecs / norms
        return vecs

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text → (768,) float32 numpy array, L2-normalised."""
        return self.encode([text])[0]


# ═══════════════════════════════════════════════════════════════════════════
# Helper — FAISS index over CUI embeddings (lazy-loaded singleton)
# ═══════════════════════════════════════════════════════════════════════════
class _CUIIndex:
    """
    Hybrid CUI index: keyword candidate generation + SapBERT re-ranking.

    Strategy:
      1. **Keyword candidate generation** – search the CUI name catalogue
         for CUIs whose preferred name contains query keywords.  This is a
         fast in-memory substring search that returns highly relevant
         candidates because medical concept names are descriptive.
      2. **FAISS coarse retrieval** – broadens the pool using normalised
         graph embeddings (different vector space, but adds diversity).
      3. **SapBERT re-ranking** – all candidates are scored by cosine
         similarity between the SapBERT-encoded query and CUI name.

    This avoids encoding 407k CUI names at startup while still producing
    highly relevant seed CUIs.
    """

    _instance: Optional["_CUIIndex"] = None

    def __init__(self, cui_emb_path: str, cui_name_path: str) -> None:
        # ── Load CUI name catalogue ──
        with open(cui_name_path, "rb") as f:
            self.cui_names: Dict[str, List[List[str]]] = pickle.load(f)

        # ── Build keyword lookup: lower-cased preferred name → CUI ──
        self._name_to_cui: Dict[str, str] = {}
        self._cui_pref_name: Dict[str, str] = {}
        # Inverted token index: token → set of CUIs
        self._token_to_cuis: Dict[str, Set[str]] = defaultdict(set)
        for cui, entries in self.cui_names.items():
            pname = self._preferred_name(entries)
            self._cui_pref_name[cui] = pname
            if pname:
                self._name_to_cui[pname.lower()] = cui
                # Index tokens for fast keyword lookup
                for tok in re.split(r"\W+", pname.lower()):
                    if tok and len(tok) > 2:
                        self._token_to_cuis[tok].add(cui)

        # ── Load & normalise graph embeddings ──
        with open(cui_emb_path, "rb") as f:
            raw_emb: Dict[str, np.ndarray] = pickle.load(f)

        self.cuis: List[str] = list(raw_emb.keys())
        mat = np.vstack(
            [raw_emb[c].reshape(1, EMBEDDING_DIM) for c in self.cuis]
        ).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        mat = mat / norms
        self._graph_mat = mat

        self._cui_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.cuis)}

        # Build FAISS index over graph embeddings (coarse retrieval)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(mat)

        # Cache for SapBERT-encoded CUI names (populated on demand)
        self._name_emb_cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _preferred_name(entries: List[List[str]]) -> str:
        """Pick the shortest non-parenthesised name, or just the shortest."""
        names = [e[1] for e in entries if e[1]]
        if not names:
            return ""
        clean = [n for n in names if "(" not in n]
        pool = clean if clean else names
        return min(pool, key=len)

    @classmethod
    def get(cls, cui_emb_path: str, cui_name_path: str) -> "_CUIIndex":
        if cls._instance is None:
            cls._instance = cls(cui_emb_path, cui_name_path)
        return cls._instance

    # ------------------------------------------------------------------
    def _keyword_candidates(self, query: str, max_candidates: int = 200) -> List[str]:
        """
        Find CUIs whose preferred name contains query keywords using
        an inverted token index.  Boost CUIs matching multiple tokens.
        Also do exact substring matching on CUI names for multi-word phrases.
        """
        stopwords = {
            "the", "a", "an", "of", "for", "in", "and", "or", "is", "are",
            "to", "with", "results", "test", "tests", "what", "how", "my",
            "severe", "mild", "moderate", "acute", "chronic",  # severity modifiers
        }
        tokens = [
            t.lower()
            for t in re.split(r"\W+", query)
            if t and t.lower() not in stopwords and len(t) > 2
        ]
        if not tokens:
            # If all tokens were stopwords, try without severity filter
            tokens = [
                t.lower()
                for t in re.split(r"\W+", query)
                if t and len(t) > 2
            ]
        if not tokens:
            return []

        # Use inverted index for O(1) per token lookup
        cui_counts: Dict[str, int] = defaultdict(int)
        for tok in tokens:
            for cui in self._token_to_cuis.get(tok, set()):
                if cui in self._cui_to_idx:  # must have graph embedding
                    cui_counts[cui] += 1

        # Also do exact substring match on the full query (case-insensitive)
        query_lower = query.lower().strip()
        for name_lower, cui in self._name_to_cui.items():
            if query_lower in name_lower or name_lower in query_lower:
                cui_counts[cui] = cui_counts.get(cui, 0) + 10  # big boost

        # Sort by match count descending
        scored = sorted(cui_counts.items(), key=lambda x: x[1], reverse=True)
        return [cui for cui, _ in scored[:max_candidates]]

    def _encode_cui_names(self, cuis: List[str]) -> Dict[str, np.ndarray]:
        """Batch-encode CUI preferred names via SapBERT (with caching)."""
        enc = _SapBERTEncoder.get()
        to_encode: List[Tuple[int, str, str]] = []
        for i, cui in enumerate(cuis):
            if cui not in self._name_emb_cache:
                name = self._cui_pref_name.get(cui, cui)
                to_encode.append((i, cui, name))

        if to_encode:
            # Batch in chunks of 256 for memory efficiency
            for start in range(0, len(to_encode), 256):
                batch = to_encode[start : start + 256]
                names = [t[2] for t in batch]
                vecs = enc.encode(names)
                for (_, cui, _), vec in zip(batch, vecs):
                    self._name_emb_cache[cui] = vec

        return {cui: self._name_emb_cache[cui] for cui in cuis if cui in self._name_emb_cache}

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = TOP_K_SEED_CUIS,
        query_text: str = "",
    ) -> List[Tuple[str, float]]:
        """
        Three-stage search:
          1. Keyword candidate generation from CUI name catalogue
          2. FAISS coarse retrieval over graph embeddings
          3. SapBERT re-ranking of the union
        Returns [(cui, score), …] sorted descending by cosine sim.
        """
        # Stage 1: keyword candidates
        kw_cuis = self._keyword_candidates(query_text, max_candidates=200) if query_text else []

        # Stage 2: FAISS coarse candidates
        coarse_k = max(top_k * 10, 100)
        qv = query_vec.reshape(1, EMBEDDING_DIM).astype(np.float32)
        _, idxs = self.index.search(qv, coarse_k)
        faiss_cuis = [self.cuis[idx] for idx in idxs[0] if idx >= 0]

        # Union (preserve order, keyword candidates first)
        seen: Set[str] = set()
        candidates: List[str] = []
        for cui in kw_cuis + faiss_cuis:
            if cui not in seen:
                seen.add(cui)
                candidates.append(cui)
            if len(candidates) >= 250:
                break

        # Stage 3: SapBERT re-ranking
        name_embs = self._encode_cui_names(candidates)
        scored: List[Tuple[str, float]] = []
        for cui in candidates:
            emb = name_embs.get(cui)
            if emb is not None:
                sim = float(np.dot(query_vec, emb))
                scored.append((cui, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_embedding(self, cui: str) -> Optional[np.ndarray]:
        """Return the SapBERT name embedding for a CUI (on-demand encoding)."""
        if cui in self._name_emb_cache:
            return self._name_emb_cache[cui]
        embs = self._encode_cui_names([cui])
        return embs.get(cui)

    def get_graph_embedding(self, cui: str) -> Optional[np.ndarray]:
        """Return normalised graph embedding (from precomputed file)."""
        idx = self._cui_to_idx.get(cui)
        if idx is None:
            return None
        return self._graph_mat[idx]

    def cui_preferred_name(self, cui: str) -> str:
        """Return shortest English name for a CUI."""
        return self._cui_pref_name.get(cui, cui)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Entity Extraction + Query Encoding
# ═══════════════════════════════════════════════════════════════════════════
def encode_query(query: str) -> Tuple[np.ndarray, List[str], List[np.ndarray]]:
    """
    Extract medical entities from the query, then encode each entity and
    the full query with SapBERT.

    Returns:
        query_vec       – (768,) pooled query vector (mean of entity vecs)
        entities        – list of extracted entity strings
        entity_vecs     – list of per-entity (768,) vectors
    """
    enc = _SapBERTEncoder.get()
    entities = extract_medical_entities(query)

    # Encode each entity separately for precise CUI matching
    entity_vecs = [enc.encode_single(e) for e in entities]

    # Pool: mean of entity vectors (better than encoding the full noisy sentence)
    if entity_vecs:
        pooled = np.mean(entity_vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(pooled)
        if norm > 1e-10:
            pooled = pooled / norm
    else:
        pooled = enc.encode_single(query)

    return pooled, entities, entity_vecs


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — CUI Retrieval + Semantic Filtering
# ═══════════════════════════════════════════════════════════════════════════
def _should_keep_cui(cui: str, cui_index: _CUIIndex) -> bool:
    """
    Heuristic filter: keep clinically relevant CUIs, discard
    genes/chemicals/organisms/legal/administrative concepts.
    """
    entries = cui_index.cui_names.get(cui, [])
    if not entries:
        return True  # no name info → keep by default
    full_text = " ".join(e[1] for e in entries).lower()

    # Hard-discard patterns (never rescued by keep keywords)
    _HARD_DISCARD = {
        "mha", "mental health act", "act 1983", "act 2005",
        "death certification", "cause of death",
        "regime", "government", "legal status",
        "population group", "ethnic group",
        "veterinary",
    }
    for hd in _HARD_DISCARD:
        if hd in full_text:
            return False

    for kw in _DISCARD_KEYWORDS:
        if kw in full_text:
            # But if it also matches a keep keyword, still keep
            for kk in _KEEP_KEYWORDS:
                if kk in full_text:
                    return True
            return False
    return True


def retrieve_seed_cuis(
    query_vec: np.ndarray,
    cui_emb_path: str,
    cui_name_path: str,
    entities: Optional[List[str]] = None,
    entity_vecs: Optional[List[np.ndarray]] = None,
    query_text: str = "",
    top_k: int = TOP_K_SEED_CUIS,
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k CUIs by searching each extracted entity separately,
    then merging and de-duplicating results.

    This gives far better precision than embedding the entire raw query.
    """
    ci = _CUIIndex.get(cui_emb_path, cui_name_path)

    # If entities were extracted, search per-entity and merge
    all_hits: Dict[str, float] = {}  # cui → best score
    if entities and entity_vecs:
        for ent_text, ent_vec in zip(entities, entity_vecs):
            raw = ci.search(ent_vec, top_k=top_k * 2, query_text=ent_text)
            for cui, score in raw:
                if cui not in all_hits or score > all_hits[cui]:
                    all_hits[cui] = score
    else:
        # Fallback: search with full query
        raw = ci.search(query_vec, top_k=top_k * 3, query_text=query_text)
        for cui, score in raw:
            all_hits[cui] = score

    # Filter and sort
    filtered: List[Tuple[str, float]] = []
    for cui, score in sorted(all_hits.items(), key=lambda x: x[1], reverse=True):
        if _should_keep_cui(cui, ci):
            filtered.append((cui, score))
            if len(filtered) >= top_k:
                break
    return filtered


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Graph Expansion (BFS with ISA throttling)
# ═══════════════════════════════════════════════════════════════════════════
def _is_allowed_relation(label: str) -> bool:
    label_l = label.lower().strip()
    if label_l in WEAK_RELATIONS:
        return False
    if label_l in ALLOWED_RELATIONS:
        return True
    # Fallback: allow unknown relations that aren't explicitly weak
    return True


def _is_taxonomy_relation(label: str) -> bool:
    return label.lower().strip() in TAXONOMY_RELATIONS


def expand_graph(
    seed_cuis: List[str],
    graph: nx.DiGraph,
    max_hops: int = MAX_HOPS,
    max_neighbors: int = MAX_NEIGHBORS_PER_NODE,
    max_nodes: int = MAX_NODES_TOTAL,
    max_paths: int = MAX_PATHS,
) -> Tuple[Set[str], List[Dict[str, Any]]]:
    """
    BFS expansion around seed CUIs with ISA throttling.

    ISA edges are limited per node (MAX_ISA_FRACTION of neighbours) and
    consecutive ISA-only chains are capped at MAX_ISA_CONSECUTIVE.

    Returns:
        expanded_cuis  – set of all visited CUI strings
        paths          – list of {"nodes": [...], "relations": [...]} dicts
    """
    visited: Set[str] = set(seed_cuis)
    # parent tracking: cui -> (parent_cui, relation_label, consecutive_isa_count)
    parent: Dict[str, Tuple[Optional[str], Optional[str], int]] = {
        c: (None, None, 0) for c in seed_cuis
    }
    queue: deque[Tuple[str, int, int]] = deque(
        (c, 0, 0) for c in seed_cuis  # (cui, depth, consecutive_isa)
    )

    while queue and len(visited) < max_nodes:
        node, depth, isa_chain = queue.popleft()
        if depth >= max_hops:
            continue
        if node not in graph:
            continue

        # Collect all valid outgoing + incoming edges
        candidate_edges: List[Tuple[str, str]] = []  # (neighbour, label)
        for _, nbr, data in graph.edges(node, data=True):
            label = data.get("label", "")
            if label == "self":
                continue
            if not _is_allowed_relation(label):
                continue
            if nbr not in visited:
                candidate_edges.append((nbr, label))

        for src, _, data in graph.in_edges(node, data=True):
            label = data.get("label", "")
            if label == "self":
                continue
            if not _is_allowed_relation(label):
                continue
            if src not in visited:
                candidate_edges.append((src, label))

        # Partition into clinical vs taxonomy (ISA)
        clinical_edges = [(n, l) for n, l in candidate_edges if not _is_taxonomy_relation(l)]
        isa_edges = [(n, l) for n, l in candidate_edges if _is_taxonomy_relation(l)]

        # Prioritise clinical edges; limit ISA edges
        max_isa = max(1, int(max_neighbors * MAX_ISA_FRACTION))
        edges_to_follow = clinical_edges[:max_neighbors]
        remaining = max_neighbors - len(edges_to_follow)
        if remaining > 0 and isa_chain < MAX_ISA_CONSECUTIVE:
            edges_to_follow.extend(isa_edges[:min(remaining, max_isa)])

        for nbr, label in edges_to_follow:
            if len(visited) >= max_nodes:
                break
            if nbr in visited:
                continue
            new_isa_chain = (isa_chain + 1) if _is_taxonomy_relation(label) else 0
            visited.add(nbr)
            parent[nbr] = (node, label, new_isa_chain)
            queue.append((nbr, depth + 1, new_isa_chain))

    # Reconstruct paths from every non-seed node back to its seed
    paths: List[Dict[str, Any]] = []
    for node in visited:
        if node in seed_cuis:
            continue
        nodes_rev: List[str] = [node]
        rels_rev: List[str] = []
        cur = node
        seen_in_path: Set[str] = {cur}
        while parent.get(cur, (None, None, 0))[0] is not None:
            p, rel, _ = parent[cur]
            if p in seen_in_path:
                break
            nodes_rev.append(p)
            rels_rev.append(rel)
            seen_in_path.add(p)
            cur = p
        nodes_path = list(reversed(nodes_rev))
        rels_path = list(reversed(rels_rev))
        if len(nodes_path) >= 2:
            paths.append({"nodes": nodes_path, "relations": rels_path})
            if len(paths) >= max_paths:
                break

    return visited, paths


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Path Ranking (with taxonomy penalty)
# ═══════════════════════════════════════════════════════════════════════════
def rank_paths(
    paths: List[Dict[str, Any]],
    query_vec: np.ndarray,
    cui_index: _CUIIndex,
    top_k: int = TOP_K_PATHS,
) -> List[Dict[str, Any]]:
    """Score and rank reasoning paths. Penalises ISA-heavy paths."""

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for p in paths:
        nodes = p["nodes"]
        rels = p["relations"]

        # --- semantic similarity of path nodes to query ---
        sims: List[float] = []
        for cui in nodes:
            emb = cui_index.get_embedding(cui)
            if emb is not None:
                sim = float(np.dot(query_vec, emb))
                sims.append(sim)
        avg_sim = float(np.mean(sims)) if sims else 0.0

        # --- relation weight ---
        rel_scores = [RELATION_WEIGHT_MAP.get(r.lower().strip(), 0.3) for r in rels]
        avg_rel = float(np.mean(rel_scores)) if rel_scores else 0.0

        # --- path length penalty (shorter = better) ---
        length_penalty = 1.0 / len(nodes)

        # --- taxonomy penalty: fraction of edges that are ISA ---
        if rels:
            isa_frac = sum(1 for r in rels if r.lower().strip() in TAXONOMY_RELATIONS) / len(rels)
        else:
            isa_frac = 0.0
        taxonomy_penalty = 1.0 - (0.5 * isa_frac)  # up to 50% reduction for all-ISA paths

        score = (
            0.45 * avg_sim
            + 0.30 * avg_rel
            + 0.10 * length_penalty
            + 0.15 * taxonomy_penalty
        )
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, p in scored[:top_k]:
        p_copy = dict(p)
        p_copy["score"] = round(score, 4)
        results.append(p_copy)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — EMR Parsing (Ruby-style JSON → Python dict)
# ═══════════════════════════════════════════════════════════════════════════
def _ruby_json_to_python(raw: str) -> dict:
    """
    Convert Ruby-style EMR text into a valid Python dictionary.

    Handles:
    - ``=>`` → ``:``
    - Unquoted top-level keys (``age:``, ``lab_data:``, etc.)
    - The leading ``Patient data:`` prefix
    """
    text = raw.strip()

    # Remove leading "Patient data:" if present
    if text.lower().startswith("patient data:"):
        text = text[len("patient data:"):].strip()

    # Replace => with :
    text = text.replace("=>", ":")

    # Quote bare symbol-style keys at the top level, e.g.  {age:  or , sex:
    # Pattern: word characters followed by colon that are NOT inside quotes
    text = re.sub(r'(?<=[{,])\s*([a-zA-Z_]\w*)\s*:', r' "\1":', text)

    # Also handle the very first key after opening brace if regex missed it
    text = re.sub(r'^\{\s*([a-zA-Z_]\w*)\s*:', r'{"\1":', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: more aggressive quoting
        # Quote any bare key that looks like  ident:
        text2 = re.sub(r'(?<!["\w])([a-zA-Z_]\w*)(?=\s*:)', r'"\1"', text)
        try:
            return json.loads(text2)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse EMR data: {e}\n---\n{text2[:500]}")


_NUMERIC_RE = re.compile(r"([\d]+\.?[\d]*)")


def _extract_numeric(value: str) -> Optional[float]:
    """Extract the first numeric value from a string like '12.9 sec' → 12.9."""
    m = _NUMERIC_RE.search(value)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _is_url(value: str) -> bool:
    return value.strip().startswith("http://") or value.strip().startswith("https://")


def _is_empty_or_dot(value: str) -> bool:
    v = value.strip()
    return v == "" or v == "."


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse dates like '04-Oct-2024'."""
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


# Evidence item schema
def _make_evidence(
    text: str,
    etype: str,
    date: Optional[datetime] = None,
    numeric_value: Optional[float] = None,
    source_field: str = "",
) -> Dict[str, Any]:
    return {
        "text": text.strip(),
        "type": etype,
        "date": date,
        "numeric_value": numeric_value,
        "source_field": source_field,
    }


def parse_emr(raw_text: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Parse Ruby-style EMR JSON.

    Returns:
        emr_dict      – the parsed dict
        evidence_items – flat list of evidence dicts
    """
    emr = _ruby_json_to_python(raw_text)
    evidence: List[Dict[str, Any]] = []
    seen_texts: Set[str] = set()  # deduplication

    def _add(text: str, etype: str, date: Optional[datetime], numeric: Optional[float] = None, src: str = ""):
        key = text.strip().lower()
        if key in seen_texts:
            return
        seen_texts.add(key)
        evidence.append(_make_evidence(text, etype, date, numeric, src))

    # --- lab_data ---
    for lab in emr.get("lab_data", []):
        name = lab.get("name", "")
        value = str(lab.get("value", ""))
        date = _parse_date(lab.get("date", "")) if lab.get("date") else None
        if _is_empty_or_dot(value) or _is_url(value):
            continue
        numeric = _extract_numeric(value)
        combined = f"{name}: {value}"
        _add(combined, "lab", date, numeric, "lab_data")

    # --- prescriptions ---
    for rx in emr.get("prescriptions", []):
        name = rx.get("name", "")
        date = _parse_date(rx.get("date", "")) if rx.get("date") else None

        # Medicines (no "name" key, just "medicine")
        if "medicine" in rx:
            med = rx["medicine"]
            _add(med, "medicine", date, src="medicine")
            continue

        value = rx.get("value", "")

        # Nested lists: Symptoms, Diagnosis, Comorbidity, Reason for Admission
        if isinstance(value, list):
            name_lower = name.lower()
            for item in value:
                if isinstance(item, dict):
                    if "sym" in item:
                        sym = item["sym"].strip()
                        if sym and not _is_empty_or_dot(sym):
                            _add(sym, "symptom", date, src=name)
                    if "diag" in item:
                        diag = item["diag"].strip()
                        if diag and not _is_empty_or_dot(diag) and not diag.startswith("@"):
                            etype = "comorbidity" if "comorbid" in name_lower else "diagnosis"
                            _add(diag, etype, date, src=name)
            continue

        # Scalar string values
        if isinstance(value, str):
            if _is_empty_or_dot(value) or _is_url(value):
                continue
            name_lower = name.lower()
            # Skip non-clinical fields
            if name_lower in (
                "consultationtype", "nextvisitmode", "nextvisit",
                "prescription picture", "blood pressure",
                "recommendedlabs",
            ):
                continue
            if "patient history" in name_lower:
                _add(value, "symptom", date, src=name)
            elif "comment" in name_lower:
                _add(value, "procedure", date, src=name)
            else:
                _add(f"{name}: {value}", "symptom", date, src=name)

    return emr, evidence


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5b — Path-Grounded EMR Record Retrieval
# ═══════════════════════════════════════════════════════════════════════════
def _flatten_emr_records(emr_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten the parsed EMR dict into a list of individual raw record objects.
    Each record gets a ``_record_type`` tag and a searchable ``_text`` field.
    """
    records: List[Dict[str, Any]] = []
    seen_texts: Set[str] = set()

    def _dedup_add(rec: Dict[str, Any], text: str) -> None:
        key = text.strip().lower()
        if key and key not in seen_texts:
            seen_texts.add(key)
            rec["_text"] = text.strip()
            records.append(rec)

    # Labs
    for lab in emr_dict.get("lab_data", []):
        name = lab.get("name", "")
        value = str(lab.get("value", ""))
        if not name or not value or value.strip() in ("", "."):
            continue
        rec = dict(lab)
        rec["_record_type"] = "lab"
        _dedup_add(rec, f"{name}: {value}")

    # Prescriptions — heterogeneous list
    for rx in emr_dict.get("prescriptions", []):
        # Medicine entries
        if "medicine" in rx:
            rec = dict(rx)
            rec["_record_type"] = "medicine"
            _dedup_add(rec, rx["medicine"])
            continue

        name = rx.get("name", "")
        value = rx.get("value", "")
        name_lower = name.lower()

        # Skip non-clinical fields
        if name_lower in (
            "consultationtype", "nextvisitmode", "nextvisit",
            "prescription picture", "recommendedlabs",
        ):
            continue

        # Nested list values (Symptoms, Diagnosis, Comorbidity)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    if "sym" in item and item["sym"].strip():
                        rec = dict(item)
                        rec["_record_type"] = "symptom"
                        rec["_source_field"] = name
                        rec["_date"] = rx.get("date", "")
                        _dedup_add(rec, item["sym"])
                    if "diag" in item and item["diag"].strip():
                        rec = dict(item)
                        rtype = "comorbidity" if "comorbid" in name_lower else "diagnosis"
                        rec["_record_type"] = rtype
                        rec["_source_field"] = name
                        rec["_date"] = rx.get("date", "")
                        _dedup_add(rec, item["diag"])
            continue

        # Scalar string values
        if isinstance(value, str) and value.strip() and value.strip() != ".":
            if value.startswith("http"):
                continue
            rec = dict(rx)
            if "patient history" in name_lower:
                rec["_record_type"] = "history"
            elif "comment" in name_lower:
                rec["_record_type"] = "procedure_note"
            else:
                rec["_record_type"] = "clinical_note"
            _dedup_add(rec, f"{name}: {value}" if name else value)

    return records


def retrieve_path_grounded_records(
    ranked_paths: List[Dict[str, Any]],
    emr_dict: Dict[str, Any],
    cui_emb_path: str,
    cui_name_path: str,
    sim_threshold: float = 0.35,
    max_records_per_path: int = 5,
) -> List[Dict[str, Any]]:
    """
    For each ranked reasoning path, find raw EMR JSON objects whose text
    is semantically similar to the CUI concept names in that path.

    Returns a list of dicts, one per path:
        {
            "path_index": int,
            "path_summary": str,           # human-readable path
            "grounded_records": [           # matched raw EMR objects
                {"record": <raw JSON obj>, "matched_concept": str, "similarity": float},
                …
            ]
        }
    """
    enc = _SapBERTEncoder.get()
    ci = _CUIIndex.get(cui_emb_path, cui_name_path)

    # Flatten EMR into searchable records
    flat_records = _flatten_emr_records(emr_dict)
    if not flat_records:
        return []

    # Batch-encode all record texts
    record_texts = [r["_text"] for r in flat_records]
    record_vecs = enc.encode(record_texts)  # (N, 768)

    # Build a small FAISS index over record embeddings
    record_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    record_index.add(record_vecs.astype(np.float32))

    results: List[Dict[str, Any]] = []
    for path_idx, path in enumerate(ranked_paths):
        nodes = path.get("nodes", [])
        rels = path.get("relations", [])

        # Collect all unique concept names from this path
        concept_names: List[Tuple[str, str]] = []  # (cui, name)
        for cui in nodes:
            name = ci.cui_preferred_name(cui)
            if name and name != cui:
                concept_names.append((cui, name))

        if not concept_names:
            results.append({
                "path_index": path_idx,
                "path_summary": _path_to_natural_language(path, ci),
                "grounded_records": [],
            })
            continue

        # Encode concept names and search against record index
        concept_texts = [name for _, name in concept_names]
        concept_vecs = enc.encode(concept_texts)  # (M, 768)

        # Search each concept against all records
        matched: List[Dict[str, Any]] = []
        seen_record_texts: Set[str] = set()

        for (cui, cname), cvec in zip(concept_names, concept_vecs):
            qv = cvec.reshape(1, EMBEDDING_DIM).astype(np.float32)
            k = min(max_records_per_path, len(flat_records))
            scores, idxs = record_index.search(qv, k)
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0 or score < sim_threshold:
                    continue
                rec = flat_records[idx]
                rec_text = rec["_text"]
                if rec_text in seen_record_texts:
                    continue
                seen_record_texts.add(rec_text)

                # Build a clean copy of the raw record (without internal fields)
                clean_rec = {k: v for k, v in rec.items() if not k.startswith("_")}
                clean_rec["record_type"] = rec["_record_type"]
                if "_date" in rec:
                    clean_rec["date"] = rec["_date"]

                matched.append({
                    "record": clean_rec,
                    "matched_concept": cname,
                    "similarity": round(float(score), 4),
                })

        # Sort by similarity, take top N
        matched.sort(key=lambda x: x["similarity"], reverse=True)
        matched = matched[:max_records_per_path]

        results.append({
            "path_index": path_idx,
            "path_summary": _path_to_natural_language(path, ci),
            "grounded_records": matched,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6 — EMR Concept Matching (hard filter)
# ═══════════════════════════════════════════════════════════════════════════
def match_emr_to_graph(
    evidence_items: List[Dict[str, Any]],
    expanded_cuis: Set[str],
    cui_emb_path: str,
    cui_name_path: str,
    query_entities: Optional[List[str]] = None,
    sim_threshold: float = 0.50,
    hard_filter: bool = True,
) -> List[Dict[str, Any]]:
    """
    Encode each evidence text with SapBERT, then check similarity against
    the expanded CUI set using a small local FAISS index.

    If hard_filter is True, evidence with NO graph connection is discarded
    unless it is a diagnosis/comorbidity/medicine (always kept).

    Returns evidence items augmented with ``matched_cuis``, ``text_embedding``,
    and ``concept_overlap``.
    """
    enc = _SapBERTEncoder.get()
    ci = _CUIIndex.get(cui_emb_path, cui_name_path)

    # Batch encode all evidence texts at once
    texts = [ev["text"] for ev in evidence_items]
    if not texts:
        return []
    embeddings = enc.encode(texts)  # (N, 768) normalised

    # Build a small FAISS index over expanded CUI name embeddings
    expanded_list = sorted(expanded_cuis)
    local_index = None
    exp_cui_names: List[str] = []
    if expanded_list:
        exp_embs = ci._encode_cui_names(expanded_list)
        exp_cuis_with_emb = [(c, exp_embs[c]) for c in expanded_list if c in exp_embs]
        if exp_cuis_with_emb:
            exp_cui_names = [c for c, _ in exp_cuis_with_emb]
            exp_mat = np.vstack([e.reshape(1, EMBEDDING_DIM) for _, e in exp_cuis_with_emb]).astype(np.float32)
            local_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            local_index.add(exp_mat)

    # Precompute query entity tokens for concept overlap scoring
    entity_tokens: Set[str] = set()
    if query_entities:
        for ent in query_entities:
            for tok in re.split(r"\W+", ent.lower()):
                if tok and len(tok) > 2:
                    entity_tokens.add(tok)

    # Types that bypass the hard filter (always clinically relevant)
    _ALWAYS_KEEP_TYPES = {"diagnosis", "comorbidity", "medicine"}

    matched: List[Dict[str, Any]] = []
    for i, ev in enumerate(evidence_items):
        vec = embeddings[i]
        matched_cuis: List[str] = []

        if local_index is not None and len(exp_cui_names) > 0:
            qv = vec.reshape(1, EMBEDDING_DIM).astype(np.float32)
            k = min(10, len(exp_cui_names))
            scores, idxs = local_index.search(qv, k)
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0:
                    continue
                if score < sim_threshold:
                    break
                matched_cuis.append(exp_cui_names[idx])

        graph_connected = len(matched_cuis) > 0

        # Concept overlap: fraction of query entity tokens found in evidence text
        concept_overlap = 0.0
        if entity_tokens:
            ev_tokens = set(re.split(r"\W+", ev["text"].lower()))
            overlap_count = len(entity_tokens & ev_tokens)
            concept_overlap = overlap_count / len(entity_tokens)

        # Hard filter: discard evidence with no graph connection
        # (unless it's a diagnosis/comorbidity/medicine or has high concept overlap)
        if hard_filter and not graph_connected:
            etype = ev.get("type", "")
            if etype not in _ALWAYS_KEEP_TYPES and concept_overlap < 0.3:
                continue

        ev_copy = dict(ev)
        ev_copy["matched_cuis"] = matched_cuis
        ev_copy["text_embedding"] = vec
        ev_copy["graph_connected"] = graph_connected
        ev_copy["concept_overlap"] = concept_overlap
        matched.append(ev_copy)

    return matched


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7 — Temporal Filtering
# ═══════════════════════════════════════════════════════════════════════════
def temporal_filter(
    evidence_items: List[Dict[str, Any]],
    reference_date: Optional[datetime] = None,
    expanded_cuis: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Filter evidence by temporal persistence windows.

    Old evidence is retained only if it has graph connections to the query.
    """
    if reference_date is None:
        reference_date = datetime.now()

    kept: List[Dict[str, Any]] = []
    for ev in evidence_items:
        etype = ev.get("type", "symptom")
        window_days = PERSISTENCE_WINDOWS.get(etype, 30)

        # Infinite window → always keep
        if window_days is None:
            kept.append(ev)
            continue

        ev_date = ev.get("date")
        if ev_date is None:
            # No date → keep conservatively
            kept.append(ev)
            continue

        cutoff = reference_date - timedelta(days=window_days)
        if ev_date >= cutoff:
            kept.append(ev)
        else:
            # Old evidence: keep only if connected to query in graph
            if ev.get("graph_connected", False):
                ev_copy = dict(ev)
                ev_copy["retained_reason"] = "graph_connected_despite_old"
                kept.append(ev_copy)

    return kept


# ═══════════════════════════════════════════════════════════════════════════
# Phase 8 — Evidence Ranking
# ═══════════════════════════════════════════════════════════════════════════
def _lab_abnormality_score(ev: Dict[str, Any]) -> float:
    """Return 1.0 if the lab value is abnormal according to thresholds, else 0."""
    if ev.get("type") != "lab":
        return 0.0
    text_lower = ev.get("text", "").lower()
    numeric = ev.get("numeric_value")
    if numeric is None:
        return 0.0
    for test_name, (comp, threshold) in LAB_THRESHOLDS.items():
        if test_name in text_lower:
            if comp == ">" and numeric > threshold:
                return 1.0
            if comp == "<" and numeric < threshold:
                return 1.0
    return 0.0


def _recency_score(ev: Dict[str, Any], reference_date: datetime) -> float:
    """Score in [0, 1] — more recent evidence scores higher."""
    ev_date = ev.get("date")
    if ev_date is None:
        return 0.5
    days_ago = (reference_date - ev_date).days
    if days_ago <= 0:
        return 1.0
    if days_ago >= 365:
        return 0.0
    return max(0.0, 1.0 - days_ago / 365.0)


def rank_evidence(
    evidence_items: List[Dict[str, Any]],
    query_vec: np.ndarray,
    reference_date: Optional[datetime] = None,
    top_k: int = TOP_K_EVIDENCE,
) -> List[Dict[str, Any]]:
    """Score and rank evidence items. Returns top-k with scores.
    
    Scoring formula (v2):
      0.30 text_sim + 0.25 graph_support + 0.20 concept_overlap
      + 0.15 lab_abnormality + 0.10 recency
    """
    if reference_date is None:
        reference_date = datetime.now()

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for ev in evidence_items:
        # Text similarity
        emb = ev.get("text_embedding")
        if emb is not None:
            text_sim = float(np.dot(query_vec, emb))
        else:
            text_sim = 0.0

        # Graph support
        graph_support = 1.0 if ev.get("graph_connected", False) else 0.0

        # Concept overlap (new in v2)
        concept_overlap = ev.get("concept_overlap", 0.0)

        # Lab abnormality
        lab_abn = _lab_abnormality_score(ev)

        # Recency
        recency = _recency_score(ev, reference_date)

        score = (
            EVIDENCE_WEIGHTS["text_sim"] * text_sim
            + EVIDENCE_WEIGHTS["graph_support"] * graph_support
            + EVIDENCE_WEIGHTS["concept_overlap"] * concept_overlap
            + EVIDENCE_WEIGHTS["lab_abnormality"] * lab_abn
            + EVIDENCE_WEIGHTS["recency"] * recency
        )
        ev_copy = dict(ev)
        ev_copy["score"] = round(score, 4)
        ev_copy["score_breakdown"] = {
            "text_sim": round(text_sim, 4),
            "graph_support": round(graph_support, 4),
            "concept_overlap": round(concept_overlap, 4),
            "lab_abnormality": round(lab_abn, 4),
            "recency": round(recency, 4),
        }
        scored.append((score, ev_copy))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ev for _, ev in scored[:top_k]]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 9 — Prompt Assembly
# ═══════════════════════════════════════════════════════════════════════════
def _path_to_natural_language(path: Dict[str, Any], cui_index: _CUIIndex) -> str:
    """Convert a reasoning path to a natural language sentence."""
    nodes = path["nodes"]
    rels = path["relations"]
    parts: List[str] = []
    for i, cui in enumerate(nodes):
        name = cui_index.cui_preferred_name(cui)
        parts.append(name)
        if i < len(rels):
            rel = rels[i].replace("_", " ")
            parts.append(f" --{rel}--> ")
    chain = "".join(parts)

    # Also produce a readable sentence
    if len(nodes) >= 2 and len(rels) >= 1:
        n0 = cui_index.cui_preferred_name(nodes[0])
        r0 = rels[0].replace("_", " ")
        n1 = cui_index.cui_preferred_name(nodes[1])
        sentence = f"{n0} {r0} {n1}."
        if len(nodes) >= 3 and len(rels) >= 2:
            n2 = cui_index.cui_preferred_name(nodes[2])
            r1 = rels[1].replace("_", " ")
            sentence += f" {n1} {r1} {n2}."
        return f"{chain}\n  → {sentence}"
    return chain


def assemble_prompt(
    query: str,
    evidence: List[Dict[str, Any]],
    paths: List[Dict[str, Any]],
    cui_emb_path: str,
    cui_name_path: str,
    path_grounded: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build a structured prompt for a downstream LLM."""
    ci = _CUIIndex.get(cui_emb_path, cui_name_path)

    sections: List[str] = []

    # Header
    sections.append("=== MEDICAL RAG PROMPT ===\n")
    sections.append(f"## Patient Query\n{query}\n")

    # Evidence
    sections.append("## Relevant Patient Evidence")
    if evidence:
        for i, ev in enumerate(evidence, 1):
            date_str = ev["date"].strftime("%d-%b-%Y") if ev.get("date") else "N/A"
            score = ev.get("score", 0)
            sections.append(
                f"  {i}. [{ev['type'].upper()}] {ev['text']}  "
                f"(date: {date_str}, relevance: {score:.3f})"
            )
    else:
        sections.append("  No relevant evidence found.")

    # Reasoning paths
    sections.append("\n## Medical Reasoning Paths")
    if paths:
        for i, p in enumerate(paths, 1):
            nl = _path_to_natural_language(p, ci)
            sections.append(f"  {i}. {nl}  (score: {p.get('score', 0):.3f})")
    else:
        sections.append("  No reasoning paths found.")

    # Path-Grounded Patient Records (new)
    if path_grounded:
        sections.append("\n## Path-Grounded Patient Records")
        sections.append(
            "  Raw EMR records matched to concepts in the reasoning paths above."
        )
        for pg in path_grounded:
            records = pg.get("grounded_records", [])
            if not records:
                continue
            path_num = pg["path_index"] + 1
            summary = pg.get("path_summary", "").split("\n")[0]  # first line only
            sections.append(f"\n  ### Path {path_num}: {summary}")
            for rec_info in records:
                rec = rec_info["record"]
                concept = rec_info["matched_concept"]
                sim = rec_info["similarity"]
                rec_type = rec.get("record_type", "unknown")
                # Format the raw record as compact JSON
                display_rec = {k: v for k, v in rec.items() if k != "record_type"}
                rec_json = json.dumps(display_rec, ensure_ascii=False)
                sections.append(
                    f"    [{rec_type.upper()}] (concept: {concept}, sim: {sim:.3f})"
                )
                sections.append(f"      {rec_json}")

    # Instructions for LLM
    sections.append(
        "\n## Instructions\n"
        "Based on the above patient evidence, medical reasoning paths, and "
        "path-grounded patient records, provide a clinically relevant answer "
        "to the patient query. Cite the evidence items by number. "
        "Use the raw patient records for specific values, dates, and context. "
        "If the evidence is insufficient, state what additional information is needed."
    )

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline (with caching)
# ═══════════════════════════════════════════════════════════════════════════

# Module-level caches for expensive objects
_graph_cache: Dict[str, nx.DiGraph] = {}
_emr_cache: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]]]] = {}


def _load_graph(graph_path: str) -> nx.DiGraph:
    """Load SNOMED graph with process-level caching."""
    if graph_path not in _graph_cache:
        with open(graph_path, "rb") as f:
            _graph_cache[graph_path] = pickle.load(f)
    return _graph_cache[graph_path]


def _load_emr(emr_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load and parse EMR with caching (keyed on file path + mtime)."""
    mtime = os.path.getmtime(emr_path)
    cache_key = f"{emr_path}:{mtime}"
    if cache_key not in _emr_cache:
        with open(emr_path, "r", encoding="utf-8") as f:
            raw_emr = f.read()
        _emr_cache[cache_key] = parse_emr(raw_emr)
    return _emr_cache[cache_key]


def run_pipeline(
    query: str,
    emr_path: str,
    cui_emb_path: str,
    graph_path: str,
    cui_name_path: str,
    reference_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Execute the full Graph-Augmented Semantic RAG pipeline (v2).

    Key improvements over v1:
      - Medical entity extraction before encoding
      - Per-entity CUI retrieval (better precision)
      - ISA-throttled graph expansion
      - Hard evidence-graph intersection filter
      - Concept overlap in evidence scoring
      - Aggressive caching (graph, EMR, embeddings)

    Returns a structured dictionary with query, entities, seed_cuis,
    reasoning_paths, evidence, and prompt.
    """
    t0 = time.time()

    if reference_date is None:
        reference_date = datetime.now()

    # ── Load graph (cached after first call) ──
    print("[1/9] Loading SNOMED graph …")
    graph = _load_graph(graph_path)

    # ── Phase 1: Entity extraction + query encoding ──
    print("[2/9] Extracting entities & encoding query …")
    query_vec, entities, entity_vecs = encode_query(query)
    print(f"       Entities: {entities}")

    # ── Phase 2: CUI retrieval (per-entity) ──
    print("[3/9] Retrieving seed CUIs …")
    seed_hits = retrieve_seed_cuis(
        query_vec, cui_emb_path, cui_name_path,
        entities=entities, entity_vecs=entity_vecs,
        query_text=query,
    )
    seed_cuis = [cui for cui, _ in seed_hits]
    ci = _CUIIndex.get(cui_emb_path, cui_name_path)

    print(f"       Seed CUIs: {[(c, ci.cui_preferred_name(c)) for c in seed_cuis[:5]]}")

    # ── Phase 3: Graph expansion (ISA-throttled) ──
    print("[4/9] Expanding knowledge graph …")
    expanded_cuis, raw_paths = expand_graph(seed_cuis, graph)
    print(f"       Expanded to {len(expanded_cuis)} nodes, {len(raw_paths)} paths")

    # ── Phase 4: Path ranking (with taxonomy penalty) ──
    print("[5/9] Ranking reasoning paths …")
    ranked_paths = rank_paths(raw_paths, query_vec, ci)

    # ── Phase 5: EMR parsing (cached) ──
    print("[6/10] Parsing EMR data …")
    emr_dict, evidence_items = _load_emr(emr_path)
    print(f"       Extracted {len(evidence_items)} evidence items")

    # ── Phase 5b: Path-grounded EMR record retrieval ──
    print("[7/10] Retrieving path-grounded patient records …")
    path_grounded = retrieve_path_grounded_records(
        ranked_paths, emr_dict, cui_emb_path, cui_name_path,
    )
    grounded_count = sum(len(pg["grounded_records"]) for pg in path_grounded)
    print(f"       {grounded_count} records matched across {len(ranked_paths)} paths")

    # ── Phase 6: EMR concept matching (hard filter) ──
    print("[8/10] Matching EMR concepts to graph …")
    matched_evidence = match_emr_to_graph(
        evidence_items, expanded_cuis, cui_emb_path, cui_name_path,
        query_entities=entities,
        hard_filter=True,
    )
    print(f"       {len(matched_evidence)} evidence items with graph connection")

    # ── Phase 7: Temporal filtering ──
    print("[9/10] Applying temporal filters …")
    filtered_evidence = temporal_filter(matched_evidence, reference_date, expanded_cuis)
    print(f"       {len(filtered_evidence)} evidence items after temporal filter")

    # ── Phase 8: Evidence ranking (with concept overlap) ──
    print("[10/10] Ranking evidence …")
    top_evidence = rank_evidence(filtered_evidence, query_vec, reference_date)

    # ── Phase 9: Prompt assembly ──
    prompt = assemble_prompt(
        query, top_evidence, ranked_paths, cui_emb_path, cui_name_path,
        path_grounded=path_grounded,
    )

    elapsed = time.time() - t0
    print(f"\n✓ Pipeline complete in {elapsed:.2f}s")

    # ── Build serialisable output ──
    def _serialise_evidence(ev: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(ev)
        out.pop("text_embedding", None)
        if out.get("date") is not None:
            out["date"] = out["date"].strftime("%d-%b-%Y")
        return out

    return {
        "query": query,
        "entities": entities,
        "seed_cuis": [
            {"cui": cui, "name": ci.cui_preferred_name(cui), "score": round(sc, 4)}
            for cui, sc in seed_hits
        ],
        "reasoning_paths": ranked_paths,
        "path_grounded_records": path_grounded,
        "evidence": [_serialise_evidence(ev) for ev in top_evidence],
        "prompt": prompt,
        "elapsed_seconds": round(elapsed, 2),
    }
