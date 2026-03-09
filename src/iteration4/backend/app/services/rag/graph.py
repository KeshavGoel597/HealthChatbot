"""
SNOMED knowledge graph loader.

Loads the DR.KNOWS SNOMED DiGraph from pickle.
Provides neighbor queries with optional relation filtering.
No torch/faiss/embedding dependencies — pure networkx.
"""

import pickle
from pathlib import Path

import networkx as nx


class KnowledgeGraph:
    """SNOMED knowledge graph. Load once at startup, query instantly."""

    def __init__(self, graph_path: str):
        """Load NetworkX DiGraph from pickle.

        Args:
            graph_path: Path to SNOMED_CUI_MAJID_Graph_wSelf.pkl
        """
        path = Path(graph_path)
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")

        with open(path, "rb") as f:
            self._graph: nx.DiGraph = pickle.load(f)
        print(
            f"[KnowledgeGraph] Loaded {self.num_nodes} nodes, "
            f"{self.num_edges} edges, "
            f"{nx.number_of_selfloops(self._graph)} self-loops."
        )

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    def has_node(self, cui: str) -> bool:
        """Check if a CUI exists in the graph."""
        return cui in self._graph

    def neighbors(
        self,
        cui: str,
        allowed_relations: set[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Direct neighbors of a CUI.

        Args:
            cui: Source CUI string.
            allowed_relations: If provided, only return neighbors connected
                               by these relation types. None = all relations.

        Returns:
            List of (neighbor_cui, relation_label) tuples.
            Self-loops are always excluded.
        """
        if cui not in self._graph:
            return []

        results = []
        for neighbor in self._graph.neighbors(cui):
            if neighbor == cui:  # skip self-loops
                continue
            edge_data = self._graph.get_edge_data(cui, neighbor)
            label = edge_data.get("label", "") if edge_data else ""
            if allowed_relations is None or label in allowed_relations:
                results.append((neighbor, label))
        return results
