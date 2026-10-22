"""
Proof graph data structure.

Represents the graph of logical derivations:
- Nodes: Formulas (axioms and derived schema/formula vertices)
- Hyperedges: Derivation steps (premise set/list to conclusion)
- Edges: 2-section projection from each premise to the conclusion

Uses NetworkX for graph operations.
"""

from typing import List, Dict, Set, Optional, Iterator, Any, Tuple, FrozenSet
from dataclasses import dataclass, field
import logging

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx is required. Install with: pip install networkx")

from core.formula import Formula

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProofNode:
    """
    A node in the proof graph representing a formula.

    Attributes:
        id: Unique identifier
        formula: The logical formula
        normalized_hash: Canonical hash for α-equivalence
        depth: Derivation depth (0 for axioms)
        rule: Inference rule used to derive (None for axioms)
        premises: IDs of premise nodes
        label: Optional human-readable label (e.g., axiom name)
        is_axiom: Whether this is an axiom (depth 0)
        skolem_constants: Skolem constants introduced in this derivation's scope
                         (formulas with these constants cannot be universally generalized)
    """
    id: int
    formula: Formula
    normalized_hash: str
    depth: int
    rule: Optional[str]
    premises: List[int]
    label: Optional[str] = None
    is_axiom: bool = False
    skolem_constants: FrozenSet[str] = field(default_factory=frozenset)

    def __str__(self) -> str:
        if self.label:
            return f"[{self.id}] {self.label}: {self.formula}"
        return f"[{self.id}] {self.formula}"

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'formula': str(self.formula),
            'hash': self.normalized_hash,
            'depth': self.depth,
            'rule': self.rule,
            'premises': self.premises,
            'label': self.label,
            'is_axiom': self.is_axiom,
            'size': self.formula.size(),
            'skolem_constants': list(self.skolem_constants),
        }


@dataclass(frozen=True, slots=True)
class ProofHyperedge:
    """
    A first-class derivation record.

    ``ProofGraph`` still exposes a NetworkX DiGraph for network metrics, but
    that graph is only the 2-section projection of these hyperedges. For MP,
    for example, the hyperedge stores both premises together.
    """

    id: int
    rule: str
    premises: Tuple[int, ...]
    conclusion: int
    is_macro: bool = False
    source_schema: Optional[str] = None
    approximation_metadata: Dict[str, Any] = field(default_factory=dict)
    # Subset of premises projected as 2-section edges (None = all premises).
    # Stored so a DEFERRED projection can be rebuilt post-hoc identically.
    projection_premises: Optional[Tuple[int, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "rule": self.rule,
            "premises": list(self.premises),
            "conclusion": self.conclusion,
            "is_macro": self.is_macro,
            "source_schema": self.source_schema,
            "approximation_metadata": dict(self.approximation_metadata),
        }


class ProofGraph:
    """
    Graph of logical derivations.

    Uses NetworkX directed graph with nodes representing formulas and edges
    representing the directed 2-section projection of proof hyperedges.
    """

    def __init__(
        self,
        *,
        graph_kind: str = "primary_schema_formula_space",
        projection_kind: str = "directed_2_section",
        metadata: Optional[Dict[str, Any]] = None,
        defer_projection: bool = False,
    ):
        """Initialize an empty proof graph.

        ``defer_projection`` skips building the 2-section projection (the nx
        edge set) during generation — it is a DERIVED view only read at
        analysis. With it on, ``add_hyperedge`` only stores hyperedges (the
        source of truth) and the projection is rebuilt lazily on first read
        (``out_degree`` etc.). This removes the dominant online cost
        (``_add_projection_edge``) from the hot merge AND keeps it out of
        checkpoints (smaller/faster pickles). The rebuilt projection is
        byte-identical to the eager one.
        """
        self._graph = nx.DiGraph()
        self._defer_projection = bool(defer_projection)
        # True once the projection edges are present. When deferring, becomes
        # False until materialized; otherwise edges are always live.
        self._projection_materialized = not self._defer_projection
        # Cheap distinct-(premise,conclusion) tally so num_edges() is accurate
        # while deferred WITHOUT building the heavy nx edge set.
        self._projection_edge_keys: set = set()
        self._nodes: Dict[int, ProofNode] = {}
        self._hash_to_id: Dict[str, int] = {}
        self._hyperedges: List[ProofHyperedge] = []
        # Parallel list to _hyperedges: rediscovery_count[i] == how many times
        # the same (rule, ordered_premises, conclusion) was applied. Preserved
        # via len(self._hyperedges) == len(self._hyperedge_counts) invariant.
        self._hyperedge_counts: List[int] = []
        # Dedup index for unique hyperedges: maps (rule, ordered_premises_tuple,
        # conclusion) → position in self._hyperedges.  Premise order is part of
        # proof identity (for MP it distinguishes the minor and major slots).
        # Lookup is O(1); avoids allocating a new ProofHyperedge on each exact
        # rediscovery hit.
        self._hyperedge_index: Dict[Tuple[str, Tuple[int, ...], int], int] = {}
        self.graph_kind = graph_kind
        self.projection_kind = projection_kind
        self.metadata: Dict[str, Any] = dict(metadata or {})

        # Cache for nx.ancestors() lookups. Invalidated on every edge mutation
        # (add_hyperedge / _add_projection_edge). Unbounded — grows with the
        # number of distinct (node_id) queries during post-analysis.
        self._ancestors_cache: Dict[int, FrozenSet[int]] = {}

        logger.debug("ProofGraph initialized")

    def _ancestors(self, node_id: int) -> FrozenSet[int]:
        """Cached wrapper for nx.ancestors().

        The cache is invalidated whenever an edge is added to the graph.
        Safe to use across calls between mutations; safe in post-analysis
        where the graph is read-only.
        """
        cached = self._ancestors_cache.get(node_id)
        if cached is not None:
            return cached
        self._ensure_projection()
        result = frozenset(nx.ancestors(self._graph, node_id))
        self._ancestors_cache[node_id] = result
        return result

    def __len__(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    def __contains__(self, node_id: int) -> bool:
        """Check if a node ID exists."""
        return node_id in self._nodes

    def add_node(self, node: ProofNode) -> None:
        """
        Add a node to the graph.

        Args:
            node: The ProofNode to add
        """
        if node.id in self._nodes:
            logger.warning(f"Node {node.id} already exists, skipping")
            return

        self._nodes[node.id] = node
        self._hash_to_id[node.normalized_hash] = node.id
        attrs = {
            "formula": str(node.formula),
            "hash": node.normalized_hash,
            "depth": node.depth,
            "is_axiom": node.is_axiom,
            "size": node.formula.size(),
        }
        if node.rule is not None:
            attrs["rule"] = node.rule
        if node.label is not None:
            attrs["label"] = node.label
        self._graph.add_node(node.id, **attrs)

        logger.debug(f"Added node {node.id}: {node.formula}")

    def _add_projection_edge(
        self,
        source_id: int,
        target_id: int,
        *,
        rule: Optional[str] = None,
        hyperedge_id: Optional[int] = None,
    ) -> None:
        """
        Add a projected edge from source to target.

        If the edge ``(source_id, target_id)`` already exists, the new
        ``rule`` is appended to the edge's ``rules`` list rather than
        overwriting the existing attribute. This preserves rule
        multiplicity for analyses that attribute structural roles to
        specific inference rules. The legacy ``rule`` attribute reflects
        the most recent rule (kept for backward compatibility).

        Args:
            source_id: ID of source node (premise)
            target_id: ID of target node (conclusion)
            rule: Inference rule used
        """
        if source_id not in self._nodes:
            logger.warning(f"Source node {source_id} not found")
            return
        if target_id not in self._nodes:
            logger.warning(f"Target node {target_id} not found")
            return

        if self._graph.has_edge(source_id, target_id):
            data = self._graph[source_id][target_id]
            # rules is a set of distinct rule names that produced this edge.
            # Switched from list (which grew unbounded on rediscovery) to set
            # since we only need rule-existence for metrics, not counts.
            if rule is not None:
                rules = data.get("rules")
                if rules is None:
                    existing = data.get("rule")
                    rules = {existing} if existing is not None else set()
                    data["rules"] = rules
                elif isinstance(rules, list):
                    # Legacy graph loaded from old checkpoint with list-rules.
                    rules = set(rules)
                    data["rules"] = rules
                rules.add(rule)
                data["rule"] = rule
            # hyperedge_ids stays a list — after H1 dedup, list growth is
            # bounded by number of unique (rule, premises, conclusion) tuples.
            if hyperedge_id is not None:
                hyperedge_ids = data.get("hyperedge_ids")
                if hyperedge_ids is None:
                    hyperedge_ids = []
                    data["hyperedge_ids"] = hyperedge_ids
                if hyperedge_id not in hyperedge_ids:
                    hyperedge_ids.append(hyperedge_id)
        else:
            attrs: Dict[str, Any] = {}
            if rule is not None:
                attrs["rule"] = rule
                attrs["rules"] = {rule}
            if hyperedge_id is not None:
                attrs["hyperedge_ids"] = [hyperedge_id]
            self._graph.add_edge(source_id, target_id, **attrs)
            if self._ancestors_cache:
                self._ancestors_cache.clear()
        logger.debug(f"Added edge {source_id} -> {target_id} [{rule}]")

    def add_edge(self, source_id: int, target_id: int, rule: Optional[str] = None) -> None:
        """
        Add a legacy projected edge from source to target.

        Prefer ``add_hyperedge`` for new derivation-producing code.
        """
        self._add_projection_edge(source_id, target_id, rule=rule)

    def add_hyperedge(
        self,
        premises: List[int],
        conclusion: int,
        *,
        rule: str,
        is_macro: bool = False,
        source_schema: Optional[str] = None,
        approximation_metadata: Optional[Dict[str, Any]] = None,
        projection_premises: Optional[List[int]] = None,
    ) -> ProofHyperedge:
        """
        Add a derivation hyperedge and its directed 2-section projection.

        Args:
            premises: IDs of premise nodes. The order is preserved for
                provenance, although the projected graph only sees individual
                premise-to-conclusion arcs.
            conclusion: ID of the conclusion node.
            rule: Rule or schema-instance policy name.
            is_macro: Whether this edge comes from a macro/shortcut rule.
            source_schema: Optional schema family/name used to create the step.
            approximation_metadata: Optional per-step metadata.
            projection_premises: Optional subset of premise IDs to project as
                2-section edges. When omitted, all premises are projected.
                This lets callers preserve full hyperedge provenance while
                suppressing projection-only artifacts such as self-loops.

        Returns:
            The created ProofHyperedge.
        """
        if conclusion not in self._nodes:
            raise ValueError(f"Conclusion node {conclusion} not found")
        for premise in premises:
            if premise not in self._nodes:
                raise ValueError(f"Premise node {premise} not found")
        projection_sources = (
            premises if projection_premises is None else projection_premises
        )
        for premise in projection_sources:
            if premise not in self._nodes:
                raise ValueError(f"Projection premise node {premise} not found")

        # Dedup: the same (rule, ordered_premises, conclusion) tuple may arrive
        # multiple times via canonical artefacts in MP.  Increment the counter
        # instead of allocating a new ProofHyperedge.  Do not sort premises:
        # that would collapse distinct ordered rule slots and make the retained
        # proof depend on parallel arrival order.
        # Lazy init for graphs loaded from pre-H1 checkpoints (legacy pickle).
        if not hasattr(self, '_hyperedge_index'):
            self._hyperedge_index = {}
            self._hyperedge_counts = [1] * len(self._hyperedges)
            for he in self._hyperedges:
                key = (he.rule, tuple(he.premises), he.conclusion)
                self._hyperedge_index[key] = he.id
        premises_tuple = tuple(premises)
        dedup_key = (rule, premises_tuple, conclusion)
        existing_id = self._hyperedge_index.get(dedup_key)
        if existing_id is not None:
            self._hyperedge_counts[existing_id] += 1
            # Rediscovery of the SAME (rule, premises, conclusion): the projection
            # edge(s) already exist with this rule from the first allocation, so
            # re-adding is a no-op. Skip it (deferred or not — saves millions of
            # redundant has_edge/set ops on rediscovery hits).
            return self._hyperedges[existing_id]

        # Record which premises to project (subset, e.g. self-loop suppression)
        # so a deferred projection can be rebuilt identically.
        proj_pre = (None if projection_premises is None
                    else tuple(projection_sources))
        hyperedge = ProofHyperedge(
            id=len(self._hyperedges),
            rule=rule,
            premises=premises_tuple,
            conclusion=conclusion,
            is_macro=is_macro,
            source_schema=source_schema,
            approximation_metadata=dict(approximation_metadata or {}),
            projection_premises=proj_pre,
        )
        self._hyperedges.append(hyperedge)
        self._hyperedge_counts.append(1)
        self._hyperedge_index[dedup_key] = hyperedge.id
        if self._defer_projection:
            # Only tally distinct edge keys (cheap); skip the heavy nx build.
            for premise in projection_sources:
                self._projection_edge_keys.add((premise, conclusion))
        else:
            for premise in projection_sources:
                self._add_projection_edge(
                    premise,
                    conclusion,
                    rule=rule,
                    hyperedge_id=hyperedge.id,
                )
        return hyperedge

    def get_hyperedges(self) -> List[ProofHyperedge]:
        """Return all first-class derivation hyperedges."""
        return list(self._hyperedges)

    def get_hyperedges_for_conclusion(self, node_id: int) -> List[ProofHyperedge]:
        """Return derivation records concluding the given node."""
        return [edge for edge in self._hyperedges if edge.conclusion == node_id]

    def num_hyperedges(self) -> int:
        """Number of first-class derivation records."""
        return len(self._hyperedges)

    def get_node(self, node_id: int) -> Optional[ProofNode]:
        """
        Get a node by ID.

        Args:
            node_id: Node ID

        Returns:
            ProofNode, or None if not found
        """
        return self._nodes.get(node_id)

    def get_node_by_hash(self, formula_hash: str) -> Optional[ProofNode]:
        """
        Get a node by formula hash.

        Args:
            formula_hash: Canonical hash of the formula

        Returns:
            ProofNode, or None if not found
        """
        node_id = self._hash_to_id.get(formula_hash)
        if node_id is not None:
            return self._nodes.get(node_id)
        return None

    def has_formula(self, formula_hash: str) -> bool:
        """Check if a formula (by hash) exists in the graph."""
        return formula_hash in self._hash_to_id

    def get_all_nodes(self) -> Iterator[ProofNode]:
        """Iterate over all nodes."""
        return iter(self._nodes.values())

    def get_all_formulas(self) -> Iterator[ProofNode]:
        """Alias for get_all_nodes."""
        return self.get_all_nodes()

    def get_axioms(self) -> Iterator[ProofNode]:
        """Iterate over axiom nodes."""
        for node in self._nodes.values():
            if node.is_axiom:
                yield node

    def get_theorems(self) -> Iterator[ProofNode]:
        """Iterate over non-axiom derived nodes.

        Kept as a backward-compatible name; in the schema/formula graph these
        nodes are not necessarily object-level theorems.
        """
        for node in self._nodes.values():
            if not node.is_axiom:
                yield node

    def get_nodes_at_depth(self, depth: int) -> Iterator[ProofNode]:
        """Get all nodes at a specific depth."""
        for node in self._nodes.values():
            if node.depth == depth:
                yield node

    def _ensure_projection(self) -> None:
        """Lazily materialize the deferred 2-section projection from hyperedges.

        Rebuilds the exact edge set the eager path would have produced: each
        UNIQUE hyperedge contributes premise→conclusion arcs (rediscoveries are
        no-ops in both paths). Defensive for legacy pickles lacking the flag.
        """
        if getattr(self, "_projection_materialized", True):
            return
        for he in self._hyperedges:
            pp = getattr(he, "projection_premises", None)
            sources = he.premises if pp is None else pp
            for premise in sources:
                self._add_projection_edge(
                    premise, he.conclusion, rule=he.rule, hyperedge_id=he.id
                )
        self._projection_materialized = True

    def num_edges(self) -> int:
        """Number of distinct 2-section edges.

        Uses the cheap key tally while the projection is deferred (no nx build),
        so per-depth stats don't force materialization. Identical to the nx
        edge count once materialized / in eager mode.
        """
        if getattr(self, "_defer_projection", False) and not getattr(
            self, "_projection_materialized", True
        ):
            return len(self._projection_edge_keys)
        return self._graph.number_of_edges()

    def get_predecessors(self, node_id: int) -> List[int]:
        """Get IDs of predecessor nodes (premises)."""
        self._ensure_projection()
        if node_id not in self._graph:
            return []
        return list(self._graph.predecessors(node_id))

    def get_successors(self, node_id: int) -> List[int]:
        """Get IDs of successor nodes (derived from this node)."""
        self._ensure_projection()
        if node_id not in self._graph:
            return []
        return list(self._graph.successors(node_id))

    def in_degree(self, node_id: int) -> int:
        """Number of incoming edges (premises used)."""
        self._ensure_projection()
        if node_id not in self._graph:
            return 0
        return self._graph.in_degree(node_id)

    def out_degree(self, node_id: int) -> int:
        """Number of outgoing edges (formulas derived from this)."""
        self._ensure_projection()
        if node_id not in self._graph:
            return 0
        return self._graph.out_degree(node_id)

    def get_networkx_graph(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph."""
        self._ensure_projection()
        return self._graph

    def max_depth(self) -> int:
        """Get the maximum depth in the graph."""
        if not self._nodes:
            return 0
        return max(node.depth for node in self._nodes.values())

    def count_by_depth(self) -> Dict[int, int]:
        """Count nodes at each depth."""
        counts: Dict[int, int] = {}
        for node in self._nodes.values():
            counts[node.depth] = counts.get(node.depth, 0) + 1
        return counts

    def count_by_rule(self) -> Dict[str, int]:
        """Count nodes by derivation rule."""
        counts: Dict[str, int] = {}
        for node in self._nodes.values():
            rule = node.rule or 'axiom'
            counts[rule] = counts.get(rule, 0) + 1
        return counts

    def get_proof_tree(self, node_id: int) -> nx.DiGraph:
        """
        Extract the proof tree for a specific formula.

        Returns a subgraph containing all ancestors of the given node.

        Args:
            node_id: ID of the target formula

        Returns:
            NetworkX DiGraph containing the proof tree
        """
        if node_id not in self._graph:
            return nx.DiGraph()

        ancestors = set(self._ancestors(node_id))
        ancestors.add(node_id)
        return self._graph.subgraph(ancestors).copy()

    def get_skolem_context(self, node_id: int) -> FrozenSet[str]:
        """
        Get all Skolem constants in the derivation context of a node.

        This includes Skolem constants from the node itself and all its ancestors.
        Formulas containing these constants cannot be universally generalized.

        Args:
            node_id: ID of the node

        Returns:
            Set of Skolem constant names in context
        """
        if node_id not in self._nodes:
            return frozenset()

        skolem_set: Set[str] = set()

        # Add from this node
        node = self._nodes[node_id]
        skolem_set.update(node.skolem_constants)

        # Add from all ancestors
        try:
            ancestors = self._ancestors(node_id)
            for ancestor_id in ancestors:
                ancestor = self._nodes.get(ancestor_id)
                if ancestor:
                    skolem_set.update(ancestor.skolem_constants)
        except nx.NetworkXError:
            pass

        return frozenset(skolem_set)

    def summary(self) -> str:
        """Get a summary string of the graph."""
        lines = [
            "=== Proof Graph Summary ===",
            f"Total nodes: {len(self._nodes)}",
            f"Total hyperedges: {self.num_hyperedges()}",
            f"Total edges: {self.num_edges()}",
            f"Max depth: {self.max_depth()}",
            "",
            "Nodes by depth:",
        ]
        for depth, count in sorted(self.count_by_depth().items()):
            lines.append(f"  Depth {depth}: {count}")

        lines.append("")
        lines.append("Nodes by rule:")
        for rule, count in sorted(self.count_by_rule().items(), key=lambda x: -x[1]):
            lines.append(f"  {rule}: {count}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        self._ensure_projection()  # JSON export needs the 2-section edges
        hyperedge_dicts = []
        for idx, edge in enumerate(self._hyperedges):
            d = edge.to_dict()
            d['rediscovery_count'] = self._hyperedge_counts[idx]
            hyperedge_dicts.append(d)
        return {
            'graph_kind': self.graph_kind,
            'projection_kind': self.projection_kind,
            'nodes': [node.to_dict() for node in self._nodes.values()],
            'hyperedges': hyperedge_dicts,
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'rule': data.get('rule'),
                    'rules': sorted(data.get('rules', [])),
                    'hyperedge_ids': data.get('hyperedge_ids', []),
                }
                for u, v, data in self._graph.edges(data=True)
            ],
            'metadata': {
                **self.metadata,
                'num_nodes': len(self._nodes),
                'num_hyperedges': self.num_hyperedges(),
                'num_edges': self.num_edges(),
                'max_depth': self.max_depth(),
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProofGraph':
        """
        Create a ProofGraph from dictionary (without formulas).

        Note: This only restores graph structure, not full Formula objects.

        Args:
            data: Dictionary from to_dict()

        Returns:
            ProofGraph with structure restored
        """
        graph = cls(
            graph_kind=data.get('graph_kind', 'primary_schema_formula_space'),
            projection_kind=data.get('projection_kind', 'directed_2_section'),
            metadata=data.get('metadata', {}),
        )

        # This is a simplified restoration - full formula objects would need
        # to be reconstructed from string representations
        for node_data in data['nodes']:
            # Create a placeholder - real usage would parse formula strings
            graph._graph.add_node(
                node_data['id'],
                **{k: v for k, v in node_data.items() if k != 'id'}
            )
            graph._hash_to_id[node_data['hash']] = node_data['id']

        for edge_data in data['edges']:
            graph._graph.add_edge(
                edge_data['source'],
                edge_data['target'],
                rule=edge_data.get('rule'),
                rules=set(edge_data.get('rules', [])),
                hyperedge_ids=edge_data.get('hyperedge_ids', []),
            )

        for hyperedge_data in data.get('hyperedges', []):
            hyperedge = ProofHyperedge(
                id=hyperedge_data['id'],
                rule=hyperedge_data['rule'],
                premises=tuple(hyperedge_data.get('premises', [])),
                conclusion=hyperedge_data['conclusion'],
                is_macro=hyperedge_data.get('is_macro', False),
                source_schema=hyperedge_data.get('source_schema'),
                approximation_metadata=hyperedge_data.get(
                    'approximation_metadata', {}
                ),
            )
            graph._hyperedges.append(hyperedge)
            graph._hyperedge_counts.append(
                hyperedge_data.get('rediscovery_count', 1)
            )
            dedup_key = (
                hyperedge.rule,
                tuple(sorted(hyperedge.premises)),
                hyperedge.conclusion,
            )
            graph._hyperedge_index[dedup_key] = hyperedge.id

        return graph
