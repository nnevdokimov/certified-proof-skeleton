"""
Graph I/O operations.

Save and load proof graphs in various formats:
- GraphML (default, good interoperability)
- GEXF (for Gephi)
- JSON (human-readable)
- Pickle (Python native, preserves all data)
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path
import gc
import json
import gzip
import os
import pickle
import logging
import tempfile

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx is required. Install with: pip install networkx")

from .proof_graph import ProofGraph, ProofNode

logger = logging.getLogger(__name__)


class GraphIO:
    """
    Save and load proof graphs.
    """

    _retained_pickle_loads = []

    @staticmethod
    def save(graph: ProofGraph, path: Union[str, Path],
             format: str = "graphml", compress: bool = False) -> None:
        """
        Save a proof graph to file.

        Args:
            graph: ProofGraph to save
            path: Output file path
            format: Output format ("graphml", "gexf", "json", "pickle")
            compress: Whether to gzip compress the output
        """
        path = Path(path)

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving graph to {path} (format={format}, compress={compress})")

        if format == "graphml":
            GraphIO._save_graphml(graph, path, compress)
        elif format == "gexf":
            GraphIO._save_gexf(graph, path, compress)
        elif format == "json":
            GraphIO._save_json(graph, path, compress)
        elif format == "pickle":
            GraphIO._save_pickle(graph, path, compress)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Graph saved successfully")

    @staticmethod
    def load(path: Union[str, Path], format: Optional[str] = None) -> ProofGraph:
        """
        Load a proof graph from file.

        Args:
            path: Input file path
            format: File format (auto-detected if None)

        Returns:
            Loaded ProofGraph
        """
        path = GraphIO._resolve_read_path(Path(path))

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Auto-detect format
        if format is None:
            format = GraphIO._detect_format(path)

        logger.info(f"Loading graph from {path} (format={format})")

        if format == "graphml":
            return GraphIO._load_graphml(path)
        elif format == "gexf":
            return GraphIO._load_gexf(path)
        elif format == "json":
            return GraphIO._load_json(path)
        elif format == "pickle":
            return GraphIO._load_pickle(path)
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def _detect_format(path: Path) -> str:
        """Detect file format from extension."""
        path = Path(path)
        while path.suffix.lower() == '.gz':
            path = Path(path.stem)
        suffix = path.suffix.lower()

        format_map = {
            '.graphml': 'graphml',
            '.gexf': 'gexf',
            '.json': 'json',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            'graphml': 'graphml',
            'gexf': 'gexf',
            'json': 'json',
            'pkl': 'pickle',
        }

        return format_map.get(suffix, 'graphml')

    @staticmethod
    def _compressed_output_path(path: Path) -> Path:
        """Return gzip output path without duplicating a trailing .gz."""
        return path if path.suffix.lower() == '.gz' else path.with_suffix(path.suffix + '.gz')

    @staticmethod
    def _legacy_double_gzip_path(path: Path) -> Path:
        """Return legacy path that accidentally duplicated the .gz suffix."""
        compressed = GraphIO._compressed_output_path(path)
        return compressed.with_suffix(compressed.suffix + '.gz')

    @staticmethod
    def _resolve_read_path(path: Path) -> Path:
        """Resolve current or legacy on-disk path for a serialized graph."""
        candidates = [path]
        compressed = GraphIO._compressed_output_path(path)
        legacy = GraphIO._legacy_double_gzip_path(path)
        for candidate in (compressed, legacy):
            if candidate not in candidates:
                candidates.append(candidate)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return path

    @staticmethod
    def _save_graphml(graph: ProofGraph, path: Path, compress: bool) -> None:
        """Save as GraphML."""
        nx_graph = graph.get_networkx_graph()

        if compress:
            path = GraphIO._compressed_output_path(path)
            with gzip.open(path, 'wb') as f:
                nx.write_graphml(nx_graph, f)
        else:
            nx.write_graphml(nx_graph, path)

    @staticmethod
    def _save_gexf(graph: ProofGraph, path: Path, compress: bool) -> None:
        """Save as GEXF (Gephi format)."""
        nx_graph = graph.get_networkx_graph()

        if compress:
            path = GraphIO._compressed_output_path(path)
            with gzip.open(path, 'wb') as f:
                nx.write_gexf(nx_graph, f)
        else:
            nx.write_gexf(nx_graph, path)

    @staticmethod
    def _save_json(graph: ProofGraph, path: Path, compress: bool) -> None:
        """Save as JSON."""
        data = graph.to_dict()

        if compress:
            path = GraphIO._compressed_output_path(path)
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _save_pickle(graph: ProofGraph, path: Path, compress: bool) -> None:
        """Save as pickle (preserves full object structure)."""
        if (
            compress
            and os.environ.get("ZFC_GRAPH_SAVE_FORK_SYNC", "").strip() == "1"
            and hasattr(os, "fork")
        ):
            pid = os.fork()
            if pid == 0:
                try:
                    GraphIO._save_pickle_atomic(
                        graph, path, compress=True, exit_after_commit=True
                    )
                except BaseException:
                    os._exit(70)
                os._exit(71)
            _, status = os.waitpid(pid, 0)
            if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
                detail = (
                    f"signal {os.WTERMSIG(status)}"
                    if os.WIFSIGNALED(status)
                    else f"exit {os.WEXITSTATUS(status)}"
                )
                raise RuntimeError(f"forked graph writer failed: {detail}")
            return
        GraphIO._save_pickle_atomic(graph, path, compress=compress)

    @staticmethod
    def _save_pickle_atomic(
        graph: ProofGraph,
        path: Path,
        *,
        compress: bool,
        exit_after_commit: bool = False,
    ) -> None:
        """Atomically save a pickle without GC traversal of a live Pickler memo."""
        target = GraphIO._compressed_output_path(path) if compress else path
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        committed = False
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            raw = os.fdopen(fd, "wb", closefd=False)
            stream = (
                gzip.GzipFile(
                    fileobj=raw,
                    mode="wb",
                    compresslevel=int(os.environ.get("ZFC_GRAPH_GZIP_LEVEL", "1")),
                )
                if compress
                else raw
            )
            with raw:
                if stream is raw:
                    pickle.dump(graph, stream, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with stream:
                        pickler = pickle.Pickler(
                            stream, protocol=pickle.HIGHEST_PROTOCOL
                        )
                        pickler.dump(graph)
            os.fsync(fd)
            os.replace(temporary, target)
            committed = True
            if exit_after_commit:
                os._exit(0)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            if not committed:
                temporary.unlink(missing_ok=True)
            if gc_was_enabled and not exit_after_commit:
                gc.enable()

    @staticmethod
    def _load_graphml(path: Path) -> ProofGraph:
        """Load from GraphML."""
        path = GraphIO._resolve_read_path(path)
        if path.suffix == '.gz':
            with gzip.open(path, 'rb') as f:
                nx_graph = nx.read_graphml(f)
        else:
            nx_graph = nx.read_graphml(path)

        return GraphIO._nx_to_proof_graph(nx_graph)

    @staticmethod
    def _load_gexf(path: Path) -> ProofGraph:
        """Load from GEXF."""
        path = GraphIO._resolve_read_path(path)
        if path.suffix == '.gz':
            with gzip.open(path, 'rb') as f:
                nx_graph = nx.read_gexf(f)
        else:
            nx_graph = nx.read_gexf(path)

        return GraphIO._nx_to_proof_graph(nx_graph)

    @staticmethod
    def _load_json(path: Path) -> ProofGraph:
        """Load from JSON."""
        path = GraphIO._resolve_read_path(path)
        if path.suffix == '.gz':
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        return ProofGraph.from_dict(data)

    @staticmethod
    def _load_pickle(path: Path) -> ProofGraph:
        """Load from pickle."""
        path = GraphIO._resolve_read_path(path)
        keep_load_state = (
            os.environ.get("ZFC_GRAPH_LOAD_KEEP_STATE", "").strip() == "1"
        )
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rb") as f:
                unpickler = pickle.Unpickler(f)
                obj = unpickler.load()
            if keep_load_state:
                # Retain both objects until the one-shot caller exits via
                # os._exit. This avoids multi-hour CPython 3.13 Unpickler-memo
                # destruction after a multi-million-object analysis load.
                GraphIO._retained_pickle_loads.append((obj, unpickler))
            else:
                del unpickler
        finally:
            if gc_was_enabled and not keep_load_state:
                gc.enable()
        # Some checkpoints pickle the serialized graph dict instead of the graph
        # object; rehydrate transparently. A dict without 'nodes' is a different
        # artifact (e.g. a generator config dump) and is returned as-is so the
        # caller fails loudly rather than getting a silently-wrong graph.
        if isinstance(obj, dict) and "nodes" in obj:
            return ProofGraph.from_dict(obj)
        return obj

    @staticmethod
    def _nx_to_proof_graph(nx_graph: nx.DiGraph) -> ProofGraph:
        """Convert NetworkX graph to ProofGraph (without full formulas)."""
        graph = ProofGraph()
        graph._graph = nx_graph

        # Reconstruct hash_to_id mapping
        for node_id, data in nx_graph.nodes(data=True):
            if 'hash' in data:
                # Convert node_id to int if needed
                int_id = int(node_id) if isinstance(node_id, str) else node_id
                graph._hash_to_id[data['hash']] = int_id

        return graph


def save_graph(graph: ProofGraph, path: Union[str, Path],
               format: str = "graphml", compress: bool = False) -> None:
    """Convenience function to save a graph."""
    GraphIO.save(graph, path, format, compress)


def load_graph(path: Union[str, Path], format: Optional[str] = None) -> ProofGraph:
    """Convenience function to load a graph."""
    return GraphIO.load(path, format)
