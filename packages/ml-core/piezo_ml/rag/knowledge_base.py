"""
ChromaDB-backed RAG Knowledge Base for Piezo.AI Agent.

Indexes research papers, materials data, and a built-in glossary.
Uses sentence-transformers/all-MiniLM-L6-v2 for local, free embeddings.
"""
import logging
import uuid
from typing import Optional

logger = logging.getLogger("piezo.rag.knowledge_base")

# Chunk config
CHUNK_SIZE = 500   # tokens (approx chars / 4)
CHUNK_OVERLAP = 50


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by approximate token count."""
    # Approximate: 1 token ≈ 4 chars
    char_size = chunk_size * 4
    char_overlap = overlap * 4
    chunks = []
    start = 0
    while start < len(text):
        end = start + char_size
        chunks.append(text[start:end])
        start = end - char_overlap
    return [c.strip() for c in chunks if c.strip()]


class KnowledgeBase:
    """Persistent ChromaDB knowledge base with sentence-transformer embeddings."""

    def __init__(self, persist_path: str = "/data/chroma"):
        logger.info("knowledge_base.init", extra={"persist_path": persist_path})
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            self._ef = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._client = chromadb.PersistentClient(path=persist_path)
            self._collection = self._client.get_or_create_collection(
                name="piezo_knowledge",
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("knowledge_base.ready", extra={"doc_count": self._collection.count()})
        except ImportError as e:
            logger.error("knowledge_base.import_error", extra={"error": str(e)})
            raise RuntimeError(
                "ChromaDB or sentence-transformers not installed. "
                "Install with: pip install chromadb sentence-transformers"
            ) from e

    # ── Index paper (PDF) ─────────────────────────────────────────────
    def index_paper(self, pdf_bytes: bytes, filename: str = "paper.pdf") -> dict:
        """Extract text from PDF, chunk, and index into ChromaDB."""
        logger.info("knowledge_base.index_paper.start", extra={"filename": filename})
        try:
            from PyPDF2 import PdfReader
            import io

            reader = PdfReader(io.BytesIO(pdf_bytes))
            full_text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            if not full_text.strip():
                return {"indexed": 0, "error": "No text could be extracted from PDF"}

            chunks = _chunk_text(full_text)
            ids = [f"paper_{filename}_{i}" for i in range(len(chunks))]
            metadatas = [
                {"type": "paper", "source": filename, "chunk_idx": i}
                for i in range(len(chunks))
            ]

            self._collection.upsert(
                ids=ids, documents=chunks, metadatas=metadatas
            )
            logger.info("knowledge_base.index_paper.success",
                        extra={"filename": filename, "chunks": len(chunks)})
            return {"indexed": len(chunks), "filename": filename}

        except ImportError:
            return {"indexed": 0, "error": "PyPDF2 not installed. Install with: pip install pypdf2"}
        except Exception as e:
            logger.error("knowledge_base.index_paper.error", exc_info=True)
            return {"indexed": 0, "error": str(e)}

    # ── Index material ────────────────────────────────────────────────
    def index_material(self, material: dict) -> None:
        """Index a single material record as a searchable document."""
        formula = material.get("formula", "unknown")
        doc_text = (
            f"Material: {formula}. "
            f"d33={material.get('d33', 'N/A')} pC/N, "
            f"Tc={material.get('tc', 'N/A')}°C. "
            f"Family: {material.get('family_name', 'N/A')}. "
            f"Notes: {material.get('notes', '')}."
        )
        doc_id = f"material_{material.get('id', uuid.uuid4().hex)}"
        self._collection.upsert(
            ids=[doc_id],
            documents=[doc_text],
            metadatas=[{"type": "material", "formula": formula}],
        )

    # ── Index glossary ────────────────────────────────────────────────
    def index_glossary(self, glossary: dict[str, str]) -> int:
        """Index a dictionary of term→definition pairs."""
        ids, docs, metas = [], [], []
        for term, definition in glossary.items():
            ids.append(f"glossary_{term.lower().replace(' ', '_')}")
            docs.append(f"{term}: {definition}")
            metas.append({"type": "glossary", "term": term})

        self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
        logger.info("knowledge_base.index_glossary", extra={"count": len(ids)})
        return len(ids)

    # ── Search ────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_type: Optional[str] = None,
    ) -> list[dict]:
        """Semantic search. Optionally filter by doc_type (paper|material|glossary)."""
        logger.info("knowledge_base.search", extra={"query": query[:80], "top_k": top_k})
        where_filter = {"type": doc_type} if doc_type else None
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )
        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })
        return docs

    # ── Stats ─────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        """Return knowledge base statistics."""
        total = self._collection.count()
        # Sample metadata to count types
        sample = self._collection.get(limit=min(total, 1000), include=["metadatas"])
        type_counts: dict[str, int] = {}
        if sample["metadatas"]:
            for m in sample["metadatas"]:
                t = m.get("type", "unknown") if m else "unknown"
                type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_documents": total,
            "by_type": type_counts,
        }
