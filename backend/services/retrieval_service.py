"""
services/retrieval_service.py
------------------------------
Advanced RAG pipeline implemented with LangGraph.

Techniques
----------
1. HyDE           – generate a hypothetical "perfect answer" and retrieve using
                    its embedding alongside the raw query.
2. Multi-Query    – expand the original question into N reformulations to
                    improve recall across diverse phrasings.
3. RRF            – Reciprocal Rank Fusion to merge ranked lists without score
                    normalisation.
4. Cross-Encoder  – re-rank (query, passage) pairs with ms-marco-MiniLM-L-6-v2.
5. Compression    – LLM-driven contextual compression of each passage to only
                    the sentences relevant to the question.

Pipeline graph
--------------
START → expand_query → retrieve → rerank → compress → generate → END
"""

import json
import operator
import re
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from sentence_transformers import CrossEncoder

from ..core.vector_store import MultiModalVectorStore


# ── Pipeline state schema ─────────────────────────────────────────────────────

class PipelineState(TypedDict):
    query:              str
    hyde_document:      str
    expanded_queries:   List[str]
    text_results:       List[Dict[str, Any]]
    image_results:      List[Dict[str, Any]]
    audio_results:      List[Dict[str, Any]]
    reranked_results:   List[Dict[str, Any]]
    compressed_context: str
    answer:             str
    sources:            List[Dict[str, Any]]
    # pipeline_trace accumulates log strings from every node via operator.add
    pipeline_trace:     Annotated[List[str], operator.add]


# ── Service class ─────────────────────────────────────────────────────────────

class RetrievalService:
    """
    Orchestrates the full RAG pipeline from query expansion through generation.
    Wraps a LangGraph state-machine graph for transparent, traceable execution.
    """

    def __init__(
        self,
        vector_store: MultiModalVectorStore,
        ollama_host: str = "http://localhost:11434",
        model: str = "qwen2:0.5b",
    ) -> None:
        self.vector_store = vector_store
        self.ollama_host  = ollama_host
        self.model_name   = model

        print(f"  Connecting to Ollama model: {model}")
        self.llm = ChatOllama(model=model, base_url=ollama_host, temperature=0.7)

        print("  Loading cross-encoder reranker (ms-marco-MiniLM-L-6-v2) …")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

        self._graph = self._build_graph()

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        """Return meaningful lowercase terms for lightweight HyDE sanity checks."""
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
            "do", "does", "for", "from", "give", "how", "i", "in", "is", "it",
            "me", "of", "on", "or", "please", "summarize", "summary", "tell",
            "that", "the", "this", "to", "what", "when", "where", "which",
            "who", "why", "with", "would", "you",
        }
        terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()))
        return terms - stopwords

    def _build_hyde_document(self, query: str) -> str:
        query_terms = self._content_terms(query)
        if not query_terms:
            return query

        # HyDE is only a retrieval aid. Keep it query-shaped and reject detached
        # generations so a hallucinated paragraph cannot dominate recall.
        hyde_prompt = ChatPromptTemplate.from_template(
            "Create a concise hypothetical passage that could appear in a document "
            "answering the user's question. Keep it generic and retrieval-focused. "
            "Do not invent file names, titles, authors, dates, awards, or named "
            "entities unless they appear in the question. Output ONLY the passage, "
            "maximum 70 words.\n\nQuestion: {query}"
        )

        try:
            hyde_document: str = (hyde_prompt | self.llm | StrOutputParser()).invoke(
                {"query": query}
            )
        except Exception:
            return query

        hyde_document = " ".join(hyde_document.split())
        hyde_terms = self._content_terms(hyde_document)
        overlap = query_terms & hyde_terms

        if not hyde_document or len(hyde_document.split()) > 90 or not overlap:
            return query

        return hyde_document

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(PipelineState)

        builder.add_node("expand_query", self._expand_query)
        builder.add_node("retrieve",     self._retrieve)
        builder.add_node("rerank",       self._rerank)
        builder.add_node("compress",     self._compress)
        builder.add_node("generate",     self._generate)

        builder.add_edge(START,          "expand_query")
        builder.add_edge("expand_query", "retrieve")
        builder.add_edge("retrieve",     "rerank")
        builder.add_edge("rerank",       "compress")
        builder.add_edge("compress",     "generate")
        builder.add_edge("generate",     END)

        return builder.compile()

    # ── Node: query expansion (HyDE + multi-query) ────────────────────────────

    def _expand_query(self, state: PipelineState) -> Dict[str, Any]:
        query = state["query"]

        hyde_document = self._build_hyde_document(query)

        # Multi-query: generate diverse reformulations to improve recall
        mq_prompt = ChatPromptTemplate.from_template(
            "Generate 3 distinct search-query reformulations of the question below "
            "to maximise document recall.\n"
            'Return ONLY a JSON array of 3 strings, e.g.: ["q1", "q2", "q3"]\n\n'
            "Question: {query}"
        )
        expanded_queries = [query]
        try:
            raw = (mq_prompt | self.llm | StrOutputParser()).invoke({"query": query})
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                variants = json.loads(match.group())
                expanded_queries += [v for v in variants if isinstance(v, str) and v != query]
        except Exception:
            pass

        return {
            "hyde_document":    hyde_document,
            "expanded_queries": expanded_queries,
            "pipeline_trace":   [
                f"HyDE doc generated; {len(expanded_queries)} query variants produced"
            ],
        }

    # ── Node: retrieval with RRF fusion ───────────────────────────────────────

    def _retrieve(self, state: PipelineState) -> Dict[str, Any]:
        hyde_doc    = state.get("hyde_document", "") or state["query"]
        all_queries = list(dict.fromkeys(state["expanded_queries"] + [hyde_doc]))

        K = 60  # RRF constant
        text_scores:  Dict[str, float] = {}
        text_docs:    Dict[str, Dict]  = {}
        image_scores: Dict[str, float] = {}
        image_docs:   Dict[str, Dict]  = {}
        audio_scores: Dict[str, float] = {}
        audio_docs:   Dict[str, Dict]  = {}

        for q in all_queries:
            for rank, doc in enumerate(self.vector_store.search_text(q, n_results=6)):
                did = doc["id"]
                text_scores[did] = text_scores.get(did, 0.0) + 1.0 / (K + rank + 1)
                text_docs[did]   = doc

            for rank, doc in enumerate(self.vector_store.search_images(q, n_results=3)):
                did = doc["id"]
                image_scores[did] = image_scores.get(did, 0.0) + 1.0 / (K + rank + 1)
                image_docs[did]   = doc

            for rank, doc in enumerate(self.vector_store.search_audio(q, n_results=4)):
                did = doc["id"]
                audio_scores[did] = audio_scores.get(did, 0.0) + 1.0 / (K + rank + 1)
                audio_docs[did]   = doc

        def _rrf_sort(scores, docs, top_n):
            return sorted(
                [{**docs[i], "rrf_score": scores[i]} for i in docs],
                key=lambda x: x["rrf_score"],
                reverse=True,
            )[:top_n]

        text_results  = _rrf_sort(text_scores,  text_docs,  8)
        image_results = _rrf_sort(image_scores, image_docs, 3)
        audio_results = _rrf_sort(audio_scores, audio_docs, 4)

        return {
            "text_results":  text_results,
            "image_results": image_results,
            "audio_results": audio_results,
            "pipeline_trace": [
                f"RRF fusion over {len(all_queries)} queries → "
                f"{len(text_results)} text + {len(image_results)} image "
                f"+ {len(audio_results)} audio results"
            ],
        }

    # ── Node: cross-encoder re-ranking ────────────────────────────────────────

    def _rerank(self, state: PipelineState) -> Dict[str, Any]:
        query         = state["query"]
        text_results  = state["text_results"]
        image_results = state["image_results"]
        audio_results = state["audio_results"]

        if text_results:
            pairs  = [(query, doc["text"]) for doc in text_results]
            scores = self.reranker.predict(pairs)
            for doc, score in zip(text_results, scores):
                doc["rerank_score"] = float(score)
            text_results = sorted(
                text_results, key=lambda x: x["rerank_score"], reverse=True
            )[:5]

        # Also rerank audio transcript chunks with the cross-encoder
        if audio_results:
            pairs  = [(query, doc["text"]) for doc in audio_results]
            scores = self.reranker.predict(pairs)
            for doc, score in zip(audio_results, scores):
                doc["rerank_score"] = float(score)
            audio_results = sorted(
                audio_results, key=lambda x: x["rerank_score"], reverse=True
            )[:3]

        return {
            "reranked_results": text_results + image_results + audio_results,
            "pipeline_trace":   [
                f"Cross-encoder reranked {len(text_results)} text "
                f"+ {len(audio_results)} audio passages"
            ],
        }

    # ── Node: contextual compression ─────────────────────────────────────────

    def _compress(self, state: PipelineState) -> Dict[str, Any]:
        query    = state["query"]
        reranked = state["reranked_results"]

        compress_prompt = ChatPromptTemplate.from_template(
            "Extract and return ONLY the sentences from the passage below that are "
            "directly relevant to the question. Preserve exact wording. "
            "If nothing is relevant, return an empty string.\n\n"
            "Question: {query}\n\nPassage:\n{passage}"
        )
        compress_chain = compress_prompt | self.llm | StrOutputParser()

        parts: List[str] = []
        for doc in reranked[:6]:
            if doc["modality"] == "text":
                try:
                    compressed = compress_chain.invoke(
                        {"query": query, "passage": doc["text"]}
                    )
                    if compressed.strip():
                        meta = doc["metadata"]
                        parts.append(
                            f"[{meta.get('title', 'Document')}]\n{compressed.strip()}"
                        )
                except Exception:
                    parts.append(doc["text"])
            else:
                meta = doc["metadata"]
                if doc["modality"] == "image":
                    parts.append(
                        f"[Image: {meta.get('file_name', 'image')}]\n"
                        f"Caption: {doc['text']}"
                    )
                else:  # audio
                    parts.append(
                        f"[Audio transcript: {meta.get('file_name', 'audio')}]\n"
                        f"{doc['text']}"
                    )

        return {
            "compressed_context": "\n\n---\n\n".join(parts),
            "pipeline_trace": [f"Contextual compression applied to {len(reranked[:6])} passages"],
        }

    # ── Node: answer generation ───────────────────────────────────────────────

    def _generate(self, state: PipelineState) -> Dict[str, Any]:
        query    = state["query"]
        context  = state.get("compressed_context", "")
        reranked = state["reranked_results"]

        if not context.strip():
            return {
                "answer": (
                    "I could not find relevant information in the knowledge base "
                    "to answer your question."
                ),
                "sources":        [],
                "pipeline_trace": ["Generation skipped — no context found"],
            }

        gen_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a precise AI assistant. Answer questions using ONLY "
                    "the provided context. Cite source titles when relevant. "
                    "If the context does not contain the answer, say so clearly.",
                ),
                (
                    "human",
                    "Context:\n{context}\n\nQuestion: {query}\n\n"
                    "Provide a clear, well-structured answer.",
                ),
            ]
        )

        try:
            answer: str = (gen_prompt | self.llm | StrOutputParser()).invoke(
                {"context": context, "query": query}
            )
        except Exception as exc:
            answer = f"Generation error: {exc}"

        # Build deduplicated source list for the UI
        sources:      List[Dict[str, Any]] = []
        seen_titles:  set                  = set()
        for doc in reranked[:6]:
            meta = doc["metadata"]
            if doc["modality"] == "text":
                title = meta.get("title", "Document")
                if title not in seen_titles:
                    seen_titles.add(title)
                    sources.append(
                        {
                            "type":  "text",
                            "title": title,
                            "file":  meta.get("file_name", ""),
                            "score": round(
                                doc.get("rerank_score", doc.get("rrf_score", 0)), 3
                            ),
                        }
                    )
            else:
                fname = meta.get("file_name", "unknown")
                if fname not in seen_titles:
                    seen_titles.add(fname)
                    if doc["modality"] == "image":
                        sources.append(
                            {
                                "type":      "image",
                                "title":     fname,
                                "file":      fname,
                                "thumbnail": meta.get("thumbnail", ""),
                                "score":     round(doc.get("rrf_score", 0), 3),
                            }
                        )
                    else:  # audio
                        sources.append(
                            {
                                "type":     "audio",
                                "title":    meta.get("title", fname),
                                "file":     fname,
                                "duration": meta.get("duration_s", "?"),
                                "score":    round(doc.get("rrf_score", 0), 3),
                            }
                        )

        return {
            "answer":         answer,
            "sources":        sources,
            "pipeline_trace": [f"Answer generated using {len(sources)} sources"],
        }

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the full pipeline for *query*.

        Returns a dict with:
            answer, sources, pipeline_trace,
            hyde_document, expanded_queries, rrf_results
        """
        initial_state: PipelineState = {
            "query":              query,
            "hyde_document":      "",
            "expanded_queries":   [],
            "text_results":       [],
            "image_results":      [],
            "audio_results":      [],
            "reranked_results":   [],
            "compressed_context": "",
            "answer":             "",
            "sources":            [],
            "pipeline_trace":     [],
        }
        result = self._graph.invoke(initial_state)

        # Merge text + image + audio results and sort by RRF score for the UI table
        all_retrieved = result["text_results"] + result["image_results"] + result["audio_results"]
        all_retrieved.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        rrf_results = [
            {
                "title":     doc["metadata"].get("title") or doc["metadata"].get("file_name", ""),
                "file":      doc["metadata"].get("file_name", ""),
                "rrf_score": round(doc.get("rrf_score", 0), 4),
                "modality":  doc.get("modality", "text"),
            }
            for doc in all_retrieved
        ]

        return {
            "answer":           result["answer"],
            "sources":          result["sources"],
            "pipeline_trace":   result["pipeline_trace"],
            "hyde_document":    result.get("hyde_document", ""),
            "expanded_queries": result.get("expanded_queries", []),
            "rrf_results":      rrf_results,
        }

    def check_llm_connection(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            r = requests.get(f"{self.ollama_host}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
