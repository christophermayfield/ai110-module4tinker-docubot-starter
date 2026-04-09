"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob

class DocuBot:
    # Minimum total score required for a snippet to be considered "meaningful"
    RELEVANCE_THRESHOLD = 1

    # Simple list of stop words to ignore during scoring
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "to", "of", "for", "in", "on", "at", "by", "with", "and", "or",
        "is", "there", "any", "mention"
    }

    # Simple manual guardrail: keywords that should always be refused.
    BANNED_KEYWORDS = ["bomb", "hack", "offensive", "steal"]

    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.raw_documents = self.load_documents()  # List of (filename, text)
        
        # Split documents into paragraph chunks
        self.documents = self.chunk_documents(self.raw_documents) # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        If the folder is empty or missing, it uses fallback docs from dataset.py.
        Returns a list of tuples: (filename, text)
        """
        from dataset import load_fallback_documents
        docs = []
        
        if os.path.exists(self.docs_folder):
            pattern = os.path.join(self.docs_folder, "*.*")
            for path in glob.glob(pattern):
                if path.endswith(".md") or path.endswith(".txt"):
                    with open(path, "r", encoding="utf8") as f:
                        text = f.read()
                    filename = os.path.basename(path)
                    docs.append((filename, text))
        
        # If no documents were loaded from the folder, use fallback docs
        if not docs:
            docs = load_fallback_documents()
            
        return docs

    def chunk_documents(self, documents):
        """
        Split each document into paragraphs (delimited by double newlines).
        Returns a list of tuples: (filename, paragraph_text)
        """
        chunks = []
        for filename, text in documents:
            # Split by one or more blank lines
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for p in paragraphs:
                chunks.append((filename, p))
        return chunks

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Build a tiny inverted index mapping lowercase words to the chunks
        they appear in. Stores the integer index of the chunk in self.documents.
        """
        index = {}
        for i, (filename, text) in enumerate(documents):
            # Simple tokenization: lowercase and split by whitespace
            # Removing basic punctuation to improve match quality
            words = text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "").split()
            for word in set(words):  # Use set for unique words per document
                if word not in index:
                    index[word] = []
                index[word].append(i)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_chunk(self, query, text):
        """
        Return a simple relevance score for how well the text matches the query.
        Ignore common stop words to ensure more meaningful matching.
        """
        query_words = [w for w in query.lower().split() if w not in self.STOP_WORDS]
        doc_words = text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "").split()
        
        score = 0
        for word in query_words:
            score += doc_words.count(word)
        return score

    def retrieve(self, query, top_k=3):
        """
        Use the index and scoring function to select top_k relevant document snippets.
        Only returns snippets that meet the RELEVANCE_THRESHOLD.

        Return a list of (filename, text) sorted by score descending.
        """
        query_words = query.lower().split()
        candidate_indices = set()
        
        # Use index to narrow down candidates
        for word in query_words:
            if word in self.index:
                candidate_indices.update(self.index[word])
        
        # Score candidates
        scored_results = []
        for i in candidate_indices:
            filename, text = self.documents[i]
            score = self.score_chunk(query, text)
            if score >= self.RELEVANCE_THRESHOLD:
                scored_results.append(((filename, text), score))
        
        # Sort by score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Extract the (filename, text) tuples
        results = [item[0] for item in scored_results]
        return results[:top_k]

    def check_guardrails(self, query):
        """
        Simple manual guardrail check.
        Returns the refusal message if a violation is found, otherwise None.
        """
        for keyword in self.BANNED_KEYWORDS:
            if keyword in query.lower():
                print(f"\n[Guardrail Triggered: Banned keyword '{keyword}']\n")
                return "i will not answer that"
        return None

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        violation = self.check_guardrails(query)
        if violation:
            return violation

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        violation = self.check_guardrails(query)
        if violation:
            return violation

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
