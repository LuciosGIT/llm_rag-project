import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FaissRetriever:
    """
    Retriever semântico baseado em embeddings + FAISS.

    Permite recuperar os chunks mais relevantes de uma base textual
    utilizando busca vetorial por similaridade.
    """

    def __init__(self, chunks):
        self.chunks = chunks

        # Modelo leve e muito usado em RAG
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Cria embeddings e indexa
        self.index = self._build_index()

    def _build_index(self):
        """
        Constrói o índice FAISS a partir dos embeddings dos chunks.
        """
        embeddings = self.model.encode(self.chunks)

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        return index

    def retrieve(self, query, k=3):
        """
        Recupera os k chunks mais relevantes para uma query.

        Args:
            query (str): Pergunta do usuário.
            k (int): Quantidade de chunks retornados.

        Returns:
            str: Contexto concatenado com os chunks mais relevantes.
        """
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = [self.chunks[i] for i in indices[0]]

        return "\n\n".join(results)
