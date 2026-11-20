import requests

class EmbeddingError(Exception):
    """Raised when embeddding generation fails"""
    pass

class EmbeddingService:
    """
    Generates text embeddings using a locally running Ollama  model.
    """

    def __init__(
            self,
            model:str = "nomic-embed-text",
            base_url : str = "http://localhost:11434",
            timeout : int = 60
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _generate_embeddings(
            self,
            texts: list[str],
            prefix: str
    )  -> list[list[float]]:
        """
        Generatee embeddings for a list of texts """

        if not texts:
            raise ValueError("Input texts cannot be empty.")

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Every input text must be a non-empty string."
                )

        prefixed_texts = [
            f"{prefix}{text.strip()}"
            for text in texts
        ]

        payload = {
            "model": self.model,
            "input": prefixed_texts
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()

            result = response.json()

            embeddings = result.get("embeddings")

            if not isinstance(embeddings, list):
                raise EmbeddingError(
                    "Invalid response: embeddings not found."
                )

            if len(embeddings) != len(texts):
                raise EmbeddingError(
                    "Embedding count does not match input count."
                )

            for vector in embeddings:
                if (
                    not isinstance(vector, list)
                    or not vector
                    or not all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        for value in vector
                    )
                ):
                    raise EmbeddingError(
                        "Invalid embedding vector received."
                    )

            dimensions = len(embeddings[0])

            if any(
                len(vector) != dimensions
                for vector in embeddings
            ):
                raise EmbeddingError(
                    "Embedding dimensions are inconsistent."
                )

            return [
                [float(value) for value in vector]
                for vector in embeddings
            ]

        except requests.exceptions.RequestException as error:
            raise EmbeddingError(
                f"Could not connect to Ollama: {error}"
            ) from error

    def embed_document(self, text: str) -> list[float]:
        """
        Generate an embedding for a document chunk.
        """

        embeddings = self._generate_embeddings(
            texts=[text],
            prefix="search_document: "
        )

        return embeddings[0]

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple document chunks.
        """

        return self._generate_embeddings(
            texts=texts,
            prefix="search_document: "
        )

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a user search query.
        """

        embeddings = self._generate_embeddings(
            texts=[query],
            prefix="search_query: "
        )

        return embeddings[0]
