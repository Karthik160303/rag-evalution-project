
import requests


class LLMError(Exception):
    """Raised when the LLM request fails."""
    pass


class OllamaLLM:

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 180
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2
    ) -> str:

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if not 0 <= temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()

            result = response.json()
            answer = result.get("response")

            if not isinstance(answer, str) or not answer.strip():
                raise LLMError("Ollama returned an empty response.")

            return answer.strip()

        except requests.exceptions.RequestException as error:
            raise LLMError(
                f"Could not connect to Ollama: {error}"
            ) from error
