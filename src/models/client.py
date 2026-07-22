"""Backend clients for API, Ollama, and HuggingFace model access."""

from functools import lru_cache

from constants.logger import setup_logger

LOGGER = setup_logger(__name__)


class ApiClient:
    """OpenAI-compatible HTTP client for chat and embeddings."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0):
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout

    def chat(self, messages: list[dict], model: str, **kwargs) -> str:
        import httpx

        payload = {"model": model, "messages": messages, **kwargs}
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def chat_stream(self, messages: list[dict], model: str, **kwargs):
        import json

        import httpx

        payload = {"model": model, "messages": messages, "stream": True, **kwargs}
        with httpx.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=self._timeout,
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    if content := delta.get("content"):
                        yield content

    def embed(self, texts: list[str], model: str, **kwargs) -> list[list[float]]:
        import httpx

        payload = {"model": model, "input": texts, **kwargs}
        response = httpx.post(
            f"{self._base_url}/embeddings",
            headers=self._headers,
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda x: x["index"])
        return [item["embedding"] for item in data]


class OllamaClient:
    """Wraps the ollama Python SDK (lazy-loaded)."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._base_url = base_url.rstrip("/")

    @lru_cache(maxsize=1)
    def _ollama(self):
        import ollama as _ollama

        return _ollama.Client(host=self._base_url)

    def chat(self, messages: list[dict], model: str, **kwargs) -> str:
        client = self._ollama()
        response = client.chat(model=model, messages=messages, **kwargs)
        return response["message"]["content"]

    def embed(self, texts: list[str], model: str, **kwargs) -> list[list[float]]:
        client = self._ollama()
        response = client.embed(model=model, input=texts, **kwargs)
        return response["embeddings"]


class HFClient:
    """Wraps a sentence-transformers model (lazy-loaded, CPU)."""

    def __init__(self, model_name: str):
        self._model_name = model_name

    @lru_cache(maxsize=1)
    def _model(self):
        from sentence_transformers import CrossEncoder

        LOGGER.info(f"Loading HuggingFace model '{self._model_name}'...")
        return CrossEncoder(self._model_name)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._model()
        scores = model.predict(pairs)
        return [float(s) for s in scores]
