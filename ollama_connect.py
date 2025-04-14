import json
import os

import requests
from typing import List, Optional, Any

from dotenv import load_dotenv
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation, LLMResult
load_dotenv()

# Flexible configuration: use env var or fallback to host.docker.internal
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

OLLAMA_API_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"

class OllamaRESTLLM(LLM):
    model: str = OLLAMA_MODEL
    base_url: str = OLLAMA_API_URL#"http://localhost:11434/api/generate"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        response = requests.post(
            self.base_url,
            json={"model": self.model, "prompt": prompt},
            headers={"Content-Type": "application/json"},
            stream=True
        )

        if response.status_code != 200:
            raise Exception(f"Ollama call failed: {response.status_code}, {response.text}")

        output = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = line.decode("utf-8")
                    json_obj = json.loads(chunk)
                    output += json_obj.get("response", "")
                except Exception as e:
                    print(f"Chunk parse failed: {chunk}", e)

        return output.strip()

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
        generations = []
        for prompt in prompts:
            output = self._call(prompt, stop=stop)
            generations.append([Generation(text=output)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "ollama_rest"
