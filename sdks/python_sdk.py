"""
Lucidity AI Python SDK
A simple client for interacting with the Lucidity AI backend API.
"""
import requests

class LucidityAIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def search(self, query: str):
        resp = requests.get(f"{self.base_url}/search", params={"query": query})
        return resp.json()

    def rag(self, query: str):
        resp = requests.post(f"{self.base_url}/rag", json={"query": query})
        return resp.json()

    def generate(self, prompt: str, model_name: str = "mixtral"):
        resp = requests.post(f"{self.base_url}/model/generate", json={"prompt": prompt, "model_name": model_name})
        return resp.json()

    def multimodal(self, file_type: str):
        resp = requests.post(f"{self.base_url}/multimodal/analyze", json={"file_type": file_type})
        return resp.json()

    def get_memory(self, user_id: str):
        resp = requests.get(f"{self.base_url}/memory/get", params={"user_id": user_id})
        return resp.json()

    def get_profile(self, user_id: str):
        resp = requests.get(f"{self.base_url}/profile/get", params={"user_id": user_id})
        return resp.json()
