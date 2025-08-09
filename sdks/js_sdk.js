// Lucidity AI JS SDK
// A simple client for interacting with the Lucidity AI backend API.

export class LucidityAIClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  async search(query) {
    const res = await fetch(`${this.baseUrl}/search?query=${encodeURIComponent(query)}`);
    return res.json();
  }

  async rag(query) {
    const res = await fetch(`${this.baseUrl}/rag`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    return res.json();
  }

  async generate(prompt, modelName = 'mixtral') {
    const res = await fetch(`${this.baseUrl}/model/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, model_name: modelName })
    });
    return res.json();
  }

  async multimodal(fileType) {
    const res = await fetch(`${this.baseUrl}/multimodal/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_type: fileType })
    });
    return res.json();
  }

  async getMemory(userId) {
    const res = await fetch(`${this.baseUrl}/memory/get?user_id=${encodeURIComponent(userId)}`);
    return res.json();
  }

  async getProfile(userId) {
    const res = await fetch(`${this.baseUrl}/profile/get?user_id=${encodeURIComponent(userId)}`);
    return res.json();
  }
}
