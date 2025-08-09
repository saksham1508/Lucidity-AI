// Lucidity AI Rust SDK
// A simple client for interacting with the Lucidity AI backend API.

use reqwest::blocking::Client;
use serde_json::Value;

pub struct LucidityAIClient {
    base_url: String,
    client: Client,
}

impl LucidityAIClient {
    pub fn new(base_url: &str) -> Self {
        LucidityAIClient {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::new(),
        }
    }

    pub fn search(&self, query: &str) -> Result<Value, reqwest::Error> {
        let url = format!("{}/search", self.base_url);
        let resp = self.client.get(&url).query(&[("query", query)]).send()?.json()?;
        Ok(resp)
    }

    pub fn rag(&self, query: &str) -> Result<Value, reqwest::Error> {
        let url = format!("{}/rag", self.base_url);
        let resp = self.client.post(&url).json(&serde_json::json!({"query": query})).send()?.json()?;
        Ok(resp)
    }

    pub fn generate(&self, prompt: &str, model_name: &str) -> Result<Value, reqwest::Error> {
        let url = format!("{}/model/generate", self.base_url);
        let resp = self.client.post(&url).json(&serde_json::json!({"prompt": prompt, "model_name": model_name})).send()?.json()?;
        Ok(resp)
    }

    pub fn multimodal(&self, file_type: &str) -> Result<Value, reqwest::Error> {
        let url = format!("{}/multimodal/analyze", self.base_url);
        let resp = self.client.post(&url).json(&serde_json::json!({"file_type": file_type})).send()?.json()?;
        Ok(resp)
    }

    pub fn get_memory(&self, user_id: &str) -> Result<Value, reqwest::Error> {
        let url = format!("{}/memory/get", self.base_url);
        let resp = self.client.get(&url).query(&[("user_id", user_id)]).send()?.json()?;
        Ok(resp)
    }

    pub fn get_profile(&self, user_id: &str) -> Result<Value, reqwest::Error> {
        let url = format!("{}/profile/get", self.base_url);
        let resp = self.client.get(&url).query(&[("user_id", user_id)]).send()?.json()?;
        Ok(resp)
    }
}
