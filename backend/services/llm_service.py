import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import openai
import anthropic
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

from config import settings
from models.schemas import GenerationRequest, GenerationResponse, ModelProvider

class LLMService:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.local_models = {}
        
        # Initialize clients
        if settings.openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
            
        if settings.anthropic_api_key:
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using the specified model"""
        start_time = time.time()
        
        try:
            # Determine provider from model name
            provider = self._get_provider(request.model)
            
            if provider == ModelProvider.OPENAI:
                response = await self._generate_openai(request)
            elif provider == ModelProvider.ANTHROPIC:
                response = await self._generate_anthropic(request)
            elif provider == ModelProvider.LOCAL:
                response = await self._generate_local(request)
            else:
                raise ValueError(f"Unsupported model: {request.model}")
            
            generation_time = time.time() - start_time
            
            return GenerationResponse(
                content=response["content"],
                model=request.model,
                tokens_used=response.get("tokens_used", 0),
                generation_time=generation_time,
                finish_reason=response.get("finish_reason", "stop")
            )
            
        except Exception as e:
            raise Exception(f"Generation failed: {str(e)}")
    
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Stream text generation"""
        provider = self._get_provider(request.model)
        
        if provider == ModelProvider.OPENAI:
            async for chunk in self._generate_openai_stream(request):
                yield chunk
        elif provider == ModelProvider.ANTHROPIC:
            async for chunk in self._generate_anthropic_stream(request):
                yield chunk
        else:
            # For non-streaming models, yield the complete response
            response = await self.generate(request)
            yield response.content
    
    async def _generate_openai(self, request: GenerationRequest) -> Dict[str, Any]:
        """Generate using OpenAI models"""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False
            )
            
            return {
                "content": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            raise Exception(f"OpenAI generation failed: {str(e)}")
    
    async def _generate_openai_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Stream generation using OpenAI models"""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")
        
        try:
            stream = await self.openai_client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise Exception(f"OpenAI streaming failed: {str(e)}")
    
    async def _generate_anthropic(self, request: GenerationRequest) -> Dict[str, Any]:
        """Generate using Anthropic models"""
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not configured")
        
        try:
            response = await self.anthropic_client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=[{"role": "user", "content": request.prompt}]
            )
            
            return {
                "content": response.content[0].text,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                "finish_reason": response.stop_reason or "stop"
            }
            
        except Exception as e:
            raise Exception(f"Anthropic generation failed: {str(e)}")
    
    async def _generate_anthropic_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Stream generation using Anthropic models"""
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not configured")
        
        try:
            async with self.anthropic_client.messages.stream(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=[{"role": "user", "content": request.prompt}]
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            raise Exception(f"Anthropic streaming failed: {str(e)}")
    
    async def _generate_local(self, request: GenerationRequest) -> Dict[str, Any]:
        """Generate using local models"""
        model_name = request.model
        
        # Load model if not already loaded
        if model_name not in self.local_models:
            await self._load_local_model(model_name)
        
        try:
            model_info = self.local_models[model_name]
            tokenizer = model_info["tokenizer"]
            model = model_info["model"]
            
            # Tokenize input
            inputs = tokenizer.encode(request.prompt, return_tensors="pt")
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output
            response_text = generated_text[len(request.prompt):].strip()
            
            return {
                "content": response_text,
                "tokens_used": len(outputs[0]),
                "finish_reason": "stop"
            }
            
        except Exception as e:
            raise Exception(f"Local generation failed: {str(e)}")
    
    async def _load_local_model(self, model_name: str):
        """Load a local model"""
        try:
            # Map model names to HuggingFace model IDs
            model_mapping = {
                "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "llama-3-8b": "meta-llama/Llama-3-8B-Instruct",
                "phi-3": "microsoft/Phi-3-mini-4k-instruct",
                "gemma-7b": "google/gemma-7b-it"
            }
            
            hf_model_id = model_mapping.get(model_name, model_name)
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.local_models[model_name] = {
                "tokenizer": tokenizer,
                "model": model
            }
            
        except Exception as e:
            raise Exception(f"Failed to load local model {model_name}: {str(e)}")
    
    def _get_provider(self, model: str) -> ModelProvider:
        """Determine the provider for a given model"""
        if model.startswith("gpt-") or model.startswith("o1-"):
            return ModelProvider.OPENAI
        elif model.startswith("claude-"):
            return ModelProvider.ANTHROPIC
        elif model in ["mixtral-8x7b", "llama-3-8b", "phi-3", "gemma-7b"]:
            return ModelProvider.LOCAL
        else:
            # Default to OpenAI for unknown models
            return ModelProvider.OPENAI
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models by provider"""
        models = {
            "openai": [
                "gpt-4-turbo-preview",
                "gpt-4",
                "gpt-3.5-turbo",
                "o1-preview",
                "o1-mini"
            ] if self.openai_client else [],
            "anthropic": [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ] if self.anthropic_client else [],
            "local": [
                "mixtral-8x7b",
                "llama-3-8b",
                "phi-3",
                "gemma-7b"
            ]
        }
        
        return models
    
    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        provider = self._get_provider(model)
        
        model_info = {
            "name": model,
            "provider": provider.value,
            "max_tokens": self._get_max_tokens(model),
            "supports_streaming": True,
            "supports_function_calling": model.startswith("gpt-4") or model.startswith("claude-3"),
            "cost_per_1k_tokens": self._get_cost_per_1k_tokens(model)
        }
        
        return model_info
    
    def _get_max_tokens(self, model: str) -> int:
        """Get maximum tokens for a model"""
        token_limits = {
            "gpt-4-turbo-preview": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            "o1-preview": 128000,
            "o1-mini": 128000,
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "mixtral-8x7b": 32768,
            "llama-3-8b": 8192,
            "phi-3": 4096,
            "gemma-7b": 8192
        }
        
        return token_limits.get(model, 4096)
    
    def _get_cost_per_1k_tokens(self, model: str) -> float:
        """Get cost per 1k tokens for a model (input tokens)"""
        costs = {
            "gpt-4-turbo-preview": 0.01,
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.0015,
            "o1-preview": 0.015,
            "o1-mini": 0.003,
            "claude-3-opus-20240229": 0.015,
            "claude-3-sonnet-20240229": 0.003,
            "claude-3-haiku-20240307": 0.00025,
            # Local models are free
            "mixtral-8x7b": 0.0,
            "llama-3-8b": 0.0,
            "phi-3": 0.0,
            "gemma-7b": 0.0
        }
        
        return costs.get(model, 0.0)