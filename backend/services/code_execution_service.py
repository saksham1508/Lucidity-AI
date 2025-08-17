import asyncio
import subprocess
import tempfile
import os
import time
import docker
from typing import Dict, Any, Optional
import json
import sys
from pathlib import Path

from config import settings
from models.schemas import CodeExecutionRequest, CodeExecutionResponse

class CodeExecutionService:
    def __init__(self):
        self.supported_languages = {
            "python": {"extension": ".py", "command": ["python"]},
            "javascript": {"extension": ".js", "command": ["node"]},
            "typescript": {"extension": ".ts", "command": ["npx", "ts-node"]},
            "bash": {"extension": ".sh", "command": ["bash"]},
            "powershell": {"extension": ".ps1", "command": ["powershell", "-File"]},
            "sql": {"extension": ".sql", "command": ["sqlite3", ":memory:"]},
        }
        
        # Try to initialize Docker client for sandboxed execution
        self.docker_client = None
        try:
            self.docker_client = docker.from_env()
        except Exception:
            print("Docker not available, using local execution")
    
    async def execute_code(self, request: CodeExecutionRequest) -> CodeExecutionResponse:
        """Execute code in a sandboxed environment"""
        start_time = time.time()
        
        if not settings.enable_code_execution:
            return CodeExecutionResponse(
                output="",
                error="Code execution is disabled",
                execution_time=0.0,
                language=request.language,
                success=False
            )
        
        if request.language not in self.supported_languages:
            return CodeExecutionResponse(
                output="",
                error=f"Unsupported language: {request.language}",
                execution_time=0.0,
                language=request.language,
                success=False
            )
        
        try:
            if request.environment == "docker" and self.docker_client:
                result = await self._execute_in_docker(request)
            else:
                result = await self._execute_locally(request)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return CodeExecutionResponse(
                output="",
                error=f"Execution failed: {str(e)}",
                execution_time=execution_time,
                language=request.language,
                success=False
            )
    
    async def _execute_locally(self, request: CodeExecutionRequest) -> CodeExecutionResponse:
        """Execute code locally with timeout"""
        lang_config = self.supported_languages[request.language]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=lang_config["extension"],
            delete=False
        ) as temp_file:
            temp_file.write(request.code)
            temp_file_path = temp_file.name
        
        try:
            # Prepare command
            command = lang_config["command"] + [temp_file_path]
            
            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tempfile.gettempdir()
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=request.timeout
                )
                
                output = stdout.decode('utf-8') if stdout else ""
                error = stderr.decode('utf-8') if stderr else ""
                success = process.returncode == 0
                
                return CodeExecutionResponse(
                    output=output,
                    error=error if error else None,
                    execution_time=0.0,  # Will be set by caller
                    language=request.language,
                    success=success
                )
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return CodeExecutionResponse(
                    output="",
                    error=f"Execution timed out after {request.timeout} seconds",
                    execution_time=0.0,
                    language=request.language,
                    success=False
                )
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
    
    async def _execute_in_docker(self, request: CodeExecutionRequest) -> CodeExecutionResponse:
        """Execute code in Docker container for better sandboxing"""
        if not self.docker_client:
            return await self._execute_locally(request)
        
        try:
            # Choose appropriate Docker image
            image_map = {
                "python": "python:3.11-slim",
                "javascript": "node:18-slim",
                "typescript": "node:18-slim",
                "bash": "ubuntu:22.04",
            }
            
            image = image_map.get(request.language, "ubuntu:22.04")
            
            # Create temporary directory for code
            with tempfile.TemporaryDirectory() as temp_dir:
                lang_config = self.supported_languages[request.language]
                code_file = os.path.join(temp_dir, f"code{lang_config['extension']}")
                
                with open(code_file, 'w') as f:
                    f.write(request.code)
                
                # Prepare Docker command
                if request.language == "python":
                    cmd = ["python", f"/tmp/code{lang_config['extension']}"]
                elif request.language in ["javascript", "typescript"]:
                    cmd = ["node", f"/tmp/code{lang_config['extension']}"]
                else:
                    cmd = ["bash", f"/tmp/code{lang_config['extension']}"]
                
                # Run container
                container = self.docker_client.containers.run(
                    image,
                    cmd,
                    volumes={temp_dir: {'bind': '/tmp', 'mode': 'ro'}},
                    working_dir='/tmp',
                    network_mode='none',  # No network access
                    mem_limit='128m',     # Memory limit
                    cpu_quota=50000,      # CPU limit
                    timeout=request.timeout,
                    remove=True,
                    detach=False,
                    stdout=True,
                    stderr=True
                )
                
                output = container.decode('utf-8') if container else ""
                
                return CodeExecutionResponse(
                    output=output,
                    error=None,
                    execution_time=0.0,
                    language=request.language,
                    success=True
                )
                
        except docker.errors.ContainerError as e:
            return CodeExecutionResponse(
                output="",
                error=e.stderr.decode('utf-8') if e.stderr else str(e),
                execution_time=0.0,
                language=request.language,
                success=False
            )
        except Exception as e:
            return CodeExecutionResponse(
                output="",
                error=f"Docker execution failed: {str(e)}",
                execution_time=0.0,
                language=request.language,
                success=False
            )
    
    def get_supported_languages(self) -> Dict[str, Any]:
        """Get list of supported programming languages"""
        return {
            lang: {
                "extension": config["extension"],
                "description": f"{lang.title()} code execution",
                "docker_available": self.docker_client is not None
            }
            for lang, config in self.supported_languages.items()
        }
    
    async def validate_code(self, code: str, language: str) -> Dict[str, Any]:
        """Validate code syntax without executing"""
        if language == "python":
            try:
                compile(code, '<string>', 'exec')
                return {"valid": True, "error": None}
            except SyntaxError as e:
                return {"valid": False, "error": str(e)}
        elif language == "javascript":
            # Basic validation - could be enhanced with a JS parser
            if "eval(" in code or "Function(" in code:
                return {"valid": False, "error": "Potentially unsafe code detected"}
            return {"valid": True, "error": None}
        else:
            # For other languages, assume valid for now
            return {"valid": True, "error": None}