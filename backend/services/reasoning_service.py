import time
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import sympy as sp
from sympy import symbols, solve, simplify, expand, factor
import numpy as np

from config import settings
from models.schemas import ReasoningRequest, ReasoningResponse, ReasoningStep, GenerationRequest
from .llm_service import LLMService

class ReasoningService:
    def __init__(self):
        self.llm_service = LLMService()
        
        # Reasoning patterns and templates
        self.reasoning_patterns = {
            'mathematical': [
                r'solve.*equation',
                r'calculate.*',
                r'find.*value',
                r'what.*is.*\+|\-|\*|\/|\^',
                r'\d+.*[\+\-\*\/\^].*\d+',
                r'derivative|integral|limit'
            ],
            'logical': [
                r'if.*then',
                r'therefore',
                r'because',
                r'since',
                r'given.*prove',
                r'assume.*conclude'
            ],
            'causal': [
                r'why.*',
                r'what.*causes',
                r'reason.*for',
                r'explain.*how',
                r'what.*leads.*to'
            ],
            'comparative': [
                r'compare.*',
                r'difference.*between',
                r'better.*than',
                r'versus|vs',
                r'pros.*cons'
            ]
        }
    
    async def solve(self, request: ReasoningRequest) -> ReasoningResponse:
        """Solve a problem using hybrid reasoning"""
        start_time = time.time()
        
        try:
            # Determine reasoning type if not specified
            if request.reasoning_type == "hybrid":
                reasoning_type = self._determine_reasoning_type(request.problem)
            else:
                reasoning_type = request.reasoning_type
            
            # Execute reasoning based on type
            if reasoning_type == "symbolic":
                solution, steps = await self._symbolic_reasoning(request.problem)
            elif reasoning_type == "neural":
                solution, steps = await self._neural_reasoning(request.problem, request.show_steps)
            else:  # hybrid
                solution, steps = await self._hybrid_reasoning(request.problem, request.show_steps)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(steps, request.confidence_threshold)
            
            reasoning_time = time.time() - start_time
            
            return ReasoningResponse(
                solution=solution,
                reasoning_steps=steps,
                confidence_score=confidence_score,
                reasoning_time=reasoning_time,
                method_used=reasoning_type
            )
            
        except Exception as e:
            # Fallback to neural reasoning if symbolic fails
            try:
                solution, steps = await self._neural_reasoning(request.problem, request.show_steps)
                confidence_score = self._calculate_confidence(steps, request.confidence_threshold)
                reasoning_time = time.time() - start_time
                
                return ReasoningResponse(
                    solution=solution,
                    reasoning_steps=steps,
                    confidence_score=confidence_score,
                    reasoning_time=reasoning_time,
                    method_used="neural_fallback"
                )
            except:
                raise Exception(f"Reasoning failed: {str(e)}")
    
    def _determine_reasoning_type(self, problem: str) -> str:
        """Determine the best reasoning approach for the problem"""
        problem_lower = problem.lower()
        
        # Check for mathematical patterns
        math_score = sum(1 for pattern in self.reasoning_patterns['mathematical'] 
                        if re.search(pattern, problem_lower))
        
        # Check for logical patterns
        logic_score = sum(1 for pattern in self.reasoning_patterns['logical'] 
                         if re.search(pattern, problem_lower))
        
        # If strong mathematical indicators, use symbolic
        if math_score >= 2 or any(re.search(pattern, problem_lower) 
                                 for pattern in [r'\d+.*[\+\-\*\/\^].*\d+', r'derivative|integral']):
            return "symbolic"
        
        # If strong logical indicators, use hybrid
        if logic_score >= 2:
            return "hybrid"
        
        # Default to neural for complex reasoning
        return "neural"
    
    async def _symbolic_reasoning(self, problem: str) -> Tuple[str, List[ReasoningStep]]:
        """Perform symbolic mathematical reasoning"""
        steps = []
        
        try:
            # Step 1: Parse mathematical expressions
            steps.append(ReasoningStep(
                step_number=1,
                description="Parsing mathematical expressions",
                reasoning_type="symbolic",
                confidence=0.9,
                intermediate_result="Identifying variables and equations"
            ))
            
            # Extract equations and variables
            equations, variables = self._extract_math_components(problem)
            
            if not equations:
                raise ValueError("No mathematical equations found")
            
            # Step 2: Set up symbolic computation
            steps.append(ReasoningStep(
                step_number=2,
                description=f"Setting up symbolic variables: {', '.join(variables)}",
                reasoning_type="symbolic",
                confidence=0.95,
                intermediate_result=f"Variables: {variables}"
            ))
            
            # Create symbolic variables
            sym_vars = {var: symbols(var) for var in variables}
            
            # Step 3: Solve equations
            solutions = []
            for i, eq in enumerate(equations):
                try:
                    # Convert string equation to sympy expression
                    sym_eq = self._string_to_sympy(eq, sym_vars)
                    
                    steps.append(ReasoningStep(
                        step_number=len(steps) + 1,
                        description=f"Solving equation {i+1}: {eq}",
                        reasoning_type="symbolic",
                        confidence=0.9,
                        intermediate_result=f"Equation: {sym_eq}"
                    ))
                    
                    # Solve the equation
                    if '=' in eq:
                        # Equation solving
                        lhs, rhs = eq.split('=')
                        lhs_expr = self._string_to_sympy(lhs.strip(), sym_vars)
                        rhs_expr = self._string_to_sympy(rhs.strip(), sym_vars)
                        equation = sp.Eq(lhs_expr, rhs_expr)
                        
                        solution = solve(equation, list(sym_vars.values()))
                        solutions.append(solution)
                        
                        steps.append(ReasoningStep(
                            step_number=len(steps) + 1,
                            description=f"Solution found: {solution}",
                            reasoning_type="symbolic",
                            confidence=0.95,
                            intermediate_result=str(solution)
                        ))
                    else:
                        # Expression evaluation
                        expr = self._string_to_sympy(eq, sym_vars)
                        simplified = simplify(expr)
                        solutions.append(simplified)
                        
                        steps.append(ReasoningStep(
                            step_number=len(steps) + 1,
                            description=f"Simplified expression: {simplified}",
                            reasoning_type="symbolic",
                            confidence=0.9,
                            intermediate_result=str(simplified)
                        ))
                        
                except Exception as e:
                    steps.append(ReasoningStep(
                        step_number=len(steps) + 1,
                        description=f"Error solving equation {i+1}: {str(e)}",
                        reasoning_type="symbolic",
                        confidence=0.3,
                        intermediate_result=f"Error: {str(e)}"
                    ))
            
            # Final solution
            if solutions:
                final_solution = f"Solutions: {solutions}"
            else:
                final_solution = "No symbolic solution found"
            
            return final_solution, steps
            
        except Exception as e:
            steps.append(ReasoningStep(
                step_number=len(steps) + 1,
                description=f"Symbolic reasoning failed: {str(e)}",
                reasoning_type="symbolic",
                confidence=0.1,
                intermediate_result=f"Error: {str(e)}"
            ))
            
            raise Exception(f"Symbolic reasoning failed: {str(e)}")
    
    async def _neural_reasoning(self, problem: str, show_steps: bool) -> Tuple[str, List[ReasoningStep]]:
        """Perform neural reasoning using LLM"""
        steps = []
        
        # Step 1: Problem analysis
        if show_steps:
            steps.append(ReasoningStep(
                step_number=1,
                description="Analyzing problem with neural reasoning",
                reasoning_type="neural",
                confidence=0.8,
                intermediate_result="Breaking down the problem"
            ))
        
        # Create reasoning prompt
        reasoning_prompt = f"""You are an advanced reasoning AI. Solve this problem step by step with clear logical reasoning.

Problem: {problem}

Please provide:
1. A clear analysis of what the problem is asking
2. Step-by-step reasoning process
3. Final answer with confidence level

Format your response as:
ANALYSIS: [Your analysis]
REASONING:
Step 1: [First step]
Step 2: [Second step]
...
ANSWER: [Final answer]
CONFIDENCE: [0.0-1.0]"""

        # Generate reasoning
        generation_request = GenerationRequest(
            prompt=reasoning_prompt,
            model=settings.default_model,
            temperature=0.3,  # Lower temperature for more consistent reasoning
            max_tokens=2000
        )
        
        response = await self.llm_service.generate(generation_request)
        
        # Parse the response
        solution, parsed_steps = self._parse_neural_response(response.content)
        
        if show_steps:
            steps.extend(parsed_steps)
        else:
            # Add summary step
            steps.append(ReasoningStep(
                step_number=1,
                description="Neural reasoning completed",
                reasoning_type="neural",
                confidence=0.8,
                intermediate_result=solution
            ))
        
        return solution, steps
    
    async def _hybrid_reasoning(self, problem: str, show_steps: bool) -> Tuple[str, List[ReasoningStep]]:
        """Perform hybrid symbolic + neural reasoning"""
        steps = []
        
        # Step 1: Try symbolic reasoning first
        try:
            if show_steps:
                steps.append(ReasoningStep(
                    step_number=1,
                    description="Attempting symbolic reasoning",
                    reasoning_type="hybrid",
                    confidence=0.7,
                    intermediate_result="Checking for mathematical patterns"
                ))
            
            symbolic_solution, symbolic_steps = await self._symbolic_reasoning(problem)
            
            # If symbolic reasoning succeeds, enhance with neural verification
            verification_prompt = f"""Verify this mathematical solution:

Problem: {problem}
Symbolic Solution: {symbolic_solution}

Is this solution correct? Explain any issues or confirm correctness."""

            generation_request = GenerationRequest(
                prompt=verification_prompt,
                model=settings.default_model,
                temperature=0.2,
                max_tokens=1000
            )
            
            verification = await self.llm_service.generate(generation_request)
            
            # Combine symbolic steps with neural verification
            steps.extend(symbolic_steps)
            steps.append(ReasoningStep(
                step_number=len(steps) + 1,
                description="Neural verification of symbolic solution",
                reasoning_type="hybrid",
                confidence=0.9,
                intermediate_result=verification.content[:200] + "..."
            ))
            
            return f"{symbolic_solution}\n\nVerification: {verification.content}", steps
            
        except:
            # Fall back to neural reasoning with structured approach
            if show_steps:
                steps.append(ReasoningStep(
                    step_number=len(steps) + 1,
                    description="Symbolic reasoning failed, using neural approach",
                    reasoning_type="hybrid",
                    confidence=0.6,
                    intermediate_result="Switching to neural reasoning"
                ))
            
            neural_solution, neural_steps = await self._neural_reasoning(problem, show_steps)
            steps.extend(neural_steps)
            
            return neural_solution, steps
    
    def _extract_math_components(self, problem: str) -> Tuple[List[str], List[str]]:
        """Extract mathematical equations and variables from problem text"""
        equations = []
        variables = set()
        
        # Find equations (expressions with = sign)
        equation_pattern = r'[^.!?]*[=][^.!?]*'
        found_equations = re.findall(equation_pattern, problem)
        
        for eq in found_equations:
            eq = eq.strip()
            if eq:
                equations.append(eq)
                # Extract variables (single letters, possibly with subscripts)
                vars_in_eq = re.findall(r'\b[a-zA-Z](?:_\w+)?\b', eq)
                variables.update(vars_in_eq)
        
        # Find standalone mathematical expressions
        math_pattern = r'\b\d+(?:\.\d+)?(?:\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?)+\b'
        math_expressions = re.findall(math_pattern, problem)
        equations.extend(math_expressions)
        
        # Common mathematical variables
        common_vars = re.findall(r'\b[xyz]\b', problem.lower())
        variables.update(common_vars)
        
        return equations, list(variables)
    
    def _string_to_sympy(self, expr_str: str, sym_vars: Dict[str, sp.Symbol]) -> sp.Expr:
        """Convert string expression to SymPy expression"""
        # Replace common mathematical notation
        expr_str = expr_str.replace('^', '**')  # Power notation
        expr_str = expr_str.replace('√', 'sqrt')  # Square root
        
        # Replace variables with symbolic variables
        for var_name, sym_var in sym_vars.items():
            expr_str = re.sub(r'\b' + var_name + r'\b', str(sym_var), expr_str)
        
        try:
            return sp.sympify(expr_str)
        except:
            # If sympify fails, try to parse as a simple expression
            return sp.parse_expr(expr_str, transformations='all')
    
    def _parse_neural_response(self, response: str) -> Tuple[str, List[ReasoningStep]]:
        """Parse neural reasoning response into solution and steps"""
        steps = []
        solution = ""
        
        try:
            # Extract sections
            analysis_match = re.search(r'ANALYSIS:\s*(.*?)(?=REASONING:|ANSWER:|$)', response, re.DOTALL)
            reasoning_match = re.search(r'REASONING:\s*(.*?)(?=ANSWER:|$)', response, re.DOTALL)
            answer_match = re.search(r'ANSWER:\s*(.*?)(?=CONFIDENCE:|$)', response, re.DOTALL)
            confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
            
            # Parse analysis
            if analysis_match:
                analysis = analysis_match.group(1).strip()
                steps.append(ReasoningStep(
                    step_number=1,
                    description="Problem analysis",
                    reasoning_type="neural",
                    confidence=0.8,
                    intermediate_result=analysis
                ))
            
            # Parse reasoning steps
            if reasoning_match:
                reasoning_text = reasoning_match.group(1).strip()
                step_matches = re.findall(r'Step\s+(\d+):\s*(.*?)(?=Step\s+\d+:|$)', reasoning_text, re.DOTALL)
                
                for step_num, step_desc in step_matches:
                    steps.append(ReasoningStep(
                        step_number=int(step_num) + len(steps),
                        description=f"Reasoning step {step_num}",
                        reasoning_type="neural",
                        confidence=0.8,
                        intermediate_result=step_desc.strip()
                    ))
            
            # Extract final answer
            if answer_match:
                solution = answer_match.group(1).strip()
            else:
                solution = response  # Use entire response if no clear answer section
            
            # Extract confidence if available
            if confidence_match:
                confidence = float(confidence_match.group(1))
                # Update confidence for all steps
                for step in steps:
                    step.confidence = min(step.confidence, confidence)
            
        except Exception as e:
            # Fallback parsing
            solution = response
            steps.append(ReasoningStep(
                step_number=1,
                description="Neural reasoning (parsed with fallback)",
                reasoning_type="neural",
                confidence=0.7,
                intermediate_result=response[:500] + "..." if len(response) > 500 else response
            ))
        
        return solution, steps
    
    def _calculate_confidence(self, steps: List[ReasoningStep], threshold: float) -> float:
        """Calculate overall confidence score for the reasoning"""
        if not steps:
            return 0.0
        
        # Weight different reasoning types
        type_weights = {
            "symbolic": 1.0,
            "neural": 0.8,
            "hybrid": 0.9
        }
        
        weighted_confidences = []
        for step in steps:
            weight = type_weights.get(step.reasoning_type, 0.7)
            weighted_confidences.append(step.confidence * weight)
        
        # Calculate weighted average
        overall_confidence = sum(weighted_confidences) / len(weighted_confidences)
        
        # Apply penalty for low-confidence steps
        low_confidence_penalty = sum(1 for step in steps if step.confidence < threshold) * 0.1
        overall_confidence = max(0.0, overall_confidence - low_confidence_penalty)
        
        return min(1.0, overall_confidence)
    
    async def explain_reasoning(self, solution: str, steps: List[ReasoningStep]) -> str:
        """Generate a natural language explanation of the reasoning process"""
        try:
            explanation_prompt = f"""Explain this reasoning process in clear, natural language:

Solution: {solution}

Reasoning Steps:
{chr(10).join([f"{i+1}. {step.description}: {step.intermediate_result}" for i, step in enumerate(steps)])}

Provide a clear, educational explanation of how we arrived at this solution."""

            generation_request = GenerationRequest(
                prompt=explanation_prompt,
                model=settings.default_model,
                temperature=0.4,
                max_tokens=1000
            )
            
            response = await self.llm_service.generate(generation_request)
            return response.content
            
        except Exception as e:
            return f"Unable to generate explanation: {str(e)}"
    
    def get_reasoning_capabilities(self) -> Dict[str, Any]:
        """Get information about reasoning capabilities"""
        return {
            "supported_types": ["symbolic", "neural", "hybrid"],
            "mathematical_capabilities": [
                "Equation solving",
                "Expression simplification",
                "Calculus operations",
                "Algebraic manipulation"
            ],
            "logical_capabilities": [
                "Deductive reasoning",
                "Inductive reasoning",
                "Causal analysis",
                "Comparative analysis"
            ],
            "hybrid_features": [
                "Symbolic verification with neural reasoning",
                "Fallback mechanisms",
                "Multi-step problem solving",
                "Confidence assessment"
            ]
        }