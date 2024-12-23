#!/usr/bin/env python3
import os
import logging
import time
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import openai
from dotenv import load_dotenv
from metrics.collector import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CodingAgent:
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize the coding agent with interactive reflection capabilities.
        
        Args:
            model: The OpenAI model to use
        """
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Initialize metrics collector
        try:
            self.metrics = MetricsCollector(model_name=model)
        except Exception as e:
            logger.warning(f"Failed to initialize metrics collector: {e}")
            self.metrics = None

    def load_text_file(self, file_path: str) -> str:
        """Load content from a text file."""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return ""
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""

    def gather_memory(self, layers_needed: List[int]) -> str:
        """Gather content from memory layers."""
        memory_content = []
        base_dir = "memory"
        
        layer_files = {
            1: "layer_1_logic.md",
            2: "layer_2_concepts.md",
            3: "layer_3_important_details.md",
            4: "layer_4_arbitrary.md"
        }
        
        for layer in layers_needed:
            if layer in layer_files:
                if self.metrics:
                    self.metrics.log_layer_access(layer)
                content = self.load_text_file(os.path.join(base_dir, layer_files[layer]))
                if content:
                    memory_content.append(f"=== Layer {layer} Knowledge ===\n{content}\n")
        
        return "\n".join(memory_content)

    def get_completion(self, messages: List[Dict[str, str]], context_type: str) -> str:
        """Get completion from OpenAI API with metrics collection."""
        try:
            start_time = datetime.now()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            
            if self.metrics:
                duration = (datetime.now() - start_time).total_seconds()
                self.metrics.log_completion(
                    context_type=context_type,
                    duration=duration,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )
                
                logger.info(f"""
                    API Call Metrics ({context_type}):
                    Prompt Tokens: {response.usage.prompt_tokens}
                    Completion Tokens: {response.usage.completion_tokens}
                    Total Tokens: {response.usage.total_tokens}
                """)
            
            return content
            
        except Exception as e:
            logger.error(f"Error getting completion: {e}")
            return f"Error: {str(e)}"

    def _get_initial_solution(self, user_query: str) -> str:
        """Generate initial solution."""
        system_prompt = self.load_text_file("prompts/system_prompt.md")
        if not system_prompt:
            raise FileNotFoundError("System prompt file not found")

        layers_to_load = [1, 2, 3]
        if any(keyword in user_query.lower() for keyword in 
              ['legacy', 'old version', 'deprecated', 'edge case']):
            layers_to_load.append(4)

        memory_snippets = self.gather_memory(layers_to_load)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Relevant memory:\n{memory_snippets}"},
            {"role": "user", "content": user_query}
        ]
        
        return self.get_completion(messages, "initial_solution")

    def _get_reflection(self, solution: str) -> str:
        """Generate reflection analysis."""
        reflection_prompt = self.load_text_file("prompts/reflection_prompt.md")
        if not reflection_prompt:
            logger.warning("Reflection prompt file not found, using fallback")
            reflection_prompt = """
            Analyze this solution and provide specific improvements for:
            1. Logic and algorithm
            2. Design patterns and principles
            3. Implementation details
            4. Edge cases and optimization
            
            Format your response with concrete suggestions and code changes.
            """
        
        messages = [
            {"role": "system", "content": "You are a code review assistant providing actionable improvements."},
            {"role": "user", "content": f"{reflection_prompt}\n\nSolution to analyze:\n{solution}"}
        ]
        
        return self.get_completion(messages, "reflection")

    def refine_solution(self, original_solution: str, reflection_feedback: str, user_query: str) -> str:
        """Refine the solution based on reflection feedback."""
        refinement_prompt = f"""
        Original Task: {user_query}

        Initial Solution:
        {original_solution}

        Reflection Feedback:
        {reflection_feedback}

        Please improve the solution based on this reflection feedback. Specifically:
        1. Implement all high-priority suggestions
        2. Address the specific code changes recommended
        3. Add any missing error handling
        4. Incorporate the recommended optimizations
        
        Provide the improved solution with comments explaining the key improvements made.
        Ensure the solution maintains readability while implementing all suggested enhancements.
        """
        
        messages = [
            {"role": "system", "content": "You are a coding assistant tasked with improving a solution based on detailed feedback."},
            {"role": "user", "content": refinement_prompt}
        ]
        
        return self.get_completion(messages, "refinement")

    def run_agent_with_reflection(self, user_query: str) -> Tuple[str, str, str]:
        """
        Run the coding agent with interactive reflection and refinement.

        Args:
            user_query: The user's coding question or task

        Returns:
            Tuple of (initial_solution, reflection_analysis, improved_solution)
        """
        accessed_layers = set()  # Track accessed layers
        initial_solution = ""
        reflection = ""
        improved_solution = ""

        try:
            if self.metrics:
                self.metrics.start_interaction()
            
            # Generate initial solution
            logger.info("Generating initial solution...")
            initial_solution = self._get_initial_solution(user_query)
            if not initial_solution.startswith("Error"):
                # Only proceed with reflection if initial solution succeeded
                logger.info("Performing reflection analysis...")
                reflection = self._get_reflection(initial_solution)
                
                if not reflection.startswith("Error"):
                    # Only proceed with refinement if reflection succeeded
                    logger.info("Refining solution based on reflection...")
                    improved_solution = self.refine_solution(
                        initial_solution, reflection, user_query
                    )
            
            # Track which layers were accessed during the process
            accessed_layers = getattr(self.metrics, 'current_interaction', {}).get('layer_access', set())
            
            # Attempt to record metrics, but don't let metrics failure affect the response
            if self.metrics:
                try:
                    self.metrics.end_interaction(
                        prompt=user_query,
                        solution=improved_solution or initial_solution,
                        reflection=reflection,
                        layers_accessed=list(accessed_layers)
                    )
                except Exception as e:
                    logger.error(f"Failed to record metrics: {e}")
            
            return (
                initial_solution or "Error: Failed to generate initial solution",
                reflection or "No reflection analysis available",
                improved_solution or "No improvements made"
            )
                
        except Exception as e:
            logger.error(f"Error in run_agent_with_reflection: {e}")
            return (
                initial_solution or f"Error: {str(e)}",
                reflection or "Reflection failed.",
                improved_solution or "No improved solution available."
            )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        if self.metrics:
            return self.metrics.get_summary()
        return {}

def main():
    """Main function to run the agent interactively."""
    try:
        agent = CodingAgent()
        print("=== Coding Agent with Interactive Reflection ===")
        print("Enter your coding questions (type 'exit' or press Ctrl+C to quit)")
        print("Example: 'How do I implement a binary search tree in Python?'")
        
        while True:
            try:
                print("\nEnter your question:")
                user_input = input("> ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                
                print("\nProcessing your request...")
                initial_solution, reflection, improved_solution = agent.run_agent_with_reflection(user_input)
                
                print("\n=== Initial Solution ===")
                print(initial_solution)
                
                print("\n=== Reflection Analysis ===")
                print(reflection)
                
                print("\n=== Improved Solution ===")
                print(improved_solution)
                
                # Display metrics if available
                metrics = agent.get_metrics_summary()
                if metrics:
                    print("\n=== Metrics Summary ===")
                    total_tokens = metrics.get('total_tokens', 0)
                    print(f"Total Tokens Used: {total_tokens:,}")  # Fixed formatting
                    print(f"Total Processing Time: {metrics.get('total_time', 0):.2f}s")
                    print(f"Layers Accessed: {sorted(list(metrics.get('layers_accessed', [])))}")
                
                print("\n" + "="*50)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"\nError: {str(e)}")
    
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()