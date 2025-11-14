import requests
import json
from typing import List, Dict, Any, Optional
import re

# --- Configuration ---
# Ollama runs on localhost:11434 by default.
OLLAMA_URL = "http://localhost:11434/api/generate"
# Define the models for their specific roles
DEEPSEEK_MODEL = "deepseek-coder:6.7b" # Best for structured/coding-related tasks
MISTRAL_MODEL = "mistral" # Best for general-purpose reasoning/strategy deduction

class OllamaAgent:
    """
    Manages communication with the local Ollama models.
    """
    # Initialize with the strategy model (Mistral) as default for the main task
    def __init__(self, strategy_model_name: str = MISTRAL_MODEL):
        self.strategy_model_name = strategy_model_name

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts and parses the command JSON object from the LLM's response.
        We now expect: {"function": "select_cards", "strategy": "Goblin Tribal", "keywords": ["Goblin", "Token", "Haste"]}
        """
        # Look for a JSON object structure enclosed in curly braces { ... }
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        
        if match:
            raw_json_str = match.group(0).strip()
            
            # 2. Attempt to parse the cleaned string into a Python dictionary
            try:
                command = json.loads(raw_json_str)
                # Basic validation to ensure it looks like a command
                if (isinstance(command, dict) and 
                    'function' in command and 
                    'strategy' in command and 
                    'keywords' in command and # Check for new keywords field
                    isinstance(command['keywords'], list)): # Ensure keywords is a list
                    return command
            except json.JSONDecodeError as e:
                print(f"DEBUG: Failed to decode command JSON. Error: {e}")
                return None
        
        return None

    def get_strategy_command(self, commander_name: str, commander_color_identity: str) -> Optional[Dict[str, Any]]:
        """
        Queries the LLM for a structured command outlining the deck strategy, 
        including a dedicated list of search keywords.
        """
        
        system_prompt = (
            "You are an expert Magic: The Gathering Commander deck builder. "
            "Your task is to determine the core synergistic strategy and a concise list of 3-5 "
            "search keywords for the Commander. "
            "Respond ONLY with a single, valid JSON object. "
            "Do NOT include any commentary, prose, or markdown fences (```)."
        )

        user_prompt = (
            f"COMMANDER: {commander_name} (Color Identity: {commander_color_identity}). "
            "Based on this card, what is the single best, most synergistic deck strategy? "
            "Provide the strategy concisely (e.g., 'Voltron', 'Lifegain', 'Artifact Ramp') AND "
            "the 3-5 most critical keywords to search for in a card's name or text (e.g., ['Goblin', 'Token', 'Haste'])."
        )
        
        # Example to force the model into the required output format
        function_schema = (
            'The required output format is a JSON object defining the function call:\n'
            '{"function": "select_cards", "strategy": "<CONCISE_STRATEGY_HERE>", "keywords": ["KW1", "KW2", "KW3"]}'
        )

        payload = {
            "model": self.strategy_model_name,
            "prompt": f"{system_prompt}\n{function_schema}\n\nUSER COMMAND: {user_prompt}",
            "stream": False,
            "format": "json", 
            "options": {
                "temperature": 0.1, 
                "num_predict": 1024
            }
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=30)
            response.raise_for_status()
            
            response_json = response.json()
            full_response_text = response_json.get('response', '')
            
            # Parse the response to extract the command
            return self._parse_json_response(full_response_text)

        except requests.exceptions.ConnectionError:
            print(f"❌ ERROR: Could not connect to Ollama server at {OLLAMA_URL}.")
            print(f"Please ensure Ollama is running and the '{self.strategy_model_name}' model is pulled.")
            return None
        except Exception as e:
            print(f"❌ An unexpected error occurred during command generation: {e}")
            return None