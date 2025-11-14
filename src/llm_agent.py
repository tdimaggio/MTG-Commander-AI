import requests
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import re

# --- Configuration ---
# Ollama runs on localhost:11434 by default.
# NOTE: These are defined as GLOBAL constants, not self attributes.
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-coder:6.7b" # Your installed Ollama model

class OllamaAgent:
    """
    Manages communication with the local DeepSeek Coder model via the Ollama API
    to return structured JSON commands instead of card names.
    """
    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.model_name = model_name

    def _parse_json_response(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extracts and parses the command JSON object from the LLM's response.
        We expect a JSON object like: {"function": "select_cards", "strategy": "Goblin Tribal"}
        """
        # Look for a JSON object structure enclosed in curly braces { ... }
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        
        if match:
            raw_json_str = match.group(0).strip()
            
            # 2. Attempt to parse the cleaned string into a Python dictionary
            try:
                command = json.loads(raw_json_str)
                # Basic validation to ensure it looks like a command
                if isinstance(command, dict) and 'function' in command and 'strategy' in command:
                    return command
            except json.JSONDecodeError as e:
                print(f"DEBUG: Failed to decode command JSON. Error: {e}")
                return None
        
        return None

    def get_strategy_command(self, commander_name: str, commander_color_identity: str) -> Optional[Dict[str, str]]:
        """
        Queries the LLM for a structured command outlining the deck strategy.
        """
        
        system_prompt = (
            "You are an expert Magic: The Gathering Commander deck builder. "
            "Your sole task is to determine the core synergistic strategy for the Commander. "
            "Respond ONLY with a single, valid JSON object that defines the function to call and the strategy. "
            "Do NOT include any commentary, prose, or markdown fences (```)."
        )

        user_prompt = (
            f"COMMANDER: {commander_name} (Color Identity: {commander_color_identity}). "
            "Based on this card, what is the single best, most synergistic deck strategy? "
            "Define the strategy concisely (e.g., 'Voltron', 'Lifegain', 'Artifact Ramp')."
        )
        
        # Example to force the model into the required output format
        function_schema = (
            'The required output format is a JSON object defining the function call:\n'
            '{"function": "select_cards", "strategy": "<CONCISE_STRATEGY_HERE>"}'
        )

        payload = {
            "model": self.model_name,
            "prompt": f"{system_prompt}\n{function_schema}\n\nUSER COMMAND: {user_prompt}",
            "stream": False,
            "format": "json", # CRITICAL FIX: Ollama's native way to enforce JSON output
            "options": {
                "temperature": 0.1, # Very low temperature for highly deterministic output
                "num_predict": 1024
            }
        }

        try:
            # FIX: Access OLLAMA_URL as a global constant
            response = requests.post(OLLAMA_URL, json=payload, timeout=30)
            response.raise_for_status()
            
            response_json = response.json()
            full_response_text = response_json.get('response', '')
            
            # Parse the response to extract the command
            return self._parse_json_response(full_response_text)

        except requests.exceptions.ConnectionError:
            print("❌ ERROR: Could not connect to Ollama server.")
            return None
        except Exception as e:
            print(f"❌ An unexpected error occurred during command generation: {e}")
            return None