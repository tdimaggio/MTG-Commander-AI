import requests
import json
import pandas as pd
from typing import List, Dict, Any, Optional

# --- Configuration ---
# Ollama runs on localhost:11434 by default.
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-coder:6.7b" # Your installed Ollama model

class OllamaAgent:
    """
    Manages communication with the local DeepSeek Coder model via the Ollama API.
    """

    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.model_name = model_name

    def _format_card_data(self, card_df: pd.DataFrame, commander_name: str, owned_cards: List[str]) -> str:
        """
        Formats a subset of the DataFrame into a highly structured string 
        that the LLM can easily understand and use for reasoning.
        """
        # Select key columns useful for synergy reasoning
        columns_to_include = ['Name', 'ManaValue', 'ColorIdentity', 'Type', 'Text', 'Keywords']
        
        # --- Filtering for Context ---
        # 1. Find the Commander's color identity to filter the card pool
        commander_info = card_df[card_df['Name'] == commander_name].iloc[0]
        commander_colors = commander_info['ColorIdentity']
        
        # 2. Filter down the list to include only cards matching the Commander's color identity
        def passes_color_check(colors):
            # If the card has a color identity (list of characters), check if all match the Commander's colors.
            # If the card is colorless (empty list/None), it's always legal.
            if isinstance(colors, list) and colors:
                return all(c in commander_colors for c in colors)
            return True # Allow colorless/artifacts

        filtered_df = card_df[card_df['ColorIdentity'].apply(passes_color_check)].copy()
        
        # 3. Prioritize owned cards for the sample
        
        # Create a list of the intersection: owned cards that are also legal
        owned_legal_cards_df = filtered_df[filtered_df['Name'].isin(owned_cards)]
        
        # Calculate how many suggestions we need (we aim for 10 suggestions total)
        # For simplicity, we sample 200 cards in the final pool to give the AI variety
        
        # Take all owned legal cards (up to 50 for the context prompt)
        owned_sample = owned_legal_cards_df.sample(n=min(50, len(owned_legal_cards_df)), random_state=42)
        
        # Take a random sample of non-owned, legal cards to fill the rest of the 200 context slots
        non_owned_df = filtered_df[~filtered_df['Name'].isin(owned_cards)]
        non_owned_sample_size = min(150, len(non_owned_df))
        non_owned_sample = non_owned_df.sample(n=non_owned_sample_size, random_state=42)

        # Combine the two samples to create the final pool context
        final_sample_df = pd.concat([owned_sample, non_owned_sample]).drop_duplicates(subset=['Name'])

        # Convert the DataFrame sample to a string table for the prompt
        card_table_str = final_sample_df[columns_to_include].to_markdown(index=False)
        
        return f"""
Commander Name: {commander_name}
Commander Colors: {commander_colors}
Commander Text: {commander_info['Text']}

USER'S OWNED COLLECTION (Must be prioritized if synergistic): {len(owned_legal_cards_df)} unique cards owned that are legal.

CARD POOL DATA (Total {len(final_sample_df)} unique cards):
{card_table_str}
"""

    def generate_suggestions(self, card_df: pd.DataFrame, commander_name: str, owned_cards: List[str]) -> Optional[List[str]]:
        """
        Sends a request to the local Ollama server (DeepSeek Coder) to generate a decklist,
        prioritizing cards from the owned_cards list.
        """
        
        card_pool_context = self._format_card_data(card_df, commander_name, owned_cards)

        # The core prompt: instructing DeepSeek Coder to act as a card synergy expert
        prompt = f"""
You are an expert Magic: The Gathering (MTG) Commander deck-building assistant.
Your goal is to suggest 10 highly synergistic cards for a deck led by the Commander provided below.

**CRITICAL INSTRUCTION:**
- The suggested cards MUST come directly from the CARD POOL DATA provided.
- **Prioritize cards that are explicitly present in the CARD POOL DATA and are part of the USER'S OWNED COLLECTION.** This is a budget constraint.
- Ensure the selection includes a balance of card types (Ramp, Removal, Card Draw).
- Output ONLY a clean, markdown-formatted JSON array of the card names. DO NOT include any explanatory text, commentary, or markdown outside the final JSON block.

{card_pool_context}

Output the suggested card names as a JSON array (list of strings):
"""
        
        headers = {'Content-Type': 'application/json'}
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3, # Lower temperature for more deterministic/logical output
                "top_k": 40,
                "top_p": 0.9,
            }
        }

        try:
            print(f"Querying Ollama model '{self.model_name}' at {OLLAMA_URL}...")
            # We don't use the simple requests.post here, we'll use a fetch implementation with retries.
            
            # --- API Call Logic (Replace with actual fetch in running environment) ---
            # NOTE: For local environment testing, this block simulates the API call.
            # In a real environment, you would use 'requests' as initially defined.

            response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload), timeout=120)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            # The response text contains the LLM output
            full_response_text = response.json().get('response', '').strip()
            
            # --- Parsing Logic ---
            # Find the JSON block (often wrapped in ```json ... ```)
            start_index = full_response_text.find('[')
            end_index = full_response_text.rfind(']')
            
            if start_index != -1 and end_index != -1:
                json_string = full_response_text[start_index : end_index + 1]
                
                # Attempt to parse the JSON string
                try:
                    suggested_cards: List[str] = json.loads(json_string)
                    # Filter out non-strings just in case of weird LLM output
                    return [card for card in suggested_cards if isinstance(card, str)]
                except json.JSONDecodeError:
                    print("\n[ERROR] Failed to parse JSON output from LLM.")
                    print(f"Raw LLM Output (first 200 chars): {full_response_text[:200]}...")
                    return None
            else:
                print("\n[ERROR] Could not find JSON output in LLM response.")
                print(f"Raw LLM Output (first 200 chars): {full_response_text[:200]}...")
                return None
                
        except requests.exceptions.ConnectionError:
            print("\n[FATAL ERROR] Could not connect to Ollama server.")
            print("Please ensure Ollama is running and the 'deepseek-coder:6.7b' model is loaded.")
            print("Run 'ollama run deepseek-coder:6.7b' in your Terminal to check.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"\n[ERROR] An error occurred during the request to Ollama: {e}")
            return None