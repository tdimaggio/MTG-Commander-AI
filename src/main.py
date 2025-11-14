import pandas as pd
from typing import Optional, List, Dict, Any
import random

# Relative import structure for modules inside the 'src/' folder
from data_loader import load_and_preprocess_data, load_collection_data
from llm_agent import OllamaAgent

# Placeholder function that replaces the need for the AI to know all card names.
# In the final app, this function would use Pandas to query the database.
def select_cards_by_strategy(strategy: str, color_identity: str, owned_cards: List[str]) -> List[str]:
    """
    Simulates a complex database query to find card names matching a given strategy.
    
    Since DeepSeek Coder does not know card names, we hardcode a list based on the
    expected "Goblin Tribal" strategy for Krenko.
    """
    print(f"Executing Strategy Query: Find top cards for '{strategy}' in {color_identity}.")
    
    # The list below is hardcoded based on the expected "Goblin Tribal" strategy.
    # In a real app, this is where you would use Pandas to filter by card text/keywords.
    goblin_synergy_cards = [
        "Goblin Warchief",
        "Skirk Prospector",
        "Krenko, Tin Street Kingpin",
        "Impact Tremors",
        "Aether Vial",
        "Lightning Greaves",
        "Vandalblast",
        "Pashalik Mons",
        "Goblin Chieftain",
        "Ruby Medallion"
    ]
    
    # Filter to prioritize owned cards, then fill up with any strong suggestions.
    final_suggestions = [card for card in goblin_synergy_cards if card in owned_cards]
    
    # Fill the list up to 10 suggestions, excluding duplicates and the Commander itself.
    while len(final_suggestions) < 10 and len(goblin_synergy_cards) > 0:
        card = random.choice(goblin_synergy_cards)
        if card not in final_suggestions:
            final_suggestions.append(card)
    
    return final_suggestions[:10]


def select_commander(card_db: pd.DataFrame) -> Optional[Dict[str, Any]]:
    # --- Configuration ---
    DEFAULT_COMMANDER_NAME = "Krenko, Mob Boss"

    # Get user input for Commander name
    commander_name = input(f"\nPlease enter the name of your Commander (or press Enter for {DEFAULT_COMMANDER_NAME}): ")
    if not commander_name:
        commander_name = DEFAULT_COMMANDER_NAME

    # Search the database for the Commander
    commander_card = card_db[card_db['Name'] == commander_name]

    if commander_card.empty:
        print(f"Error: Commander '{commander_name}' not found in legal card database.")
        return None
    
    # Return the first (and hopefully only) match as a dictionary
    return commander_card.iloc[0].to_dict()

def run_deck_builder_app():
    print("--- MTG Commander AI Deck Builder Starting ---")

    # 1. Load Data
    card_database_df: Optional[pd.DataFrame] = load_and_preprocess_data()
    if card_database_df is None:
        print("Application stopped due to data loading error.")
        return

    # 2. Load Collection
    owned_cards: Optional[List[str]] = load_collection_data()
    if owned_cards is None:
        print("Application stopped due to collection loading error.")
        return

    # 3. Select Commander
    commander_card = select_commander(card_database_df)
    if commander_card is None:
        return

    # Display Confirmation (This should now show R, not ['R'] from the correct data_loader)
    print(f"\nCommander Selected: {commander_card['Name']} | Colors: {commander_card['ColorIdentity']}")
    print("\n--- Starting AI Card Suggestion Process ---")

    # 4. Initialize AI Agent and get the Strategy Command
    agent = OllamaAgent()
    
    # Get the strategy from the AI (e.g., {"function": "select_cards", "strategy": "Goblin Tribal"})
    strategy_command = agent.get_strategy_command(
        commander_name=commander_card['Name'],
        commander_color_identity=commander_card['ColorIdentity']
    )
    
    suggested_cards = None
    
    # 5. Execute the strategy command locally in Python
    if strategy_command and strategy_command.get('function') == 'select_cards':
        strategy = strategy_command.get('strategy', 'Unknown Strategy')
        
        suggested_cards = select_cards_by_strategy(
            strategy=strategy,
            color_identity=commander_card['ColorIdentity'],
            owned_cards=owned_cards
        )
    else:
        print("❌ AI failed to generate a valid strategy command. Check Ollama logs.")

    # 6. Display Results
    print("\n--- AI Suggested Cards ---")
    if suggested_cards:
        # Note: We now print the strategy the AI suggested before the list
        print(f"Strategy Determined by AI: {strategy_command.get('strategy')}")
        print("---")
        for i, card_name in enumerate(suggested_cards, 1):
            print(f" {i}. {card_name}")
        print("\n✅ Deck building assistance complete.")
    else:
        print("❌ AI suggestion failed or returned an empty list.")

if __name__ == "__main__":
    run_deck_builder_app()