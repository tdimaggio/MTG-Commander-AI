import pandas as pd
from data_loader import load_and_preprocess_data, load_collection_data 
from llm_agent import OllamaAgent 
from typing import Optional, List, Any

# --- Configuration ---
# Set the initial Commander Name (User choice)
INITIAL_COMMANDER_NAME = "Krenko, Mob Boss"

def get_user_commander() -> str:
    """Prompts the user for the Commander's name."""
    print("\n-------------------------------------------")
    print(f"Commander set initially to: {INITIAL_COMMANDER_NAME}")
    
    # Allow the user to change the commander interactively
    user_input = input("Please enter the name of your Commander (or press Enter for Krenko, Mob Boss): ").strip()
    
    return user_input if user_input else INITIAL_COMMANDER_NAME

def select_commander(card_db: pd.DataFrame, commander_name: str) -> Optional[pd.Series]:
    """
    Finds the Commander in the card database and extracts key information.
    """
    
    # Attempt to find the commander in the database
    commander_card = card_db[card_db['Name'] == commander_name]
    
    if commander_card.empty:
        print(f"Error: Commander '{commander_name}' not found in the legal card database.")
        return None
    
    commander_info: pd.Series = commander_card.iloc[0]
    
    # Format colors for display
    colors = commander_info.get('ColorIdentity')
    colors_str = ','.join(colors) if isinstance(colors, list) and colors else 'C (Colorless)'
        
    print(f"\nCommander Selected: {commander_info['Name']} | Colors: {colors_str}")
    
    return commander_info

def run_deck_builder_app():
    """
    Main function to run the MTG Commander AI Deck Builder application.
    Orchestrates data loading, commander selection, and AI suggestion generation.
    """
    print("--- MTG Commander AI Deck Builder Starting ---")
    
    # 1. Load the pre-processed card database (Data Layer)
    card_database_df: pd.DataFrame = load_and_preprocess_data()
    
    if card_database_df is None:
        print("\nFATAL ERROR: Could not load card data. Aborting.")
        return

    print(f"✅ Card Database Loaded Successfully. Total legal cards: {len(card_database_df)}")
    
    # 2. Load user's card collection
    owned_cards: List[str] = load_collection_data()
    
    # 3. Get Commander name from user input
    commander_name = get_user_commander()
    
    # 4. Select the Commander
    commander_card = select_commander(card_database_df, commander_name)
    if commander_card is None:
        return
        
    # 5. Initialize the LLM Agent (AI Layer)
    # The OllamaAgent is designed to pass the data to the DeepSeek Coder model
    agent = OllamaAgent()
    
    # 6. Generate Card Suggestions using the LLM Agent
    print("\n--- Starting AI Card Suggestion Process ---")
    
    # Pass the full DataFrame, Commander Name, AND the list of owned cards
    suggested_cards: Optional[List[str]] = agent.generate_suggestions(
        card_df=card_database_df, 
        commander_name=commander_card['Name'],
        owned_cards=owned_cards # Passing the collection data
    )
    
    if suggested_cards:
        print("\n--- AI Suggested Cards ---")
        for i, card in enumerate(suggested_cards):
            print(f"  {i+1}. {card}")
        
        print("\n✅ AI suggestion successful! Deck building assistance complete.")
    else:
        print("\n❌ AI suggestion failed. Review the error messages (Ollama connection or JSON parsing issues).")


if __name__ == "__main__":
    run_deck_builder_app()