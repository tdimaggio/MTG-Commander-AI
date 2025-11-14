import pandas as pd
from typing import Optional, List, Dict, Any
import random
import re
import string

# Relative import structure for modules inside the 'src/' folder
from data_loader import load_and_preprocess_data, load_collection_data
from llm_agent import OllamaAgent

# --- HYPOTHETICAL PRICE DATA ---
# Since we don't have real card price data, we will use a small list of known 
# expensive Krenko staples to simulate the "Pricier" category for demonstration.
PRICEY_CARDS = {
    "Goblin Piledriver", "Kiki-Jiki, Mirror Breaker", "Gilded Drake", 
    "Imperial Recruiter", "Ancient Tomb", "Sneak Attack", "Chandra's Ignition"
}

def select_cards_by_strategy(
    card_database_df: pd.DataFrame, 
    strategy_keywords: List[str], 
    commander_color_identity: str, 
    owned_cards: List[str]
) -> Dict[str, List[str]]:
    """
    Filters the card DataFrame for cards matching the AI-deduced keywords, ranks them
    by relevance, and separates them into Owned, Missing Budget, and Missing Pricier lists.

    :param card_database_df: The DataFrame containing all legal MTG cards.
    :param strategy_keywords: The list of clean keywords provided by the AI (e.g., ['Goblin', 'Token', 'Haste']).
    :param commander_color_identity: The color identity string (e.g., 'R', 'UB').
    :param owned_cards: A list of unique card names owned by the user.
    :return: A dictionary with keys 'owned', 'missing_budget', and 'missing_pricier'.
    """
    print(f"Executing Strategy Query: Finding and ranking top cards matching keywords: {strategy_keywords} in {commander_color_identity}.")
    
    # 1. Keyword Standardization
    strategy_keywords = [word.lower() for word in strategy_keywords if word and len(word) > 1]
    if not strategy_keywords:
        return {'owned': [], 'missing_budget': [], 'missing_pricier': []}

    # 2. Color Filter (Ensures only on-color cards are considered)
    def is_within_colors(card_color_identity: str, commander_ci: str) -> bool:
        return card_color_identity == 'C' or all(color in commander_ci for color in card_color_identity)
        
    color_filter = card_database_df['ColorIdentity'].apply(
        lambda x: is_within_colors(x, commander_color_identity)
    )
    
    # Start with the color-filtered DataFrame
    synergy_cards_df = card_database_df[color_filter].copy()
    
    # 3. Keyword Scoring (The new ranking mechanism)
    
    def score_card(card_row: pd.Series, keywords: List[str]) -> int:
        """Scores a card based on how many keywords appear in its Name or Text."""
        score = 0
        card_text = str(card_row['Name']) + " " + str(card_row['Text'])
        card_text_lower = card_text.lower()
        
        for keyword in keywords:
            if keyword in card_text_lower:
                score += 1
        return score

    # Apply the scoring function to the filtered DataFrame
    synergy_cards_df['Score'] = synergy_cards_df.apply(
        lambda row: score_card(row, strategy_keywords), axis=1
    )
    
    # Filter out cards with a score of 0 and non-card entities
    synergy_cards_df = synergy_cards_df[synergy_cards_df['Score'] > 0]
    synergy_cards_df = synergy_cards_df[~synergy_cards_df['Name'].isin(["Token", "Emblem", "Scheme", "Krenko, Mob Boss"])]

    # 4. Final Ranking and Categorization
    
    # Sort: Highest Score first, then lowest ManaValue (for efficiency), then random shuffle for ties
    synergy_cards_df = synergy_cards_df.sort_values(
        by=['Score', 'ManaValue'], 
        ascending=[False, True]
    ).sample(frac=1).sort_values(by=['Score', 'ManaValue'], ascending=[False, True])
    
    
    all_suggestions = synergy_cards_df['Name'].unique().tolist()
    
    owned = []
    missing_budget = []
    missing_pricier = []
    
    # Separate the list into categories
    for card_name in all_suggestions:
        if card_name in owned_cards:
            if len(owned) < 10: # Limit owned suggestions
                owned.append(card_name)
        else:
            # Categorize missing cards based on the hypothetical price list
            if card_name in PRICEY_CARDS:
                if len(missing_pricier) < 5: # Limit pricier suggestions
                    missing_pricier.append(card_name)
            elif len(missing_budget) < 10: # Limit budget suggestions
                missing_budget.append(card_name)
                
    return {
        'owned': owned,
        'missing_budget': missing_budget,
        'missing_pricier': missing_pricier
    }


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

    # Display Confirmation
    # FIX for Color Identity Display
    color_id_display = commander_card['ColorIdentity']
    if isinstance(color_id_display, list):
        color_id_display = "".join(color_id_display) 

    print(f"\nCommander Selected: {commander_card['Name']} | Colors: {color_id_display}")
    print("\n--- Starting AI Card Suggestion Process ---")

    # 4. Initialize AI Agent and get the Strategy Command
    agent = OllamaAgent()
    
    # Get the strategy from the AI (now includes keywords list)
    strategy_command = agent.get_strategy_command(
        commander_name=commander_card['Name'],
        commander_color_identity=commander_card['ColorIdentity']
    )
    
    suggested_cards_dict = None
    strategy = 'Unknown Strategy'
    
    # 5. Execute the strategy command locally in Python
    if strategy_command and strategy_command.get('function') == 'select_cards':
        strategy = strategy_command.get('strategy', 'Unknown Strategy')
        strategy_keywords = strategy_command.get('keywords', []) # Extract the clean keywords list
        
        # Pass the clean keywords list to the filtering function
        suggested_cards_dict = select_cards_by_strategy(
            card_database_df=card_database_df, 
            strategy_keywords=strategy_keywords, 
            commander_color_identity=commander_card['ColorIdentity'],
            owned_cards=owned_cards
        )
    else:
        print("❌ AI failed to generate a valid strategy command. Check Ollama logs.")

    # 6. Display Results
    print("\n--- AI Card Suggestion Summary ---")
    if suggested_cards_dict:
        print(f"Strategy Determined by AI: **{strategy}**")
        
        # 6a. Owned Cards
        owned = suggested_cards_dict.get('owned', [])
        if owned:
            print(f"\n## ✅ Top Owned Cards ({len(owned)} suggestions)")
            print("These are the most relevant cards currently in your collection:")
            for i, card_name in enumerate(owned, 1):
                print(f" {i}. **{card_name}**")
        else:
            print("\n## ❌ No owned cards match the derived strategy.")
            
        # 6b. Missing Cards (Budget)
        budget = suggested_cards_dict.get('missing_budget', [])
        if budget:
            print(f"\n## 🛒 Budget Card Singles to Look Out For ({len(budget)} suggestions)")
            print("These are highly synergistic, budget-friendly cards that are missing from your collection:")
            for i, card_name in enumerate(budget, 1):
                print(f" {i}. **{card_name}**")
                
        # 6c. Missing Cards (Pricier)
        pricier = suggested_cards_dict.get('missing_pricier', [])
        if pricier:
            print(f"\n## 💎 Pricier Upgrade Options ({len(pricier)} suggestions)")
            print("These are powerful staples for this strategy that would significantly upgrade your deck:")
            for i, card_name in enumerate(pricier, 1):
                print(f" {i}. **{card_name}**")
                
        print("\n✅ Deck building assistance complete.")
    else:
        print("❌ AI suggestion failed or returned an empty list.")

if __name__ == "__main__":
    run_deck_builder_app()