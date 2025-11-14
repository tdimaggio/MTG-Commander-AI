import json
import pandas as pd
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

# --- Configuration ---
# Define the root directory and file paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE_PATH = DATA_DIR / "AtomicCards.json"
OUTPUT_FILE_PATH = DATA_DIR / "commander_legal_cards.csv"
COLLECTION_FILE_PATH = DATA_DIR / "1125_collection.csv" # New path for the user's collection

def load_and_preprocess_data() -> Optional[pd.DataFrame]:
    """
    Loads AtomicCards.json, flattens the nested structure, and filters 
    for Commander-legal cards, returning a clean Pandas DataFrame.
    """
    # 1. Ensure the data directory exists
    DATA_DIR.mkdir(exist_ok=True)

    # 2. Check for file existence before proceeding
    if not DATA_FILE_PATH.exists():
        print(f"Error: Data file not found at '{DATA_FILE_PATH}'")
        print("Please download AtomicCards.json and place it in the project's 'data/' folder.")
        return None 

    # 3. Check if the pre-processed CSV already exists to speed up loading
    if OUTPUT_FILE_PATH.exists():
        print(f"Loading pre-processed data from {OUTPUT_FILE_PATH}...")
        return pd.read_csv(OUTPUT_FILE_PATH)

    print(f"Loading raw JSON data from {DATA_FILE_PATH}...")

    # 4. Load the deeply nested JSON data
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            data: Dict[str, Any] = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {DATA_FILE_PATH}. File may be corrupted.")
        return None

    # MTGJSON structure is: { 'meta': {...}, 'data': { 'Card Name': [{...}], ... } }
    card_data: Dict[str, List[Dict[str, Any]]] = data.get('data', {})

    # 5. Flatten the dictionary of cards into a list of card objects
    card_list: List[Dict[str, Any]] = []
    for card_name, details in card_data.items():
        card_details = details[0] if isinstance(details, list) and details else details
        if not isinstance(card_details, dict):
            continue

        card_details['card_name'] = card_name
        card_list.append(card_details)

    # 6. Create the DataFrame and flatten complex nested columns like 'legalities'
    print("Creating and normalizing DataFrame...")
    df = pd.json_normalize(card_list)

    # --- Data Cleaning and Commander Filtering ---

    # 7. Filter for Commander Legality
    if 'legalities.commander' not in df.columns:
        print("Warning: 'legalities.commander' column not found. Skipping legality filter.")
        commander_legal_df = df.copy()
    else:
        commander_legal_df = df[df['legalities.commander'] == 'Legal'].copy().dropna(subset=['legalities.commander'])

    print(f"\nTotal unique cards loaded: {len(df)}")
    print(f"Commander-legal cards found: {len(commander_legal_df)}")

    # 8. Select and rename core columns for easier coding later
    final_columns = {
        'card_name': 'Name',
        'colorIdentity': 'ColorIdentity',
        'manaValue': 'ManaValue',
        'type': 'Type',
        'text': 'Text',
        'keywords': 'Keywords',
        'power': 'Power',
        'toughness': 'Toughness',
        'legalities.commander': 'CommanderLegality',
    }

    # Select columns safely and rename them
    cols_to_keep = [col for col, new_name in final_columns.items() if col in commander_legal_df.columns]
    final_df = commander_legal_df[cols_to_keep].rename(columns={col: final_columns[col] for col in cols_to_keep})

    # --- CRITICAL FIX FOR COLOR IDENTITY ---
    # This fixes the bug where cards like Krenko were read as Colorless ('C').
    # It converts the list of colors (e.g., ['R']) into a joined string ('R').
    final_df['ColorIdentity'] = final_df['ColorIdentity'].apply(
        lambda x: "".join(sorted(x)) if isinstance(x, list) and x else 'C'
    )
    
    # Add back any missing core columns (important for later AI features)
    for new_name in final_columns.values():
        if new_name not in final_df.columns:
            default_value = [] if new_name == 'Keywords' else ''
            final_df[new_name] = default_value

    # 9. Save the clean, filtered data as a CSV for fast loading in the future
    final_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"\nCleaned data saved to {OUTPUT_FILE_PATH} (for faster loading next time).")

    return final_df

def load_collection_data() -> Optional[List[str]]:
    """
    Loads card names from the user's local Moxfield CSV file (1125_collection.csv).
    Returns a list of unique card names owned by the user.
    """
    if not COLLECTION_FILE_PATH.exists():
        print(f"Warning: Collection file not found at '{COLLECTION_FILE_PATH}'. Skipping collection filter.")
        return []

    print(f"Loading user collection from {COLLECTION_FILE_PATH}...")
    try:
        collection_df = pd.read_csv(COLLECTION_FILE_PATH)
        
        # We only need the names of the cards, filtered by having a positive count
        owned_cards = collection_df[collection_df['Count'] > 0]['Name'].unique().tolist()
        
        print(f"✅ User collection loaded. Total unique owned cards: {len(owned_cards)}")
        return owned_cards

    except Exception as e:
        print(f"Error loading collection CSV: {e}")
        return []

if __name__ == "__main__":
    # This ensures that when the script is run directly, it will load the data.
    card_db = load_and_preprocess_data()
    collection = load_collection_data()

    if card_db is not None:
        print("\n--- Sample of the Cleaned Data (First 5 Rows) ---")
        print(card_db.head())