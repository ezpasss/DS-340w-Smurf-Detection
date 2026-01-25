import json
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILE = 'Full_Data.jsonl'
OUTPUT_X = 'X_47.npy'
OUTPUT_Y = 'y_47.npy'

# The Exact List of 47 Features you provided
FEATURE_LIST = [
    'championStats_abilityHaste', 'championStats_abilityPower', 'championStats_armor', 
    'championStats_armorPen', 'championStats_armorPenPercent', 'championStats_attackDamage', 
    'championStats_attackSpeed', 'championStats_bonusArmorPenPercent', 'championStats_bonusMagicPenPercent', 
    'championStats_ccReduction', 'championStats_cooldownReduction', 'championStats_health', 
    'championStats_healthMax', 'championStats_healthRegen', 'championStats_lifesteal', 
    'championStats_magicPen', 'championStats_magicPenPercent', 'championStats_magicResist', 
    'championStats_movementSpeed', 'championStats_omnivamp', 'championStats_physicalVamp', 
    'championStats_power', 'championStats_powerMax', 'championStats_powerRegen', 
    'championStats_spellVamp', 'currentGold', 'damageStats_magicDamageDone', 
    'damageStats_magicDamageDoneToChampions', 'damageStats_magicDamageTaken', 'damageStats_physicalDamageDone', 
    'damageStats_physicalDamageDoneToChampions', 'damageStats_physicalDamageTaken', 'damageStats_totalDamageDone', 
    'damageStats_totalDamageDoneToChampions', 'damageStats_totalDamageTaken', 'damageStats_trueDamageDone', 
    'damageStats_trueDamageDoneToChampions', 'damageStats_trueDamageTaken', 'goldPerSecond', 
    'jungleMinionsKilled', 'level', 'minionsKilled', 'position_x', 
    'position_y', 'timeEnemySpentControlled', 'totalGold', 'xp'
]

RANK_MAP = {
    'IRON': 0, 'BRONZE': 1, 'SILVER': 2, 'GOLD': 3, 'PLATINUM': 4,
    'EMERALD': 5, 'DIAMOND': 6, 'MASTER': 7, 'GRANDMASTER': 8, 'CHALLENGER': 9
}

def get_nested_value(data_dict, feature_name):
    """
    Automatically traverses dictionaries based on underscores.
    Example: 'championStats_armor' -> data_dict['championStats']['armor']
    Example: 'totalGold' -> data_dict['totalGold']
    """
    try:
        # Handle special case: 'position_x' -> ['position']['x']
        if feature_name == 'position_x':
            return float(data_dict.get('position', {}).get('x', 0))
        if feature_name == 'position_y':
            return float(data_dict.get('position', {}).get('y', 0))
            
        # Handle standard nesting (championStats_armor)
        if '_' in feature_name:
            parts = feature_name.split('_')
            # Assuming 1 level of nesting based on your list
            category = parts[0] # e.g., championStats
            key = parts[1]      # e.g., armor
            return float(data_dict.get(category, {}).get(key, 0))
            
        # Handle top-level keys (totalGold, xp, level)
        else:
            return float(data_dict.get(feature_name, 0))
            
    except (ValueError, TypeError, AttributeError):
        return 0.0

def process_47_features():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run the RAW scraper first.")
        return

    X_list = []
    y_list = []

    print(f"Processing {INPUT_FILE} into 47-Feature Matrix...")

    with open(INPUT_FILE, 'r') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line)
                
                if data['tier'] not in RANK_MAP: continue
                label = RANK_MAP[data['tier']]
                
                timeline = data.get('timeline')
                if not timeline or len(timeline) < 26: continue

                # Iterate Players (1-10)
                for pid in range(1, 11):
                    seq = []
                    valid_player = True
                    
                    # Iterate Minutes (0-25)
                    for frame_obj in timeline[:26]:
                        try:
                            # Access the raw participant data
                            p_data = frame_obj['participantFrames'][str(pid)]
                            
                            # BUILD THE VECTOR (The 47 Features)
                            vector = []
                            for feature in FEATURE_LIST:
                                val = get_nested_value(p_data, feature)
                                vector.append(val)
                            
                            seq.append(vector)
                            
                        except KeyError:
                            valid_player = False
                            break
                    
                    if valid_player and len(seq) == 26:
                        X_list.append(seq)
                        y_list.append(label)

            except json.JSONDecodeError:
                continue

    # Convert to Numpy
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print("\n" + "="*40)
    print("      PROCESSING COMPLETE      ")
    print("="*40)
    print(f"Total Samples: {len(y)}")
    print(f"X Shape: {X.shape} -> (Samples, 26, 47)")
    print(f"y Shape: {y.shape}")
    
    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)
    print(f"[✓] Saved files with 47 features.")

if __name__ == "__main__":
    process_47_features()