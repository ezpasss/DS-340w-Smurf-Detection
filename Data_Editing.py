import json
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILE = "Full_Data.jsonl"
OUTPUT_X = "X_47_t60.npy"
OUTPUT_Y = "y_47_t60.npy"

MAX_T = 60          # target timesteps (pad/truncate to this)
PAD_VALUE = 0.0     # value to pad missing minutes with

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
    try:
        if feature_name == 'position_x':
            return float(data_dict.get('position', {}).get('x', 0))
        if feature_name == 'position_y':
            return float(data_dict.get('position', {}).get('y', 0))

        if '_' in feature_name:
            parts = feature_name.split('_')
            category = parts[0]
            key = parts[1]
            return float(data_dict.get(category, {}).get(key, 0))
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
    print(f"Padding/truncating to MAX_T={MAX_T} with PAD_VALUE={PAD_VALUE}")

    with open(INPUT_FILE, "r") as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line)

                tier = data.get("tier")
                if tier not in RANK_MAP:
                    continue
                label = RANK_MAP[tier]

                timeline = data.get("timeline")
                if not timeline or len(timeline) == 0:
                    continue

                # Truncate if longer than MAX_T
                timeline_trunc = timeline[:MAX_T]

                # Iterate Players (1-10)
                for pid in range(1, 11):
                    seq = []
                    valid_player = True

                    # Build exactly MAX_T timesteps
                    for t in range(MAX_T):
                        if t < len(timeline_trunc):
                            frame_obj = timeline_trunc[t]

                            pf = frame_obj.get("participantFrames")
                            if not isinstance(pf, dict):
                                vector = [PAD_VALUE] * len(FEATURE_LIST)
                            else:
                                p_data = pf.get(str(pid))
                                if not isinstance(p_data, dict):
                                    vector = [PAD_VALUE] * len(FEATURE_LIST)
                                else:
                                    vector = []
                                    for feature in FEATURE_LIST:
                                        vector.append(get_nested_value(p_data, feature))
                        else:
                            vector = [PAD_VALUE] * len(FEATURE_LIST)

                        seq.append(vector)


                    if valid_player and len(seq) == MAX_T:
                        X_list.append(seq)
                        y_list.append(label)

            except json.JSONDecodeError:
                continue

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print("\n" + "=" * 40)
    print("      PROCESSING COMPLETE      ")
    print("=" * 40)
    print(f"Total Samples: {len(y)}")
    print(f"X Shape: {X.shape} -> (Samples, {MAX_T}, 47)")
    print(f"y Shape: {y.shape}")

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)
    print("[✓] Saved files with 47 features.")

if __name__ == "__main__":
    process_47_features()
