import json
import os
import pandas as pd
import math

# --- CONFIGURATION ---
DATA_DIR = "data"
OUTPUT_FILE = "justice_league_dataset_final.csv"
MAX_MINUTES = 15  # Analyze first 15 minutes (Laning Phase)

def extract_features(data, label):
    """
    Parses a match JSON into 15 time-series rows with Economy, Combat, and Micro stats.
    """
    rows = []
    
    # 1. VALIDATION: Skip short games
    frames = data['info']['frames']
    if len(frames) <= MAX_MINUTES:
        return [] 

    # 2. IDENTIFY TARGET PLAYER
    target_puuid = data['metadata'].get('target_puuid')
    if not target_puuid: 
        target_puuid = data['metadata']['participants'][0]

    try:
        pid_index = data['metadata']['participants'].index(target_puuid)
        pid = pid_index + 1
    except ValueError:
        return [] 

    # 3. EXTRACT ROLE (Top, Jungle, Middle, Bottom, Utility)
    participant_info = data['info']['participants'][pid_index]
    role = participant_info.get('teamPosition', 'UNKNOWN')
    if not role: role = 'UNKNOWN'

    # 4. INITIALIZE CUMULATIVE COUNTERS
    prev_pos = None
    cum_kills = 0
    cum_deaths = 0
    cum_assists = 0
    cum_wards_placed = 0
    cum_wards_killed = 0

    # 5. LOOP THROUGH MINUTES 1 TO 15
    for i in range(1, MAX_MINUTES + 1):
        frame = frames[i]
        p_frame = frame['participantFrames'][str(pid)]
        
        # --- A. EVENT PROCESSING (KDA / Vision / APM) ---
        apm_events = 0
        
        for event in frame['events']:
            event_type = event.get('type')
            actor_id = event.get('participantId')
            
            # KILLS / DEATHS / ASSISTS
            if event_type == 'CHAMPION_KILL':
                if event.get('killerId') == pid:
                    cum_kills += 1
                    apm_events += 1
                if event.get('victimId') == pid:
                    cum_deaths += 1
                if pid in event.get('assistingParticipantIds', []):
                    cum_assists += 1

            # VISION
            elif event_type == 'WARD_PLACED' and actor_id == pid:
                cum_wards_placed += 1
                apm_events += 1
            elif event_type == 'WARD_KILL' and actor_id == pid:
                cum_wards_killed += 1
                apm_events += 1
                
            # GENERAL APM
            elif actor_id == pid:
                if event_type in ['SKILL_LEVEL_UP', 'ITEM_PURCHASED', 'BUILDING_KILL', 'TURRET_PLATE_DESTROYED', 'ELITE_MONSTER_KILL']:
                    apm_events += 1

        # --- B. MOVEMENT (Micro) ---
        curr_pos = p_frame.get('position')
        dist = 0
        if prev_pos and curr_pos:
            dist = math.sqrt((curr_pos['x'] - prev_pos['x'])**2 + (curr_pos['y'] - prev_pos['y'])**2)
        prev_pos = curr_pos

        # --- C. COMBAT STATS (Damage & CC) ---
        # Riot usually stores these in 'damageStats' inside the frame
        dmg_stats = p_frame.get('damageStats', {})
        total_damage = dmg_stats.get('totalDamageDoneToChampions', 0)
        cc_score = p_frame.get('timeEnemySpentControlled', 0)

        # --- D. SAVE ROW ---
        rows.append({
            "match_id": data['metadata']['matchId'],
            "label": label,  # 0 = Bronze, 1 = Challenger
            "role": role,    # Context for the AI
            "minute": i,     # Time Step (1..15)
            
            # Economy (The Basics)
            "gold": p_frame['totalGold'],
            "cs": p_frame['minionsKilled'] + p_frame['jungleMinionsKilled'],
            "xp": p_frame['xp'],
            
            # Combat (The Smurf Indicators)
            "kills": cum_kills,
            "deaths": cum_deaths,
            "assists": cum_assists,
            "damage_dealt": total_damage, # <--- NEW
            "cc_score": cc_score,         # <--- NEW
            
            # Micro/Macro (The Behavior)
            "apm": apm_events,
            "distance_moved": int(dist),
            "wards_placed": cum_wards_placed,
            "wards_killed": cum_wards_killed
        })
        
    return rows

if __name__ == "__main__":
    all_data = []
    print(f"--- Processing Dataset (Features: Damage, CC, Role, KDA) ---")

    for label_name in ["Bronze", "Challenger"]:
        folder_path = os.path.join(DATA_DIR, label_name)
        numeric_label = 1 if label_name == "Challenger" else 0
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Folder '{folder_path}' not found.")
            continue
            
        print(f"Reading {label_name} matches...")
        
        files = os.listdir(folder_path)
        count = 0
        
        for filename in files:
            if not filename.endswith(".json"): continue
            
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    match_data = json.load(f)
                    
                new_rows = extract_features(match_data, numeric_label)
                if new_rows:
                    all_data.extend(new_rows)
                    count += 1
            except: pass

        print(f"   -> Processed {count} matches.")

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ SUCCESS! Final dataset saved to: {OUTPUT_FILE}")
        print(f"   Total Rows: {len(df)}")
        print(f"   Unique Matches: {df['match_id'].nunique()}")
        print(f"   Columns: {list(df.columns)}")
    else:
        print("\n❌ Error: No data found. Make sure Script 2 downloaded files!")