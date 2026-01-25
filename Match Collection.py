import requests
import json
import time
import os

# --- CONFIGURATION ---
API_KEY = "RGAPI-6035c577-d04f-4ace-b1ab-83a2f5935dbf" # <--- PASTE KEY
REGION_V5 = "americas"
REGION_V4 = "na1"

# Target: 1000 UNIQUE matches per class
TARGET_COUNT = 1000 

headers = {"X-Riot-Token": API_KEY}

def resolve_puuid(summ_id):
    url = f"https://{REGION_V4}.api.riotgames.com/lol/summoner/v4/summoners/{summ_id}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json()['puuid']
    return None

def get_match_ids(puuid):
    # Fetch 10 games to increase odds of finding a unique one
    url = f"https://{REGION_V5}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {'start': 0, 'count': 10, 'queue': 420} 
    resp = requests.get(url, headers=headers, params=params)
    return resp.json() if resp.status_code == 200 else []

def get_timeline(match_id):
    url = f"https://{REGION_V5}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    while True:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200: return resp.json()
        elif resp.status_code == 429: 
            print("   ⚠️ Rate Limit! Sleeping 10s...")
            time.sleep(10)
            continue
        elif resp.status_code == 404: return None
        else: return None

if __name__ == "__main__":
    if not os.path.exists("player_list.json"):
        print("❌ Error: Run Script 1 first!")
        exit()

    with open("player_list.json", "r") as f:
        players = json.load(f)

    # --- DUPLICATE PROTECTION ---
    seen_matches = set()
    counts = {"Bronze": 0, "Challenger": 0}

    # Load existing files into registry so we don't re-download them
    for label in ["Bronze", "Challenger"]:
        d = os.path.join("data", label)
        if os.path.exists(d):
            files = os.listdir(d)
            for f in files:
                seen_matches.add(f.replace(".json", ""))
            counts[label] = len(files)
    
    print(f"Resuming with {len(seen_matches)} unique matches already on disk.")

    # --- MAIN LOOP ---
    for p in players:
        label = p['label']
        
        # STOP if this class is full
        if counts[label] >= TARGET_COUNT:
            if counts["Bronze"] >= TARGET_COUNT and counts["Challenger"] >= TARGET_COUNT:
                break
            continue

        # 1. Resolve PUUID
        puuid = p.get('puuid')
        if not puuid:
            puuid = resolve_puuid(p['id'])
            time.sleep(1.2)
        
        if not puuid: continue

        # 2. Get Matches
        match_ids = get_match_ids(puuid)
        time.sleep(1.2)

        # 3. Download (With Duplicate Check)
        save_dir = os.path.join("data", label)
        os.makedirs(save_dir, exist_ok=True)

        for mid in match_ids:
            if counts[label] >= TARGET_COUNT: break
            
            # SKIPS if anyone has downloaded this match before
            if mid in seen_matches: continue

            print(f"[{counts['Bronze']}/{TARGET_COUNT} B] [{counts['Challenger']}/{TARGET_COUNT} C] -> Downloading {mid}...")
            
            timeline = get_timeline(mid)
            if timeline:
                timeline['metadata']['target_puuid'] = puuid # Tag target
                
                with open(os.path.join(save_dir, f"{mid}.json"), "w") as f:
                    json.dump(timeline, f)
                
                seen_matches.add(mid)
                counts[label] += 1
            
            time.sleep(1.2) 

    print("\n🎉 DATASET COLLECTION COMPLETE!")