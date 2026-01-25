import requests
import json
import time

# --- CONFIGURATION ---
API_KEY = "RGAPI-6035c577-d04f-4ace-b1ab-83a2f5935dbf"  # <--- PASTE YOUR KEY AGAIN
REGION = "na1"

headers = {"X-Riot-Token": API_KEY}

def get_low_elo(tier, division, page):
    """Fetch Low Elo players (Bronze)"""
    url = f"https://{REGION}.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}"
    params = {'page': page}
    resp = requests.get(url, headers=headers, params=params)
    return resp.json() if resp.status_code == 200 else []

def get_high_elo(tier):
    """Fetch High Elo players (Master/Grandmaster)"""
    url = f"https://{REGION}.api.riotgames.com/lol/league/v4/{tier}leagues/by-queue/RANKED_SOLO_5x5"
    resp = requests.get(url, headers=headers)
    return resp.json()['entries'] if resp.status_code == 200 else []

if __name__ == "__main__":
    targets = []
    
    # 1. SCRAPE BRONZE (Aiming for ~1200 candidates)
    print("--- 1. Scraping Bronze Candidates ---")
    for div in ["I", "II", "III"]:
        for page in range(1, 4): # Pages 1-3
            print(f"   Fetching Bronze {div} - Page {page}...")
            players = get_low_elo("BRONZE", div, page)
            
            for p in players:
                # FIX: Use .get() because Bronze players might NOT have summonerId anymore
                s_id = p.get('summonerId')
                puuid = p.get('puuid')
                
                # As long as we have ONE valid ID, we keep them
                if s_id or puuid:
                    targets.append({
                        "id": s_id, 
                        "puuid": puuid,
                        "label": "Bronze"
                    })
            time.sleep(1.2)

    # 2. SCRAPE MASTER/GM (Aiming for ~1200 candidates)
    print("\n--- 2. Scraping High Elo Candidates ---")
    for tier in ["master", "grandmaster"]:
        print(f"   Fetching {tier} list...")
        players = get_high_elo(tier)
        
        for p in players:
            # High Elo usually has summonerId but NO puuid yet
            s_id = p.get('summonerId')
            puuid = p.get('puuid')
            
            if s_id or puuid:
                targets.append({
                    "id": s_id,
                    "puuid": puuid, 
                    "label": "Challenger" 
                })
            
            if len(targets) > 2500: break # Cap total list size

    # 3. SAVE
    with open("player_list.json", "w") as f:
        json.dump(targets, f)
    
    print(f"\n✅ SUCCESS: Found {len(targets)} candidates.")
    print("-> Run '2_fetch_matches.py' next.")