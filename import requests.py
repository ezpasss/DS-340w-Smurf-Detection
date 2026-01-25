import time
import random
import json
import os
import signal
import sys
from riotwatcher import LolWatcher, ApiError

# --- CONFIGURATION ---
API_KEY = 'RGAPI-6be47dac-4559-4c68-b516-1cbae331c5fa'  # <--- PASTE KEY
REGION = 'na1'
OUTPUT_FILE = 'Full_Data.jsonl'

# How deep to go per player?
MATCHES_PER_PLAYER = 3  

# Ranks to cycle through (Justice League uses all of them)
TIERS = ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND', 'MASTER', 'GRANDMASTER', 'CHALLENGER']
DIVISIONS = ['I', 'II', 'III', 'IV']

watcher = LolWatcher(API_KEY)
seen_match_ids = set()
running = True

# --- 1. GRACEFUL SHUTDOWN HANDLER ---
def signal_handler(sig, frame):
    global running
    print("\n\n[!] WAKE UP SIGNAL RECEIVED (Ctrl+C). Finishing current item and saving...")
    running = False

# Register the "Stop" signal
signal.signal(signal.SIGINT, signal_handler)

# --- 2. LOAD PREVIOUS PROGRESS ---
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                seen_match_ids.add(data['match_id'])
            except:
                pass
    print(f"Loaded {len(seen_match_ids)} existing matches from database.")

# --- 3. HELPER FUNCTIONS ---
def get_puuid_robust(summoner_id):
    try:
        summ = watcher.summoner.by_id(REGION, summoner_id)
        return summ['puuid']
    except ApiError:
        return None

def get_timeline_25min(match_id):
    try:
        timeline = watcher.match.timeline_by_match(REGION, match_id)
        frames = timeline['info']['frames']
        if len(frames) < 26: return None # Must be 25 mins+
        
        extracted = []
        for i in range(26):
            frame = frames[i]
            
            # INSTEAD OF PICKING 4 STATS, WE SAVE THE WHOLE THING
            # We specifically want the 'participantFrames' dictionary
            # which contains keys "1" through "10" with ALL the data.
            extracted.append({
                'minute': i,
                'participantFrames': frame['participantFrames'] 
            })
            
        return extracted
    except: return None

# --- 4. THE INFINITE LOOP ---
print("--- STARTING OVERNIGHT HARVEST (Press Ctrl+C to Stop) ---")

while running:
    # A. Pick a Random Tier
    tier = random.choice(TIERS)
    division = random.choice(DIVISIONS) if tier not in ['MASTER', 'GRANDMASTER', 'CHALLENGER'] else 'I'
    
    print(f"\n>>> Scavenging {tier} {division}...")
    
    try:
        # B. Get Random Players
        if tier in ['MASTER', 'GRANDMASTER', 'CHALLENGER']:
            # Apex tiers
            if tier == 'MASTER': func = watcher.league.masters_by_queue
            elif tier == 'GRANDMASTER': func = watcher.league.grandmaster_by_queue
            else: func = watcher.league.challenger_by_queue
            entries = func(REGION, 'RANKED_SOLO_5x5')['entries']
        else:
            # Standard tiers (Random Page 1-5 to avoid top-of-ladder bias)
            page = random.randint(1, 5)
            entries = watcher.league.entries(REGION, 'RANKED_SOLO_5x5', tier, division, page=page)
        
        # Shuffle and Pick 5 Players from this rank
        if not entries: continue
        random.shuffle(entries)
        batch = entries[:5] 

        # C. Process This Batch
        for entry in batch:
            if not running: break # Stop requested?
            
            # Resolve PUUID
            puuid = entry.get('puuid')
            if not puuid:
                puuid = get_puuid_robust(entry['summonerId'])
            
            if not puuid: continue

            # Get Matches
            try:
                matches = watcher.match.matchlist_by_puuid(REGION, puuid, queue=420, count=MATCHES_PER_PLAYER)
            except ApiError: continue

            print(f"  > Processing {entry.get('summonerName', 'Player')} ({len(matches)} matches)...")

            # D. Download Timelines
            for m_id in matches:
                if not running: break 
                if m_id in seen_match_ids: continue

                timeline = get_timeline_25min(m_id)
                
                if timeline:
                    # E. Save to Disk IMMEDIATELY
                    record = {
                        'match_id': m_id,
                        'tier': tier,
                        'division': division,
                        'timeline': timeline
                    }
                    
                    with open(OUTPUT_FILE, 'a') as f:
                        f.write(json.dumps(record) + '\n')
                    
                    seen_match_ids.add(m_id)
                    print(f"    + Secured Match {m_id}")
                
                time.sleep(1.2) # Rate limit nap
                
    except ApiError as e:
        print(f"  ! API Error (probably rate limit): {e}")
        time.sleep(10)
    except Exception as e:
        print(f"  ! Error: {e}")

print("\n[✓] Script stopped safely. All data saved.")