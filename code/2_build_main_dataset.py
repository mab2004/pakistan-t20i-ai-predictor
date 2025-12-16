import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_T20I = 'data/T20I.csv'
INPUT_PLAYER_STATS = 'data/player_stats.csv'
INPUT_VENUE_STATS = 'data/venue_stats.csv'
OUTPUT_FILE = 'data/final_advanced_data.csv'

# Constants for Differential Features
GLOBAL_PAR_SCORE = 160
GLOBAL_STRIKE_RATE_BASELINE = 130

def build_advanced_dataset():
    print("--- Starting Phase 2: Building Advanced Dataset (With Rolling Stats) ---")

    # 1. Load Data
    # ----------------------------------------------------------
    if not os.path.exists(INPUT_T20I):
        print(f"Error: {INPUT_T20I} not found.")
        return

    print("Loading datasets...")
    # Low memory=False handles mixed types warning, or specify dtypes if known
    df = pd.read_csv(INPUT_T20I, low_memory=False)
    psl_player_stats = pd.read_csv(INPUT_PLAYER_STATS)
    psl_venue_stats = pd.read_csv(INPUT_VENUE_STATS)

    # 2. Preprocessing
    # ----------------------------------------------------------
    column_mapping = {
        'batter': 'striker',
        'batsman_runs': 'runs_off_bat',
        'extra_runs': 'extras',
        'inning': 'innings',
        'match_id': 'match_id',
        'start_date': 'date'
    }
    df.rename(columns=column_mapping, inplace=True)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    if 'total_runs' not in df.columns:
        df['total_runs'] = df['runs_off_bat'] + df['extras']

    # 3. Global Venue Baselines (Tier 2 & 3)
    # ----------------------------------------------------------
    print("Calculating Global Venue Baselines...")
    innings_1 = df[df['innings'] == 1]
    match_venue_scores = innings_1.groupby(['match_id', 'venue'])['total_runs'].sum().reset_index()
    t20i_venue_avgs = match_venue_scores.groupby('venue')['total_runs'].mean().to_dict()
    global_t20i_avg = match_venue_scores['total_runs'].mean()

    # 4. Filter for Pakistan Matches
    # ----------------------------------------------------------
    print("Filtering for Pakistan matches...")
    pak_mask = (df['batting_team'] == 'Pakistan') | (df['bowling_team'] == 'Pakistan')
    df_pak = df[pak_mask].copy()

    # 5. CALCULATE ROLLING STATS (The "Secret Sauce")
    # ----------------------------------------------------------
    print("Calculating Recent Form (Last 5 Matches)...")

    matches_meta = []
    unique_matches = df_pak['match_id'].unique()

    for mid in unique_matches:
        m_data = df_pak[df_pak['match_id'] == mid]
        teams = m_data['batting_team'].unique()

        # Identify Opponent (Robust Fix)
        if 'Pakistan' not in teams:
            continue

        opponent_list = [t for t in teams if t != 'Pakistan']
        if not opponent_list:
            # Skip if no opponent found (e.g. data error or abandoned match with only 1 team listed)
            continue
        opponent = opponent_list[0]

        # Calculate Scores
        scores = m_data.groupby('batting_team')['total_runs'].sum()
        pak_score = scores.get('Pakistan', 0)
        opp_score = scores.get(opponent, 0)

        # Determine Win
        won = 1 if pak_score > opp_score else 0

        matches_meta.append({
            'match_id': mid,
            'date': m_data['date'].iloc[0],
            'pak_raw_score': pak_score,
            'pak_won': won
        })

    # Create Meta DataFrame
    df_meta = pd.DataFrame(matches_meta)
    df_meta.sort_values('date', inplace=True)

    # Rolling Calculations
    df_meta['pak_recent_form_batting'] = df_meta['pak_raw_score'].shift(1).rolling(window=5, min_periods=1).mean()
    df_meta['pak_recent_form_win_rate'] = df_meta['pak_won'].shift(1).rolling(window=5, min_periods=1).mean()

    # Fill NaNs
    df_meta['pak_recent_form_batting'].fillna(140, inplace=True)
    df_meta['pak_recent_form_win_rate'].fillna(0.5, inplace=True)

    # Map back
    form_lookup = df_meta.set_index('match_id')[['pak_recent_form_batting', 'pak_recent_form_win_rate']].to_dict('index')

    # 6. Main Aggregation Loop (Feature Enrichment)
    # ----------------------------------------------------------
    print("Aggregating with Hybrid Features...")
    dataset_rows = []

    for mid in unique_matches:
        match_data = df_pak[df_pak['match_id'] == mid]

        # Basics
        # Check if date exists, otherwise skip
        if match_data['date'].isnull().all():
             continue
        date = match_data['date'].iloc[0]
        venue = match_data['venue'].iloc[0]
        teams = match_data['batting_team'].unique()

        if 'Pakistan' not in teams: continue

        opponent_list = [t for t in teams if t != 'Pakistan']
        if not opponent_list: continue
        opponent = opponent_list[0]

        # Targets
        scores = match_data.groupby('batting_team')['total_runs'].sum()
        pak_score = scores.get('Pakistan', 0)
        opp_score = scores.get(opponent, 0)
        pakistan_won = 1 if pak_score > opp_score else 0

        # Toss (Robust)
        if 'toss_winner' in match_data.columns and 'toss_decision' in match_data.columns:
            t_winner = match_data['toss_winner'].iloc[0]
            t_decision = match_data['toss_decision'].iloc[0]
            toss_winner_is_pakistan = 1 if t_winner == 'Pakistan' else 0
            toss_bat = 1 if (not pd.isna(t_decision) and t_decision == 'bat') else 0
        else:
            toss_winner_is_pakistan = 0
            toss_bat = 0

        # --- HYBRID FEATURE 1: Venue Stats ---
        psl_v = psl_venue_stats[psl_venue_stats['venue'] == venue]
        if not psl_v.empty:
            venue_avg_score = psl_v['venue_avg_1st_innings_score'].values[0]
        elif venue in t20i_venue_avgs:
            venue_avg_score = t20i_venue_avgs[venue]
        else:
            venue_avg_score = global_t20i_avg

        # --- HYBRID FEATURE 2: Player Stats ---
        pak_bat_mask = match_data['batting_team'] == 'Pakistan'
        pak_bowl_mask = match_data['bowling_team'] == 'Pakistan'
        p1 = match_data[pak_bat_mask]['striker'].unique()
        p2 = match_data[pak_bowl_mask]['bowler'].unique()

        # Handle empty arrays if Pak didn't bat or bowl in a washout
        p1 = p1 if len(p1) > 0 else np.array([])
        p2 = p2 if len(p2) > 0 else np.array([])

        pak_xi = list(set(np.concatenate([p1, p2])))

        xi_df = pd.DataFrame({'player': pak_xi})
        xi_stats = xi_df.merge(psl_player_stats, on='player', how='left')

        # Means
        team_avg_batting = xi_stats['psl_batting_avg'].mean()
        team_avg_sr = xi_stats['psl_strike_rate'].mean()

        valid_bowlers = xi_stats[xi_stats['psl_economy'] > 0]
        team_avg_economy = valid_bowlers['psl_economy'].mean() if not valid_bowlers.empty else 8.5

        if pd.isna(team_avg_batting): team_avg_batting = 20.0
        if pd.isna(team_avg_sr): team_avg_sr = 120.0

        # --- NEW FEATURES: Rolling Form & Differentials ---
        recent_form = form_lookup.get(mid, {'pak_recent_form_batting': 140, 'pak_recent_form_win_rate': 0.5})

        venue_diff_from_avg = venue_avg_score - GLOBAL_PAR_SCORE
        form_diff = team_avg_sr - GLOBAL_STRIKE_RATE_BASELINE

        # Build Row
        row = {
            'date': date,
            'venue_avg_score': venue_avg_score,
            'venue_diff_from_avg': venue_diff_from_avg,
            'team_avg_psl_batting_avg': team_avg_batting,
            'team_avg_psl_strike_rate': team_avg_sr,
            'form_diff': form_diff,
            'team_avg_psl_economy': team_avg_economy,
            'pak_recent_form_batting': recent_form['pak_recent_form_batting'],
            'pak_recent_form_win_rate': recent_form['pak_recent_form_win_rate'],
            'toss_winner_is_pakistan': toss_winner_is_pakistan,
            'toss_bat': toss_bat,
            'opponent': opponent,
            'pakistan_score': pak_score,
            'pakistan_won': pakistan_won
        }
        dataset_rows.append(row)

    # 7. Finalize
    # ----------------------------------------------------------
    final_df = pd.DataFrame(dataset_rows)
    final_df.sort_values('date', inplace=True)

    print("One-Hot Encoding Opponents...")
    final_df = pd.get_dummies(final_df, columns=['opponent'], prefix='opponent')

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("--------------------------------------------------")
    print(f"SUCCESS! Advanced Dataset built: {OUTPUT_FILE}")
    print(f"Total Matches: {len(final_df)}")
    print(f"New Features Sample:\n{final_df[['pak_recent_form_win_rate', 'venue_diff_from_avg', 'form_diff']].tail(5)}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    build_advanced_dataset()