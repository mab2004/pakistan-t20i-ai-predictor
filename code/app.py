import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import shap

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Pakistan T20I AI Brain",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Dark Theme CSS
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Main Background - Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f3d3e 100%);
        border-right: 1px solid #00ff88;
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    /* Header Styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem; 
        background: linear-gradient(90deg, #00ff88 0%, #00d4aa 50%, #00b4d8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center; 
        font-weight: 800;
        margin-bottom: 0;
        text-shadow: 0 0 30px rgba(0,255,136,0.3);
    }
    
    .sub-header {
        font-size: 1.3rem; 
        color: #888; 
        text-align: center;
        margin-bottom: 30px;
        letter-spacing: 2px;
    }
    
    /* Metric Cards - Glassmorphism */
    .metric-container {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0,255,136,0.2);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        border-color: #00ff88;
        box-shadow: 0 10px 40px rgba(0,255,136,0.2);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00ff88, #00d4aa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }
    
    .metric-sub {
        font-size: 0.85rem;
        color: #666;
    }
    
    /* Prediction Banners */
    .prediction-banner {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 20px 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(0,255,136,0.4); }
        50% { box-shadow: 0 0 40px rgba(0,255,136,0.6); }
    }
    
    .win-banner { 
        background: linear-gradient(135deg, #00ff88 0%, #00b894 50%, #00a085 100%);
    }
    
    .loss-banner { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 50%, #d63031 100%);
        animation-name: pulse-red;
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 20px rgba(255,107,107,0.4); }
        50% { box-shadow: 0 0 40px rgba(255,107,107,0.6); }
    }
    
    .prediction-banner h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-banner h2 {
        font-size: 1.5rem;
        font-weight: 500;
        margin: 10px 0 0 0;
        opacity: 0.9;
    }
    
    /* Info Cards */
    .info-card {
        background: rgba(0,255,136,0.1);
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .info-card h4 {
        color: #00ff88;
        margin: 0 0 10px 0;
    }
    
    .info-card p {
        color: #ccc;
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Player Cards */
    .player-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .player-name {
        color: #fff;
        font-weight: 600;
    }
    
    .player-stat {
        color: #00ff88;
        font-weight: 700;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #00ff88 0%, #00d4aa 100%);
        color: #000000 !important;
        font-weight: 800;
        border: none;
        padding: 18px 50px;
        border-radius: 30px;
        font-size: 1.3rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: none;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(0,255,136,0.4);
        color: #000000 !important;
    }
    
    .stButton > button:focus {
        color: #000000 !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #888;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00ff88 !important;
    }
    
    /* Multiselect */
    .stMultiSelect {
        background: rgba(255,255,255,0.05);
    }
    
    /* Hide Streamlit branding and deploy button */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header[data-testid="stHeader"] {background: transparent;}
    
    /* Section Headers */
    .section-header {
        color: #00ff88;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(0,255,136,0.3);
    }
    
    /* Fix text visibility for metrics, labels, and inputs */
    .stMetric label, .stMetric [data-testid="stMetricLabel"] {
        color: #aaa !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #00ff88 !important;
    }
    
    /* Selectbox and input styling */
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #e0e0e0 !important;
    }
    
    .stSelectbox > div > div {
        background-color: #1a2a3a !important;
        border: 1px solid #00ff88 !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #1a2a3a !important;
        color: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="select"] span {
        color: #ffffff !important;
    }
    
    /* Dropdown menu styling */
    [data-baseweb="menu"] {
        background-color: #1a2a3a !important;
    }
    
    [data-baseweb="menu"] li {
        color: #ffffff !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background-color: #00ff88 !important;
        color: #000000 !important;
    }
    
    /* Main content text colors */
    p, span, label {
        color: #e0e0e0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #fff !important;
    }
    
    /* Warning and info boxes */
    .stAlert {
        background-color: rgba(255,255,255,0.05) !important;
        color: #fff !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Constants
GLOBAL_PAR_SCORE = 160.0
GLOBAL_STRIKE_RATE_BASELINE = 130.0

# Pakistani Players Filter
PAK_PLAYERS_FILTER = [
    'Babar Azam', 'Mohammad Rizwan', 'Shaheen Shah Afridi', 'Shadab Khan', 
    'Fakhar Zaman', 'Haris Rauf', 'Naseem Shah', 'Iftikhar Ahmed', 
    'Imad Wasim', 'Mohammad Nawaz', 'Saim Ayub', 'Azam Khan', 
    'Usman Khan', 'Abrar Ahmed', 'Abbas Afridi', 'Mohammad Amir', 
    'Hasan Ali', 'Faheem Ashraf', 'Asif Ali', 'Haider Ali', 
    'Khushdil Shah', 'Shan Masood', 'Mohammad Haris', 'Sarfaraz Ahmed',
    'Shoaib Malik', 'Wahab Riaz', 'Sohail Tanvir', 'Umar Akmal',
    'Ahmed Shehzad', 'Kamran Akmal', 'Mohammad Hafeez', 'Shahid Afridi',
    'Rumman Raees', 'Usman Qadir', 'Shahnawaz Dahani', 'Mohammad Hasnain',
    'Arshad Iqbal', 'Zaman Khan', 'Ihsanullah', 'Aamer Jamal', 'Tayyab Tahir',
    'Sahibzada Farhan', 'Haseebullah Khan', 'Irfan Khan', 'Mehran Mumtaz',
    'Mubasir Khan', 'Qasim Akram', 'Omair Yousuf', 'Aamer Yamin', 'Sharjeel Khan'
]

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_resource
def load_resources():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)
        
        models_dir = os.path.join(base_dir, 'models')
        data_dir = os.path.join(base_dir, 'data')
        
        shap_path = os.path.join(base_dir, 'shap_summary_plot_final.png')
        if not os.path.exists(shap_path):
            shap_path = os.path.join(models_dir, 'shap_summary_plot_final.png')
            if not os.path.exists(shap_path):
                shap_path = None

        return (
            joblib.load(os.path.join(models_dir, 'score_model_final.pkl')),
            joblib.load(os.path.join(models_dir, 'win_model_final.pkl')),
            joblib.load(os.path.join(models_dir, 'feature_names.pkl')),
            pd.read_csv(os.path.join(data_dir, 'player_stats.csv')),
            pd.read_csv(os.path.join(data_dir, 'venue_stats.csv')),
            pd.read_csv(os.path.join(data_dir, 'final_advanced_data.csv')),
            shap_path
        )
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None, None, None, None, None

score_model, win_model, feature_names, player_stats, venue_stats, advanced_data, shap_plot_path = load_resources()

if score_model is None:
    st.error("🚨 Critical Error: Could not load models. Please ensure the training script ran successfully.")
    st.stop()

# Get Defaults from data
latest_row = advanced_data.iloc[-1]
default_win_rate = float(latest_row['pak_recent_form_win_rate'])
default_score_form = float(latest_row['pak_recent_form_batting'])

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("## ⚙️ Match Settings")

# Opponent & Venue
opponents = sorted([c.replace('opponent_', '') for c in feature_names if c.startswith('opponent_')])
sel_opponent = st.sidebar.selectbox("🆚 Opponent", opponents, index=opponents.index('India') if 'India' in opponents else 0)

venues = sorted(venue_stats['venue'].unique().tolist())
sel_venue = st.sidebar.selectbox("🏟️ Venue", venues)

# Toss values - not shown in UI since feature doesn't work in training data
# Setting default values that won't affect prediction
toss_winner = "Pakistan"
toss_decision = "Bat"

# Team Momentum Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Team Momentum")

# Add info expander explaining momentum
with st.sidebar.expander("ℹ️ What is Team Momentum?"):
    st.markdown("""
    **Team Momentum** captures Pakistan's recent form:
    
    - **Win Rate (Last 5)**: The percentage of wins in the last 5 matches. Higher = team is confident and in form.
    
    - **Avg Score (Last 5)**: The average runs scored in recent matches. Higher = batting is clicking.
    
    These are **rolling statistics** calculated from historical data. You can adjust them to simulate "What If" scenarios:
    
    - *What if Pakistan was on a 5-match winning streak?*
    - *What if they've been struggling lately?*
    
    The model uses these to factor in psychological momentum!
    """)

form_win_rate = st.sidebar.slider("Win Rate (Last 5)", 0.0, 1.0, default_win_rate, 0.1, 
                                   help="Pakistan's win percentage in last 5 T20I matches")
form_avg_score = st.sidebar.slider("Avg Score (Last 5)", 100, 220, int(default_score_form), 5,
                                    help="Pakistan's average score in last 5 T20I matches")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.markdown('<div class="main-header">🏏 Pakistan T20I AI Brain</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">LIVE PREDICTION ENGINE • PAK vs {sel_opponent.upper()}</div>', unsafe_allow_html=True)

# Squad Selection
st.markdown('<div class="section-header">👥 Select Your Playing XI</div>', unsafe_allow_html=True)

pak_player_stats = player_stats[player_stats['player'].isin(PAK_PLAYERS_FILTER)]
available_players = sorted(pak_player_stats['player'].unique().tolist())

default_xi = ['Babar Azam', 'Mohammad Rizwan', 'Fakhar Zaman', 'Iftikhar Ahmed', 
              'Shadab Khan', 'Imad Wasim', 'Shaheen Shah Afridi', 'Haris Rauf', 
              'Naseem Shah', 'Saim Ayub', 'Azam Khan']
default_xi = [p for p in default_xi if p in available_players]
if len(default_xi) < 11: 
    default_xi = available_players[:11]

selected_squad = st.multiselect(
    "Choose 11 players from PSL data:",
    available_players,
    default=default_xi,
    max_selections=11
)

if len(selected_squad) != 11:
    st.warning(f"⚠️ Please select exactly 11 players. Currently: {len(selected_squad)}/11")
    st.stop()

# Show selected squad stats preview
squad_df = player_stats[player_stats['player'].isin(selected_squad)]
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Batters", len(squad_df[squad_df['psl_batting_avg'] > 15]))
with col2:
    st.metric("Bowlers", len(squad_df[squad_df['psl_economy'] > 0]))
with col3:
    avg_sr = squad_df['psl_strike_rate'].mean()
    st.metric("Avg Strike Rate", f"{avg_sr:.1f}" if not pd.isna(avg_sr) else "N/A")
with col4:
    bowlers_eco = squad_df[squad_df['psl_economy'] > 0]['psl_economy'].mean()
    st.metric("Avg Economy", f"{bowlers_eco:.2f}" if not pd.isna(bowlers_eco) else "N/A")

st.markdown("---")

# ==========================================
# 5. PREDICTION ENGINE
# ==========================================
if st.button("🚀 GENERATE PREDICTION", type="primary", use_container_width=True):
    
    with st.spinner("⚡ AI is analyzing match conditions..."):
        # Feature Engineering
        v_row = venue_stats[venue_stats['venue'] == sel_venue]
        v_avg = v_row['venue_avg_1st_innings_score'].values[0] if not v_row.empty else 160.0
        v_diff = v_avg - GLOBAL_PAR_SCORE
        
        sq_bat_avg = squad_df['psl_batting_avg'].mean() if not squad_df.empty else 20.0
        sq_sr = squad_df['psl_strike_rate'].mean() if not squad_df.empty else 125.0
        bowlers = squad_df[squad_df['psl_economy'] > 0]
        sq_eco = bowlers['psl_economy'].mean() if not bowlers.empty else 8.5
        
        form_diff = sq_sr - GLOBAL_STRIKE_RATE_BASELINE
        
        toss_win_val = 1 if toss_winner == "Pakistan" else 0
        toss_bat_val = 1 if toss_decision == "Bat" else 0
        
        input_dict = {col: 0 for col in feature_names}
        input_dict.update({
            'venue_avg_score': v_avg,
            'venue_diff_from_avg': v_diff,
            'team_avg_psl_batting_avg': sq_bat_avg,
            'team_avg_psl_strike_rate': sq_sr,
            'form_diff': form_diff,
            'team_avg_psl_economy': sq_eco,
            'pak_recent_form_batting': form_avg_score,
            'pak_recent_form_win_rate': form_win_rate,
            'toss_winner_is_pakistan': toss_win_val,
            'toss_bat': toss_bat_val
        })
        
        opp_key = f"opponent_{sel_opponent}"
        if opp_key in input_dict: 
            input_dict[opp_key] = 1
        
        input_df = pd.DataFrame([input_dict])
        
        # Predictions
        pred_score = score_model.predict(input_df)[0]
        win_prob = win_model.predict_proba(input_df)[0][1]
        
        # SHAP Analysis for this specific prediction
        try:
            explainer = shap.TreeExplainer(win_model)
            shap_values = explainer.shap_values(input_df)
        except:
            shap_values = None
    
    # ==========================================
    # 6. RESULTS DASHBOARD
    # ==========================================
    
    win_pct = round(win_prob * 100, 1)
    if win_pct >= 50:
        st.markdown(f"""
        <div class="prediction-banner win-banner">
            <h1>🇵🇰 PAKISTAN WINS</h1>
            <h2>Confidence: {win_pct}%</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-banner loss-banner">
            <h1>🏴 {sel_opponent.upper()} WINS</h1>
            <h2>Pakistan Chance: {win_pct}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Metrics Row
    st.markdown('<div class="section-header">📊 Match Analysis</div>', unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Predicted Score</div>
            <div class="metric-value">{int(pred_score)}</div>
            <div class="metric-sub">Par: {int(v_avg)}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Win Probability</div>
            <div class="metric-value">{win_pct}%</div>
            <div class="metric-sub">vs {sel_opponent}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Team Strike Rate</div>
            <div class="metric-value">{sq_sr:.0f}</div>
            <div class="metric-sub">PSL Average</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Bowling Economy</div>
            <div class="metric-value">{sq_eco:.1f}</div>
            <div class="metric-sub">PSL Average</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Analysis Tabs
    st.markdown('<div class="section-header">🔬 Deep Dive Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Why This Prediction?", "👥 Squad Analysis", "📈 Feature Breakdown"])
    
    with tab1:
        st.markdown("### SHAP Analysis: What's Driving This Prediction?")
        
        if shap_values is not None:
            # Create waterfall-style feature importance for this prediction
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': shap_values[0] if len(shap_values.shape) == 2 else shap_values[1][0],
                'Input Value': input_df.values[0]
            })
            
            # Filter to top features
            feature_importance['Abs_SHAP'] = feature_importance['SHAP Value'].abs()
            top_features = feature_importance.nlargest(10, 'Abs_SHAP')
            
            # Create a horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0f0f23')
            ax.set_facecolor('#0f0f23')
            
            colors = ['#00ff88' if x > 0 else '#ff6b6b' for x in top_features['SHAP Value']]
            
            y_pos = range(len(top_features))
            ax.barh(y_pos, top_features['SHAP Value'], color=colors, height=0.6)
            
            # Clean feature names for display
            clean_names = top_features['Feature'].str.replace('_', ' ').str.title()
            ax.set_yticks(y_pos)
            ax.set_yticklabels(clean_names, fontsize=10, color='white')
            ax.set_xlabel('Impact on Win Prediction', fontsize=12, color='white')
            ax.set_title('Feature Contributions (This Prediction)', fontsize=14, color='#00ff88', fontweight='bold')
            
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('#333')
            ax.spines['left'].set_color('#333')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.axvline(x=0, color='#666', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown("""
            <div class="info-card">
                <h4>📖 How to Read This Chart</h4>
                <p>
                    • <span style="color:#00ff88">Green bars</span> = Features pushing prediction toward PAKISTAN WIN<br>
                    • <span style="color:#ff6b6b">Red bars</span> = Features pushing prediction toward opponent win<br>
                    • Longer bars = Stronger influence on this specific prediction
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("SHAP analysis not available. Please ensure shap package is installed correctly.")
    
    with tab2:
        st.markdown("### Selected Squad Performance (PSL Data)")
        
        # Batting Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏏 Batting Power")
            batting_df = squad_df[squad_df['psl_batting_avg'] > 0][['player', 'psl_batting_avg', 'psl_strike_rate']].copy()
            batting_df.columns = ['Player', 'Average', 'Strike Rate']
            batting_df = batting_df.sort_values('Strike Rate', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0f0f23')
            ax.set_facecolor('#0f0f23')
            
            x = range(len(batting_df))
            ax.bar(x, batting_df['Strike Rate'], color='#00ff88', alpha=0.8, label='Strike Rate')
            ax.set_xticks(x)
            ax.set_xticklabels(batting_df['Player'], rotation=45, ha='right', fontsize=8, color='white')
            ax.set_ylabel('Strike Rate', color='white')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('#333')
            ax.spines['left'].set_color('#333')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.axhline(y=130, color='#ff6b6b', linestyle='--', alpha=0.7, label='T20 Baseline (130)')
            ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### 🎯 Bowling Attack")
            bowling_df = squad_df[squad_df['psl_economy'] > 0][['player', 'psl_economy', 'wickets']].copy()
            bowling_df.columns = ['Player', 'Economy', 'Wickets']
            bowling_df = bowling_df.sort_values('Economy', ascending=True)
            
            if not bowling_df.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor('#0f0f23')
                ax.set_facecolor('#0f0f23')
                
                x = range(len(bowling_df))
                colors = ['#00ff88' if e < 8 else '#ffbe0b' if e < 9 else '#ff6b6b' for e in bowling_df['Economy']]
                ax.bar(x, bowling_df['Economy'], color=colors, alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(bowling_df['Player'], rotation=45, ha='right', fontsize=8, color='white')
                ax.set_ylabel('Economy Rate', color='white')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('#333')
                ax.spines['left'].set_color('#333')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.axhline(y=8, color='#00ff88', linestyle='--', alpha=0.7, label='Good (<8)')
                ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No bowling data available for selected players.")
    
    with tab3:
        st.markdown("### All Features Used in Prediction")
        
        # Create a nice display of all input features
        feature_display = pd.DataFrame({
            'Feature': feature_names,
            'Value': input_df.values[0]
        })
        
        # Filter out zero opponent features
        non_zero = feature_display[
            (~feature_display['Feature'].str.startswith('opponent_')) | 
            (feature_display['Value'] != 0)
        ]
        
        # Format nicely
        non_zero['Feature'] = non_zero['Feature'].str.replace('_', ' ').str.title()
        non_zero['Value'] = non_zero['Value'].round(2)
        
        st.dataframe(
            non_zero.style.background_gradient(subset=['Value'], cmap='Greens'),
            use_container_width=True,
            hide_index=True
        )

else:
    # Landing state
    st.markdown("""
    <div class="info-card">
        <h4>🎯 How to Use</h4>
        <p>
            1. Select your <b>Playing XI</b> from Pakistani players above<br>
            2. Configure match settings in the <b>sidebar</b> (opponent, venue, toss)<br>
            3. Adjust <b>Team Momentum</b> sliders to simulate different scenarios<br>
            4. Click <b>GENERATE PREDICTION</b> to see AI analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show Team Momentum explanation
    st.markdown('<div class="section-header">📈 Understanding Team Momentum</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>Win Rate (Last 5 Games)</h4>
            <p>
                This represents Pakistan's winning percentage in their last 5 T20I matches. 
                A team on a winning streak (80-100%) typically has higher confidence 
                and better coordination compared to a struggling team (0-20%).
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>Average Score (Last 5 Games)</h4>
            <p>
                The average runs scored by Pakistan in recent matches. Higher scores 
                (180+) indicate batting is in excellent form. Lower scores (120-140) 
                suggest struggles with timing or conditions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>💡 Why Momentum Matters</h4>
        <p>
            Cricket is a mental game. Teams on hot streaks often play with more freedom 
            and less pressure. The AI model learned this pattern from historical data - 
            recent form is one of the strongest predictors of match outcomes!
        </p>
    </div>
    """, unsafe_allow_html=True)