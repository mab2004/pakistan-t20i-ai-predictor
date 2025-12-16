import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = 'data/final_advanced_data.csv'
MODELS_DIR = 'models'
SHAP_PLOT_FILE = 'shap_summary_plot_final.png'

def train_final_models():
    print("--- Starting Phase 3: Final Model Training (Deep Tuning) ---")

    # 1. Load Data
    # ----------------------------------------------------------
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please run '2_build_advanced_dataset.py' first.")
        return

    print(f"Loading advanced training data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 2. Define Features (X) and Targets (y)
    # ----------------------------------------------------------
    # Core Features (Including NEW Rolling & Differential Stats)
    feature_cols = [
        'venue_avg_score',
        'venue_diff_from_avg',
        'team_avg_psl_batting_avg',
        'team_avg_psl_strike_rate',
        'form_diff',
        'team_avg_psl_economy',
        'pak_recent_form_batting',
        'pak_recent_form_win_rate',
        'toss_winner_is_pakistan',
        'toss_bat'
    ]
    # Dynamic Opponent Features
    opponent_cols = [col for col in df.columns if col.startswith('opponent_')]
    feature_cols.extend(opponent_cols)

    X = df[feature_cols]
    y_score = df['pakistan_score']
    y_win = df['pakistan_won']

    print(f"Features selected: {len(feature_cols)}")
    print(f"Data Shape: {X.shape}")

    # Standard Split
    X_train, X_test, y_score_train, y_score_test, y_win_train, y_win_test = train_test_split(
        X, y_score, y_win, test_size=0.2, random_state=42
    )

    # ==========================================
    # LARGE HYPERPARAMETER GRID (Deep Tuning)
    # ==========================================
    param_grid = {
        'n_estimators': [100, 300, 500, 1000],
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6, 8],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    # 3. Final Model 1: Score Prediction (Regression)
    # ----------------------------------------------------------
    print("\n--- Training Final Score Model (XGBoost Regressor) ---")
    print("Running GridSearchCV (This may take 5-10 mins)...")

    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

    grid_reg = GridSearchCV(
        estimator=xgb_reg,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        verbose=1
    )

    grid_reg.fit(X_train, y_score_train)
    best_score_model = grid_reg.best_estimator_

    print(f"Best Regression Params: {grid_reg.best_params_}")
    print(f"Best Cross-Val MAE: {-grid_reg.best_score_:.2f}")

    # Evaluation
    y_score_pred = best_score_model.predict(X_test)
    test_mae = mean_absolute_error(y_score_test, y_score_pred)
    test_r2 = r2_score(y_score_test, y_score_pred)
    print(f"Test Set MAE: {test_mae:.2f} runs")
    print(f"Test Set R2: {test_r2:.3f}")

    # 4. Final Model 2: Win Prediction (Classification)
    # ----------------------------------------------------------
    print("\n--- Training Final Win Model (XGBoost Classifier) ---")
    print("Running GridSearchCV (This may take 5-10 mins)...")

    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1)

    grid_clf = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1
    )

    grid_clf.fit(X_train, y_win_train)
    best_win_model = grid_clf.best_estimator_

    print(f"Best Classification Params: {grid_clf.best_params_}")
    print(f"Best Cross-Val Accuracy: {grid_clf.best_score_:.2%}")

    # Evaluation
    y_win_pred = best_win_model.predict(X_test)
    test_acc = accuracy_score(y_win_test, y_win_pred)
    print(f"Test Set Accuracy: {test_acc:.2%}")
    print(classification_report(y_win_test, y_win_pred))

    # 5. SHAP Analysis (Clean & Professional)
    # ----------------------------------------------------------
    print("\n--- Generating Clean SHAP Plot ---")
    explainer = shap.TreeExplainer(best_win_model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(12, 10))
    # max_display=15 limits the chart to top 15 features to reduce clutter
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15, show=False)
    plt.title("Top 15 Drivers of Victory (XGBoost Final)")
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_FILE)
    plt.close()
    print(f"Clean SHAP plot saved to {SHAP_PLOT_FILE}")

    # 6. Save Artifacts
    # ----------------------------------------------------------
    print("\n--- Saving Final Models ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_score_model, os.path.join(MODELS_DIR, 'score_model_final.pkl'))
    joblib.dump(best_win_model, os.path.join(MODELS_DIR, 'win_model_final.pkl'))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_names.pkl'))

    print("Phase 3 Complete. High-performance models are ready in 'models/'.")

if __name__ == "__main__":
    train_final_models()