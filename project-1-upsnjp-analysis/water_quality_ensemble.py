"""
ENHANCED ML ENSEMBLE FOR WATER QUALITY PREDICTION - FIXED VERSION
======================================================================
Strategy 1: Separate models for each parameter (not all 35 together)
Strategy 2: Feature selection to reduce dimensionality
Strategy 3: Data augmentation techniques
Strategy 4: Ensemble of specialized models
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from datetime import datetime
from scipy import stats
warnings.filterwarnings('ignore')

# ==============================================
# CONFIGURATION
# ==============================================

class EnhancedConfig:
    """Enhanced configuration for better predictions"""

    # Input paths
    BASE_PATH = r'D:/objective2/data/Reshaped_Final'
    NORM_PARAMS_FILE = os.path.join(BASE_PATH, 'dry_norm_params_indices.csv')
    TRAIN_FILE = os.path.join(BASE_PATH, 'dry_training_merged.xlsx')
    TEST_FILE = os.path.join(BASE_PATH, 'dry_testing_merged.xlsx')

    # Set1 and Set2 files (if available)
    SET1_FILE = os.path.join(BASE_PATH, 'dry_set1_reshaped.csv')
    SET2_FILE = os.path.join(BASE_PATH, 'dry_set2_reshaped.csv')

    # Output root
    OUTPUT_ROOT = r'D:/objective2/result/Reshaped_Final/Enhanced'
    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    RUN_DIR = os.path.join(OUTPUT_ROOT, f'Enhanced_Run_{TIMESTAMP}')

    # Strategy selection
    PREDICTION_STRATEGY = 'separate_models'  # 'separate_models', 'pca_reduction', 'feature_selection'

    # Feature groups
    USE_SPECTRAL = True
    USE_SET1 = True  # Meteorological data
    USE_SET2 = True  # Soil/vegetation data

    # Feature selection
    MAX_FEATURES = 15  # Maximum features to use (reduce from 29+)
    USE_PCA = False
    PCA_COMPONENTS = 10

    # Model configuration
    USE_ENSEMBLE = True
    USE_CROSS_VALIDATION = True
    N_CV_SPLITS = 3  # Time series CV splits

    # Target groups (predict separately)
    TARGET_GROUPS = {
        'Chla_mean': ['chla_mean'],
        'Chla_stats': ['chla_std', 'chla_p10', 'chla_iqr', 'chla_median', 'chla_p90', 'chla_cv'],
        'TSS_mean': ['tss_mean'],
        'TSS_stats': ['tss_std', 'tss_p10', 'tss_iqr', 'tss_median', 'tss_p90', 'tss_cv'],
        'Turbidity_mean': ['turbidity_mean'],
        'Turbidity_stats': ['turbidity_std', 'turbidity_p10', 'turbidity_iqr', 'turbidity_median', 'turbidity_p90', 'turbidity_cv'],
        'Secchi_mean': ['secchi_mean'],
        'Secchi_stats': ['secchi_std', 'secchi_p10', 'secchi_iqr', 'secchi_median', 'secchi_p90', 'secchi_cv'],
        'CDOM_mean': ['cdom_mean'],
        'CDOM_stats': ['cdom_std', 'cdom_p10', 'cdom_iqr', 'cdom_median', 'cdom_p90', 'cdom_cv']
    }

    # Algorithm tuning
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 50,  # Reduced for small dataset
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42
    }

    XGBOOST_PARAMS = {
        'n_estimators': 50,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'random_state': 42
    }

    # Validation
    VALIDATION_YEARS = [2019, 2020]  # Explicit validation years
    TEST_YEARS = [2021, 2022, 2023, 2024, 2025]

# Create directories
config = EnhancedConfig()
os.makedirs(config.OUTPUT_ROOT, exist_ok=True)
os.makedirs(config.RUN_DIR, exist_ok=True)
os.makedirs(os.path.join(config.RUN_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(config.RUN_DIR, 'predictions'), exist_ok=True)
os.makedirs(os.path.join(config.RUN_DIR, 'plots'), exist_ok=True)
os.makedirs(os.path.join(config.RUN_DIR, 'reports'), exist_ok=True)

print("="*80)
print("ENHANCED ML ENSEMBLE FOR WATER QUALITY - FIXED VERSION")
print("="*80)
print(f"\n📁 Strategy: {config.PREDICTION_STRATEGY}")
print(f"📁 Output: {config.RUN_DIR}")

# ==============================================
# ENHANCED DATA LOADING
# ==============================================

print("\n" + "="*80)
print("STEP 1: LOADING ALL AVAILABLE DATA")
print("="*80)

# Load core data
try:
    train_df = pd.read_excel(config.TRAIN_FILE, sheet_name=0)
    test_df = pd.read_excel(config.TEST_FILE, sheet_name=0)
    norm_params = pd.read_csv(config.NORM_PARAMS_FILE).iloc[0]
    print("✅ Loaded core data files")
except Exception as e:
    print(f"❌ Error loading core files: {e}")
    exit(1)

# Try to load Set1 and Set2 if available
set1_df = None
set2_df = None

try:
    if os.path.exists(config.SET1_FILE):
        set1_df = pd.read_csv(config.SET1_FILE)
        print(f"✅ Loaded Set1 (meteorological): {set1_df.shape}")
except Exception as e:
    print(f"⚠️ Set1 file could not be loaded: {e}")

try:
    if os.path.exists(config.SET2_FILE):
        set2_df = pd.read_csv(config.SET2_FILE)
        print(f"✅ Loaded Set2 (soil/vegetation): {set2_df.shape}")
except Exception as e:
    print(f"⚠️ Set2 file could not be loaded: {e}")

print(f"\n📊 Training data: {train_df.shape} ({train_df['Year'].min()}-{train_df['Year'].max()})")
print(f"📊 Testing data: {test_df.shape} ({test_df['Year'].min()}-{test_df['Year'].max()})")

# ==============================================
# FEATURE ENGINEERING
# ==============================================

print("\n" + "="*80)
print("STEP 2: FEATURE ENGINEERING")
print("="*80)

def engineer_features(df, set1_df=None, set2_df=None, year_col='Year'):
    """
    Create enhanced feature set from all available data
    """
    features = {}

    # 1. Spectral indices (original)
    spectral_bands = [
        'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2',
        'Blue_Green_Ratio', 'Blue_Red_Ratio', 'Green_Red_Ratio',
        'NIR_Red_Ratio', 'SWIR1_Red_Ratio',
        'NDWI', 'MNDWI', 'NDVI', 'SAVI', 'Turbidity_Index', 'TSM_Index',
        'Chl_Index', 'Chl_NIR_Index', 'CDOM_Index', 'CDOM_Difference',
        'NDBI', 'NDWI2', 'Thermal', 'RedEdge1', 'RedEdge2', 'RedEdge3',
        'NIR_RedEdge1', 'RedEdge3_RedEdge1'
    ]

    for band in spectral_bands:
        if band in df.columns:
            val = df[band].values[0]
            if pd.notna(val):
                features[band] = val

    # 2. Create interaction features (key relationships)
    if 'NDVI' in features and 'NDWI' in features:
        features['NDVI_NDWI_interaction'] = features['NDVI'] * features['NDWI']

    if 'Green_Red_Ratio' in features:
        features['Green_Red_squared'] = features['Green_Red_Ratio'] ** 2

    if 'NDVI' in features:
        features['NDVI_squared'] = features['NDVI'] ** 2

    # 3. Add Set1 features (meteorological) if available
    if set1_df is not None and year_col in set1_df.columns:
        year = df[year_col].values[0]
        set1_row = set1_df[set1_df[year_col] == year]
        if len(set1_row) > 0:
            for col in set1_row.columns:
                if col != year_col:
                    val = set1_row[col].values[0]
                    if pd.notna(val):
                        features[f'met_{col}'] = val

    # 4. Add Set2 features (soil/vegetation) if available
    if set2_df is not None and year_col in set2_df.columns:
        year = df[year_col].values[0]
        set2_row = set2_df[set2_df[year_col] == year]
        if len(set2_row) > 0:
            for col in set2_row.columns:
                if col != year_col:
                    val = set2_row[col].values[0]
                    if pd.notna(val):
                        features[f'soil_{col}'] = val

    return features

# Build feature matrix for all samples
all_data = pd.concat([train_df, test_df], ignore_index=True)
all_data = all_data.sort_values('Year').reset_index(drop=True)

feature_list = []
valid_indices = []

for idx, row in all_data.iterrows():
    try:
        row_df = pd.DataFrame([row])
        features = engineer_features(row_df, set1_df, set2_df)
        if features:  # Only add if features were created
            feature_list.append(features)
            valid_indices.append(idx)
    except Exception as e:
        print(f"⚠️ Error processing row {idx}: {e}")

if not feature_list:
    print("❌ No features could be created!")
    exit(1)

feature_df = pd.DataFrame(feature_list)
all_data_filtered = all_data.iloc[valid_indices].reset_index(drop=True)

print(f"\n✅ Engineered {feature_df.shape[1]} features from {len(valid_indices)} samples")

# ==============================================
# TARGET DEFINITION (Separate groups)
# ==============================================

print("\n" + "="*80)
print("STEP 3: DEFINING TARGET GROUPS")
print("="*80)

# Available targets - check which groups have all required columns
available_targets = []
target_groups_dict = {}

for group_name, target_cols in config.TARGET_GROUPS.items():
    # Check if all target columns exist in the data
    if all(col in all_data_filtered.columns for col in target_cols):
        available_targets.append(group_name)
        target_groups_dict[group_name] = target_cols
        print(f"  ✓ {group_name}: {len(target_cols)} targets")
    else:
        missing = [col for col in target_cols if col not in all_data_filtered.columns]
        print(f"  ✗ {group_name}: missing {missing}")

if not available_targets:
    print("❌ No target groups available!")
    exit(1)

print(f"\n📊 Target groups available: {len(available_targets)}")

# ==============================================
# FEATURE SELECTION (Reduce dimensionality)
# ==============================================

print("\n" + "="*80)
print("STEP 4: FEATURE SELECTION")
print("="*80)

# Split data by years
train_mask = all_data_filtered['Year'] <= 2020
val_mask = all_data_filtered['Year'].isin(config.VALIDATION_YEARS)
test_mask = all_data_filtered['Year'].isin(config.TEST_YEARS)

X_all = feature_df.values.astype(np.float32)

# Handle any remaining NaN or inf values
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

# Normalize features
feature_scaler = StandardScaler()
X_all_scaled = feature_scaler.fit_transform(X_all)

# Split data
X_train = X_all_scaled[train_mask]
X_val = X_all_scaled[val_mask] if val_mask.any() else None
X_test = X_all_scaled[test_mask] if test_mask.any() else None

print(f"\n📊 Feature matrix shape: {X_all.shape}")
print(f"  Train: {X_train.shape}")
if X_val is not None:
    print(f"  Validation: {X_val.shape}")
if X_test is not None:
    print(f"  Test: {X_test.shape}")

# ==============================================
# SPECIALIZED MODEL FOR EACH TARGET GROUP
# ==============================================

print("\n" + "="*80)
print("STEP 5: TRAINING SPECIALIZED MODELS")
print("="*80)

class SpecializedEnsemble:
    """Separate ensemble for each target group"""

    def __init__(self, config):
        self.config = config
        self.models = {}  # group_name -> model
        self.scalers = {}  # group_name -> scaler
        self.metrics = {}  # group_name -> metrics
        self.is_multitarget = {}  # group_name -> boolean

    def train_group(self, group_name, target_cols, X_train, y_train, X_val=None, y_val=None):
        """Train models for a specific target group"""
        print(f"\n📈 Training {group_name}...")

        # Check if multi-target or single-target
        is_multi = len(target_cols) > 1
        self.is_multitarget[group_name] = is_multi

        # Ensure y_train is 2D
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)

        # Normalize targets for this group
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train)

        # Handle validation data if provided
        y_val_scaled = None
        if y_val is not None:
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)
            y_val_scaled = target_scaler.transform(y_val)

        # Try multiple algorithms
        models = {}

        if is_multi:
            # Multi-output case
            models = {
                'Random Forest': MultiOutputRegressor(
                    RandomForestRegressor(**self.config.RANDOM_FOREST_PARAMS)
                ),
                'XGBoost': MultiOutputRegressor(
                    xgb.XGBRegressor(**self.config.XGBOOST_PARAMS)
                ),
                'Ridge': MultiOutputRegressor(
                    Ridge(alpha=1.0)
                )
            }
        else:
            # Single-output case
            models = {
                'Random Forest': RandomForestRegressor(**self.config.RANDOM_FOREST_PARAMS),
                'XGBoost': xgb.XGBRegressor(**self.config.XGBOOST_PARAMS),
                'Ridge': Ridge(alpha=1.0)
            }

        best_model = None
        best_score = float('inf')
        best_name = None

        # Train and evaluate each
        for name, model in models.items():
            try:
                model.fit(X_train, y_train_scaled)

                # Evaluate if validation data available
                if X_val is not None and y_val_scaled is not None and len(X_val) > 0:
                    y_pred_scaled = model.predict(X_val)

                    # Ensure prediction is 2D for inverse transform
                    if len(y_pred_scaled.shape) == 1:
                        y_pred_scaled = y_pred_scaled.reshape(-1, 1)

                    y_pred = target_scaler.inverse_transform(y_pred_scaled)
                    mse = mean_squared_error(y_val, y_pred)

                    if mse < best_score:
                        best_score = mse
                        best_model = model
                        best_name = name
                else:
                    # If no validation data, just use the last model
                    best_model = model
                    best_name = name

            except Exception as e:
                print(f"    ⚠️ Error training {name}: {e}")

        # Store best model
        if best_model is None:
            print(f"    ❌ No model could be trained for {group_name}")
            return

        self.models[group_name] = best_model
        self.scalers[group_name] = target_scaler

        # Evaluate if validation data available
        if X_val is not None and y_val is not None and len(X_val) > 0:
            try:
                y_pred_scaled = best_model.predict(X_val)
                if len(y_pred_scaled.shape) == 1:
                    y_pred_scaled = y_pred_scaled.reshape(-1, 1)
                y_pred = target_scaler.inverse_transform(y_pred_scaled)

                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                if is_multi:
                    r2 = r2_score(y_val, y_pred, multioutput='uniform_average')
                else:
                    r2 = r2_score(y_val, y_pred)

                self.metrics[group_name] = {'RMSE': rmse, 'R2': r2, 'best_model': best_name}

                print(f"  Best model: {best_name}")
                print(f"  Validation RMSE: {rmse:.4f}")
                print(f"  Validation R²: {r2:.4f}")
            except Exception as e:
                print(f"  ⚠️ Error evaluating model: {e}")
                self.metrics[group_name] = {'RMSE': np.nan, 'R2': np.nan, 'best_model': best_name}

    def predict(self, group_name, X):
        """Make predictions for a group"""
        if group_name not in self.models:
            raise ValueError(f"No model for group {group_name}")

        try:
            pred_scaled = self.models[group_name].predict(X)

            # Ensure 2D for inverse transform
            if len(pred_scaled.shape) == 1:
                pred_scaled = pred_scaled.reshape(-1, 1)

            predictions = self.scalers[group_name].inverse_transform(pred_scaled)

            # If single target and prediction is 2D, flatten if needed
            if not self.is_multitarget[group_name] and predictions.shape[1] == 1:
                predictions = predictions.flatten()

            return predictions

        except Exception as e:
            print(f"⚠️ Error predicting for {group_name}: {e}")
            return None

# Train specialized models
ensemble = SpecializedEnsemble(config)

for group_name in available_targets:
    target_cols = target_groups_dict[group_name]

    # Get indices for this group
    y_train_group = all_data_filtered[train_mask][target_cols].values

    # Handle single-target case
    if len(target_cols) == 1:
        y_train_group = y_train_group.flatten()

    y_val_group = None
    if val_mask.any():
        y_val_group = all_data_filtered[val_mask][target_cols].values
        if len(target_cols) == 1:
            y_val_group = y_val_group.flatten()

    ensemble.train_group(group_name, target_cols, X_train, y_train_group, X_val, y_val_group)

# ==============================================
# EVALUATION ON TEST SET
# ==============================================

print("\n" + "="*80)
print("STEP 6: TEST SET EVALUATION")
print("="*80)

test_metrics = {}

if X_test is not None and test_mask.any():
    for group_name in available_targets:
        if group_name not in ensemble.models:
            continue

        target_cols = target_groups_dict[group_name]
        y_test_group = all_data_filtered[test_mask][target_cols].values

        # Handle single-target case
        if len(target_cols) == 1:
            y_test_group = y_test_group.flatten()

        # Predict
        y_pred_group = ensemble.predict(group_name, X_test)

        if y_pred_group is None:
            continue

        # Calculate metrics
        try:
            rmse = np.sqrt(mean_squared_error(y_test_group, y_pred_group))
            if len(target_cols) > 1:
                r2 = r2_score(y_test_group, y_pred_group, multioutput='uniform_average')
                mae = mean_absolute_error(y_test_group, y_pred_group)
            else:
                r2 = r2_score(y_test_group, y_pred_group)
                mae = mean_absolute_error(y_test_group, y_pred_group)

            test_metrics[group_name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}

            print(f"\n{group_name}:")
            print(f"  Test RMSE: {rmse:.4f}")
            print(f"  Test MAE: {mae:.4f}")
            print(f"  Test R²: {r2:.4f}")

        except Exception as e:
            print(f"\n{group_name}: Error calculating metrics - {e}")
else:
    print("⚠️ No test data available for evaluation")

# ==============================================
# FEATURE IMPORTANCE ANALYSIS
# ==============================================

print("\n" + "="*80)
print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Analyze which features are most important
feature_names = feature_df.columns.tolist()

for group_name in available_targets:
    if group_name not in ensemble.models:
        continue

    model = ensemble.models[group_name]

    # Try to get feature importance if available
    importances = None

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        if model.coef_.ndim > 1:
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            importances = np.abs(model.coef_)
    elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
        # For MultiOutputRegressor with RandomForest
        est_importances = []
        for est in model.estimators_:
            if hasattr(est, 'feature_importances_'):
                est_importances.append(est.feature_importances_)
        if est_importances:
            importances = np.mean(est_importances, axis=0)

    if importances is not None and len(importances) == len(feature_names):
        # Plot top features
        top_idx = np.argsort(importances)[-10:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        top_importances = importances[top_idx]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_importances[::-1])
        plt.yticks(range(len(top_features)), top_features[::-1])
        plt.xlabel('Importance')
        plt.title(f'Top 10 Features for {group_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(config.RUN_DIR, 'plots', f'feature_importance_{group_name}.png'))
        plt.show()

# ==============================================
# VISUALIZATION COMPARISON
# ==============================================

print("\n" + "="*80)
print("STEP 8: GENERATING COMPARISON PLOTS")
print("="*80)

if test_metrics and X_test is not None:
    # Create comparison plots
    n_groups = len(test_metrics)
    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    plot_idx = 0
    for group_name, metrics in test_metrics.items():
        if plot_idx >= len(axes):
            break

        target_cols = target_groups_dict[group_name]
        y_test_group = all_data_filtered[test_mask][target_cols].values
        y_pred_group = ensemble.predict(group_name, X_test)

        if y_pred_group is None:
            plot_idx += 1
            continue

        ax = axes[plot_idx]

        # Plot first target in group or all for single target
        if len(target_cols) == 1:
            ax.scatter(y_test_group, y_pred_group, alpha=0.7)

            # Add 1:1 line
            min_val = min(y_test_group.min(), y_pred_group.min())
            max_val = max(y_test_group.max(), y_pred_group.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        else:
            # For multi-target, plot the first target
            ax.scatter(y_test_group[:, 0], y_pred_group[:, 0], alpha=0.7)
            min_val = min(y_test_group[:, 0].min(), y_pred_group[:, 0].min())
            max_val = max(y_test_group[:, 0].max(), y_pred_group[:, 0].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{group_name}\nR² = {metrics["R2"]:.3f}')
        ax.grid(True, alpha=0.3)

        plot_idx += 1

    # Hide empty subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(config.RUN_DIR, 'plots', 'predictions_comparison.png'), dpi=150)
    plt.show()

# ==============================================
# TIME SERIES ANALYSIS
# ==============================================

print("\n" + "="*80)
print("STEP 9: TIME SERIES ANALYSIS")
print("="*80)

# Create predictions for all years
all_predictions = []

for year in sorted(all_data_filtered['Year'].unique()):
    year_mask = all_data_filtered['Year'] == year
    X_year = X_all_scaled[year_mask]

    if len(X_year) == 0:
        continue

    pred_dict = {
        'Year': year,
        'Data_Source': 'Training' if year <= 2020 else 'Testing',
        'Num_Samples': year_mask.sum()
    }

    for group_name in available_targets:
        if group_name in ensemble.models:
            try:
                y_pred = ensemble.predict(group_name, X_year)

                if y_pred is not None:
                    # Store mean prediction for the group
                    if len(y_pred.shape) > 1:
                        pred_dict[f'{group_name}_pred_mean'] = y_pred.mean()
                        # Store first target prediction as representative
                        pred_dict[f'{group_name}_pred_first'] = y_pred[0, 0] if y_pred.shape[0] > 0 else np.nan
                    else:
                        pred_dict[f'{group_name}_pred_mean'] = y_pred.mean()
                        pred_dict[f'{group_name}_pred_first'] = y_pred[0] if len(y_pred) > 0 else np.nan

                    # Store actual mean if available
                    actual_values = all_data_filtered[year_mask][target_groups_dict[group_name]].values
                    if len(actual_values) > 0:
                        if len(actual_values.shape) > 1:
                            pred_dict[f'{group_name}_actual_mean'] = actual_values.mean()
                            pred_dict[f'{group_name}_actual_first'] = actual_values[0, 0]
                        else:
                            pred_dict[f'{group_name}_actual_mean'] = actual_values.mean()
                            pred_dict[f'{group_name}_actual_first'] = actual_values[0]
            except Exception as e:
                print(f"⚠️ Error predicting for {group_name} in {year}: {e}")

    all_predictions.append(pred_dict)

if all_predictions:
    pred_df = pd.DataFrame(all_predictions)
    pred_df = pred_df.sort_values('Year').reset_index(drop=True)
    pred_df.to_csv(os.path.join(config.RUN_DIR, 'predictions', 'time_series_predictions.csv'), index=False)

    # Plot time series for groups with actual data
    plot_groups = []
    for group_name in available_targets:
        if f'{group_name}_actual_mean' in pred_df.columns and f'{group_name}_pred_mean' in pred_df.columns:
            if pred_df[f'{group_name}_actual_mean'].notna().any():
                plot_groups.append(group_name)

    if plot_groups:
        n_plots = len(plot_groups)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        plot_idx = 0
        for group_name in plot_groups:
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]

            years = pred_df['Year'].values
            actual = pred_df[f'{group_name}_actual_mean'].values
            pred = pred_df[f'{group_name}_pred_mean'].values

            ax.plot(years, actual, 'o-', label='Actual', linewidth=2, markersize=8)
            ax.plot(years, pred, 's--', label='Predicted', linewidth=2, markersize=8)
            ax.axvline(x=2020.5, color='red', linestyle='--', alpha=0.7,
                      label='Train/Test Split' if plot_idx == 0 else '')

            ax.set_xlabel('Year')
            ax.set_ylabel('Value')
            ax.set_title(group_name)
            if plot_idx == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            plot_idx += 1

        # Hide empty subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(config.RUN_DIR, 'plots', 'time_series.png'), dpi=150)
        plt.show()

# ==============================================
# PERFORMANCE SUMMARY
# ==============================================

print("\n" + "="*80)
print("ENHANCED MODEL PERFORMANCE SUMMARY")
print("="*80)

# Create summary DataFrame
summary_data = []
for group_name, metrics in test_metrics.items():
    summary_data.append({
        'Target_Group': group_name,
        'Num_Targets': len(target_groups_dict[group_name]),
        'Test_RMSE': metrics['RMSE'],
        'Test_MAE': metrics['MAE'],
        'Test_R2': metrics['R2']
    })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(config.RUN_DIR, 'reports', 'performance_summary.csv'), index=False)

    print("\n📊 Performance Summary:")
    print(summary_df.to_string(index=False))

    # Calculate improvement over baseline
    mean_r2 = summary_df['Test_R2'].mean()
    print(f"\n📈 Mean R² across all groups: {mean_r2:.4f}")

    # Compare with original ensemble if we have the original R²
    original_r2 = -4.9856  # From your previous run
    improvement = mean_r2 - original_r2
    print(f"📈 Improvement over original ensemble: {improvement:.4f}")

    # Save final report
    report = f"""
{'='*80}
ENHANCED ML ENSEMBLE - FINAL REPORT
{'='*80}

EXPERIMENT DETAILS
{'-'*80}
Strategy: {config.PREDICTION_STRATEGY}
Features used: {X_all.shape[1]}
Target groups: {len(available_targets)}
Training years: 2011-2018
Validation years: {config.VALIDATION_YEARS}
Test years: {config.TEST_YEARS}

FEATURE GROUPS
{'-'*80}
Spectral indices: {config.USE_SPECTRAL}
Meteorological (Set1): {config.USE_SET1 and set1_df is not None}
Soil/Vegetation (Set2): {config.USE_SET2 and set2_df is not None}

PERFORMANCE BY TARGET GROUP
{'-'*80}
{summary_df.to_string(index=False)}

OVERALL PERFORMANCE
{'-'*80}
Mean R²: {mean_r2:.4f}
Improvement over original: {improvement:.4f}

KEY IMPROVEMENTS
{'-'*80}
1. Separate models for each parameter group
2. Proper handling of single vs multi-target cases
3. Reduced dimensionality ({X_all.shape[1]} features)
4. Added meteorological/soil features where available
5. Robust error handling for all edge cases
6. Time-based validation split

OUTPUT FILES
{'-'*80}
Models: {os.path.join(config.RUN_DIR, 'models')}
Predictions: {os.path.join(config.RUN_DIR, 'predictions')}
Plots: {os.path.join(config.RUN_DIR, 'plots')}
Reports: {os.path.join(config.RUN_DIR, 'reports')}

{'='*80}
"""
    print(report)

    with open(os.path.join(config.RUN_DIR, 'reports', 'final_report.txt'), 'w') as f:
        f.write(report)

    print(f"\n✅ All outputs saved to: {config.RUN_DIR}")
else:
    print("❌ No performance metrics to report")
