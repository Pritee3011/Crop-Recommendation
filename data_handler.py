# data_handler.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # MUST be set before pyplot is imported for web applications
import matplotlib.pyplot as plt

from scipy.stats import f, t, f_oneway, ttest_ind
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from io import BytesIO
import base64
import statsmodels.api as sm
from statsmodels.formula.api import ols


# --- ACTUAL DATA LOADING ---
try:
    # --- VERIFY THIS PATH! ---
    file_path = r"C:\Users\prite\Downloads\indiancrop_dataset (2).csv"
    df_clean = pd.read_csv(file_path)
    
    # Validation (ensure all columns exist)
    required_cols = ['CROP', 'CROP_PRICE', 'STATE', 'N_SOIL', 'P_SOIL', 'K_SOIL', 
                     'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL']
                     
    if not all(col in df_clean.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_clean.columns]
        raise ValueError(f"Missing essential columns in CSV: {missing}. Check column names.")

    df_clean = df_clean.round(2)
    print(f"Successfully loaded {len(df_clean)} rows of data from CSV.")

except FileNotFoundError:
    print(f"🚨 FATAL ERROR: Data file not found at: {file_path}")
    print("Please check the path and file name.")
    exit(1)
except ValueError as e:
    print(f"🚨 FATAL ERROR: Data validation failed. {e}")
    exit(1)


# --- HELPER FUNCTION TO GENERATE PLOT IMAGES ---
def get_plot_base64(plot_func, *args, **kwargs):
    """Executes a plotting function, saves it to memory, and returns a base64 string."""
    plt.figure(figsize=(8, 5)) 
    plot_func(*args, **kwargs)
    plt.tight_layout() # Prevents labels from being cut off
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# --- GLOBAL MODEL TRAINING ---
def train_global_model(df):
    print("Training Random Forest Model...")
    target_col = 'CROP'
    
    # Features for the model
    feature_cols = ['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL']
    X = df[feature_cols].copy()
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    model.feature_names = X.columns.tolist() 
    
    y_pred = model.predict(X_test)
    model.accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained with {model.accuracy*100:.2f}% accuracy.")
    return model

# Train the model once when the server starts
CROP_MODEL = train_global_model(df_clean)
AVAILABLE_CROPS = list(df_clean['CROP'].unique())
AVAILABLE_STATES = list(df_clean['STATE'].unique())


# --- 1. RANDOM FOREST PREDICTION FUNCTION ---
def predict_crop(input_data, model):
    """Predicts the best crop based on user input dictionary."""
    feature_names = model.feature_names
    input_list = [input_data[name] for name in feature_names]
    user_df = pd.DataFrame([input_list], columns=feature_names)

    predicted_crop = model.predict(user_df)[0]
    return predicted_crop, model.accuracy


# --- 2. ONE-WAY ANOVA FUNCTION ---
def run_one_way_anova(df, selected_crops):
    results = {}
    selected = [c.strip() for c in selected_crops if c.strip() in df['CROP'].unique()]
    if len(selected) < 2:
        return {"error": "Please select at least two valid crops for ANOVA."}, None
        
    data = df[df['CROP'].isin(selected)]
    groups = [data[data['CROP'] == c]['CROP_PRICE'].dropna() for c in selected]
    
    F_stat, p_value = f_oneway(*groups)

    conclusion = ("The differences in crop prices are statistically significant."
                  if p_value < 0.05 else
                  "The differences in crop prices are NOT statistically significant.")
    
    results['F_stat'] = f"{F_stat:.4f}"
    results['p_value'] = f"{p_value:.4f}"
    results['conclusion'] = conclusion
    results['crops'] = ", ".join(selected)

    plot_base64 = get_plot_base64(
        sns.boxplot, 
        x='CROP', y='CROP_PRICE', data=data, 
        palette=['lightgreen']
    )
    return results, plot_base64

# --- 3. T-TEST FUNCTION ---
def run_t_test(df, crop1_name, crop2_name):
    results = {}
    crop1_name, crop2_name = crop1_name.strip(), crop2_name.strip()
    
    if crop1_name not in df['CROP'].unique() or crop2_name not in df['CROP'].unique():
        return {"error": "Invalid crop names entered."}, None
        
    group1 = df[df['CROP'] == crop1_name]['CROP_PRICE'].dropna()
    group2 = df[df['CROP'] == crop2_name]['CROP_PRICE'].dropna()
    
    t_stat, p_value = ttest_ind(group1, group2, equal_var=False) 

    conclusion = (f"The difference in prices between {crop1_name} and {crop2_name} is statistically significant."
                  if p_value < 0.05 else
                  f"The difference in prices between {crop1_name} and {crop2_name} is NOT statistically significant.")
    
    results['t_stat'] = f"{t_stat:.4f}"
    results['p_value'] = f"{p_value:.4f}"
    results['mean1'] = f"{np.mean(group1):.2f}"
    results['mean2'] = f"{np.mean(group2):.2f}"
    results['crop1'] = crop1_name
    results['crop2'] = crop2_name
    results['conclusion'] = conclusion

    plot_base64 = get_plot_base64(
        sns.boxplot, 
        x='CROP', y='CROP_PRICE', data=df[df['CROP'].isin([crop1_name, crop2_name])], 
        palette=['brown']
    )
    return results, plot_base64


# --- 4. TWO-WAY ANOVA FUNCTION ---
def run_two_way_anova_state(df, selected_crops, selected_states):
    results = {}
    
    # 1. Clean and validate inputs against unique data values
    selected_crops = [c.strip() for c in selected_crops if c.strip() in df['CROP'].unique()]
    selected_states = [s.strip() for s in selected_states if s.strip() in df['STATE'].unique()]
    
    if len(selected_crops) < 2 or len(selected_states) < 2:
        return {"error": "Need at least two crops and two states for Two-Way ANOVA."}, None

    data = df[df['CROP'].isin(selected_crops) & df['STATE'].isin(selected_states)]
    
    if data.empty:
        return {"error": "No data found for selected crops and states."}, None
        
    # CRITICAL FIX: Ensure sufficient data points (replicates) for the model.
    # We must have at least 2 observations for every combination of CROP and STATE.
    # Group by both factors and check the size of each group.
    min_obs = data.groupby(['CROP', 'STATE']).size().min()
    
    if min_obs < 2:
        return {"error": f"Insufficient data. Two-Way ANOVA requires at least 2 observations for every combination (e.g., Rice in Maharashtra). Found a minimum of {min_obs} observation(s) in a group."}, None

    try:
        # 2. Run the OLS model
        # Note: If this still fails, try removing the C() wrappers: 'CROP_PRICE ~ CROP * STATE'
        model = ols('CROP_PRICE ~ C(CROP) * C(STATE)', data=data).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
    except Exception as e:
        # This catches model failures like singular matrices (perfect collinearity, often due to 0 variance)
        return {"error": f"ANOVA calculation failed. The model encountered a mathematical issue (e.g., singular matrix). Try different crop/state combinations. Error: {e}"}, None

    
    # 3. Process results
    def get_p_value(factor):
        """Safely retrieves the p-value for the main effects or interaction term."""
        key = f"C({factor})"
        # Special handling for the interaction term key in the ANOVA table index
        if factor == 'CROP:STATE':
            # The keys might appear as 'C(CROP):C(STATE)' or sometimes just 'CROP:STATE'
            if "C(CROP):C(STATE)" in aov_table.index:
                 return aov_table.loc["C(CROP):C(STATE)", 'PR(>F)']
            elif "CROP:STATE" in aov_table.index:
                 return aov_table.loc["CROP:STATE", 'PR(>F)']
                 
        # Handling for main effect keys
        if key in aov_table.index:
             return aov_table.loc[key, 'PR(>F)']
        
        return 1.0 # Default to 1.0 (not significant) if term not found

    p_crop = get_p_value('CROP')
    p_state = get_p_value('STATE')
    p_interaction = get_p_value('CROP:STATE') 

    # Generate conclusions
    results['crop_conclusion'] = "Statistically significant." if p_crop < 0.05 else "NOT statistically significant."
    results['state_conclusion'] = "Statistically significant." if p_state < 0.05 else "NOT statistically significant."
    results['interaction_conclusion'] = "Statistically significant." if p_interaction < 0.05 else "NOT statistically significant."
    
    results['p_crop'] = f"{p_crop:.4f}"
    results['p_state'] = f"{p_state:.4f}"
    results['p_interaction'] = f"{p_interaction:.4f}"

    # 4. Generate the plot (using a more standard palette for safety)
    plot_base64 = get_plot_base64(
        sns.boxplot, 
        x='CROP', y='CROP_PRICE', hue='STATE', data=data, 
        # Using a default seaborn palette that is always available
        palette='Set2'
    )
    return results, plot_base64