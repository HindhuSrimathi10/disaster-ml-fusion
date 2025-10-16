# File Path: Disaster_ML_Fusion/src/ml_models.py (FINAL CORRECTED VERSION - Saves Preprocessor)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import os
import pickle

# --- Define Paths ---
INPUT_PATH = "../data/Disaster2021_Scaled.csv" 
RESULTS_PATH = "../data/model_performance_summary_historical.csv"
MODEL_DIR = '../models'

def load_and_encode_data():
    """Loads the scaled data and performs One-Hot Encoding."""
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: Scaled data not found at {INPUT_PATH}. Please ensure you ran your EDA notebook/script first.")
        return None, None # <-- FIX 1: Return None for preprocessor too
    
    print("--- 1. Performing Final Feature Encoding (One-Hot) ---")

    # Define features based on the scaled data
    categorical_features = ['Disaster_Subgroup', 'Disaster_Type', 'ISO']
    numerical_features = [
        'Latitude', 'Longitude', 'Dis_Mag_Value', 'CPI', 
        'Total_Deaths', 'Total_Affected', 
    ]
    
    # Features to process (no target here)
    features_to_process = categorical_features + numerical_features
    
    # Preprocessor for One-Hot Encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'  # Keeps numerical features
    )
    
    # Fit and transform the features
    X_processed = preprocessor.fit_transform(df[features_to_process])
    
    # Get final feature names (includes 'remainder__...' prefix)
    feature_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X_processed, columns=feature_names)

    # --- FIX: RENAME COLUMNS TO REMOVE 'remainder__' PREFIX ---
    remainder_cols = [col for col in X_df.columns if col.startswith('remainder__')]
    rename_mapping = {col: col.replace('remainder__', '') for col in remainder_cols}
    X_df.rename(columns=rename_mapping, inplace=True)
    
    # Add the target variable (Severity_Level) back for easy split
    X_df['Severity_Level'] = df['Severity_Level']
    
    print(f"Encoding complete. Total final features: {len(X_df.columns) - 1}")
    return X_df, preprocessor # <-- FIX 1: Now correctly returning the preprocessor
    
def setup_data(df_encoded, target_type='classification'):
    """Separates features (X) and target (y) based on the task."""
    
    y_class = df_encoded['Severity_Level'] 
    
    if target_type == 'classification':
        X = df_encoded.drop(columns=['Severity_Level']) 
        return X, y_class
        
    elif target_type == 'regression':
        # Use scaled Total_Affected as the continuous regression target
        y_reg = df_encoded['Total_Affected'] 
        # Drop both Total_Affected (target) and Severity_Level from features
        X_reg = df_encoded.drop(columns=['Total_Affected', 'Severity_Level']) 
        return X_reg, y_reg

def run_ml_pipeline():
    """Executes all requested ML models and analysis."""
    # <-- FIX 2: Capture both the DataFrame and the preprocessor
    df_encoded, preprocessor = load_and_encode_data() 
    if df_encoded is None: return

    # --- Classification Setup (Primary Task) ---
    X_class, y_class = setup_data(df_encoded, 'classification')
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.3, random_state=42)
    
    performance = {}
    
    print("\n" + "="*80)
    print("STARTING MACHINE LEARNING MODEL PIPELINE (Historical Data Only)")
    print("="*80)
    
    # =================================================================================
    # 1. REGRESSION MODELS (Total_Affected Prediction)
    # =================================================================================
    print("\n--- 1. Regression Model (Predicting Scaled Total_Affected) ---")
    X_reg, y_reg = setup_data(df_encoded, 'regression')
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

    model_reg = LinearRegression()
    model_reg.fit(X_reg_train, y_reg_train)
    y_pred_reg = model_reg.predict(X_reg_test)
    
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
    cv_scores = cross_val_score(model_reg, X_reg, y_reg, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    print(f"Linear Regression RMSE: {rmse:.4f}")
    print(f"Cross-Validation (5-fold) Mean RMSE: {cv_rmse:.4f}")
    performance['Linear Regression'] = {'Model Type': 'Regression', 'Metric': 'CV RMSE', 'Value': cv_rmse}
    
    # =================================================================================
    # 2. & 5. CLASSIFICATION & ENSEMBLE MODELS (Severity_Level Prediction)
    # =================================================================================
    print("\n--- 2. & 5. Classification and Ensemble Models ---")
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naïve Bayes': GaussianNB(),
        'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
        'Random Forest (RF)': RandomForestClassifier(n_estimators=100, random_state=42),
        'Bagging (with DT)': BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=42),
        'Boosting (AdaBoost)': AdaBoostClassifier(n_estimators=50, random_state=42)
    }

    best_cv_accuracy = 0
    best_model_name = ""
    best_model_instance = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_accuracy = cross_val_score(model, X_class, y_class, cv=5, scoring='accuracy').mean()
        
        print(f"\nModel: {name}")
        print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Cross-Validation (5-fold) Mean Accuracy: {cv_accuracy:.4f}")
        
        performance[name] = {'Model Type': 'Classification', 'Metric': 'CV Accuracy', 'Value': cv_accuracy}
        
        if cv_accuracy > best_cv_accuracy:
             best_cv_accuracy = cv_accuracy
             best_model_name = name
             best_model_instance = model
             
    # --- Serialize/Save the Best Model (e.g., Decision Tree or the best one) ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    if best_model_instance:
        MODEL_PATH = os.path.join(MODEL_DIR, f'final_{best_model_name.replace(" ", "_")}.pkl') 
        with open(MODEL_PATH, 'wb') as file:
            pickle.dump(best_model_instance, file)
        print(f"\n--- Saved Best Model ({best_model_name}) to {MODEL_PATH} ---")

    # --- CRITICAL FIX: Serialize/Save the Preprocessor ---
    if preprocessor is not None:
        PREP_PATH = os.path.join(MODEL_DIR, 'preprocessor.pkl')
        with open(PREP_PATH, 'wb') as file:
            pickle.dump(preprocessor, file)
        print(f"--- Saved Preprocessor to {PREP_PATH} ---")
    
    # =================================================================================
    # 4. DIMENSIONALITY REDUCTION (PCA)
    # =================================================================================
    print("\n--- 4. Dimensionality Reduction (PCA) Impact ---")

    # Determine number of components based on 95% variance explained
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_class)
    n_components = X_pca.shape[1]
    
    # Split PCA data
    X_pca_train, X_pca_test, _, _ = train_test_split(X_pca, y_class, test_size=0.3, random_state=42)

    # Train Random Forest on reduced data
    model_pca = RandomForestClassifier(n_estimators=100, random_state=42)
    model_pca.fit(X_pca_train, y_train)
    
    cv_accuracy_pca = cross_val_score(model_pca, X_pca, y_class, cv=5, scoring='accuracy').mean()
    
    print(f"PCA reduced features from {X_class.shape[1]} to {n_components} components (95% variance).")
    print(f"Random Forest (PCA) CV Accuracy: {cv_accuracy_pca:.4f}")
    
    performance['RF (PCA)'] = {'Model Type': 'Classification', 'Metric': 'CV Accuracy', 'Value': cv_accuracy_pca}

    # =================================================================================
    # 3. CLUSTERING (Unsupervised Learning)
    # =================================================================================
    print("\n--- 3. Clustering (K-means and Hierarchical) ---")

    X_cluster = X_class.copy() 
    k = 3 

    # K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_encoded['KMeans_Cluster'] = kmeans.fit_predict(X_cluster)
    
    # Hierarchical Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=k)
    df_encoded['HCluster'] = agg_clustering.fit_predict(X_cluster)
    
    # Compare Cluster Distribution with Severity Level
    print("\nK-Means Cluster Distribution vs. Severity:")
    print(pd.crosstab(df_encoded['KMeans_Cluster'], df_encoded['Severity_Level']))
    
    print("\nHierarchical Cluster Distribution vs. Severity:")
    print(pd.crosstab(df_encoded['HCluster'], df_encoded['Severity_Level']))
    
    print("\nCommentary on Quality of Clustering:")
    print("The quality is measured by how well the clusters separate the two target classes (Severity 0 vs 1).")
    print("A high-quality clustering for this context would result in one cluster being dominated by Severity=1 and others by Severity=0.")
    print("The crosstab tables show the degree of this alignment.")
    
    # =================================================================================
    # 6. NON-PARAMETRIC LOCALLY WEIGHTED REGRESSION (LOWESS/Loess)
    # =================================================================================
    print("\n--- 6. Non-parametric Regression (LOWESS) ---")

    # Target: Analyze the trend of Scaled Total_Affected vs. CPI
    X_lowess = df_encoded['CPI']
    y_lowess = df_encoded['Total_Affected']

    # Apply Loess
    lowess = sm.nonparametric.lowess(y_lowess, X_lowess, frac=0.2) 

    # Plotting the trend
    plt.figure(figsize=(10, 6))
    plt.scatter(X_lowess, y_lowess, alpha=0.3, label='Scaled Total Affected Data Points')
    plt.plot(lowess[:, 0], lowess[:, 1], color='red', linewidth=2, label='LOWESS Smoothing Trend')
    plt.title('Non-parametric Regression: Scaled Total Affected vs. CPI')
    plt.xlabel('Scaled CPI (Continuous Feature)')
    plt.ylabel('Scaled Total Affected')
    plt.legend()
    # Save the plot so it can potentially be displayed in the web UI later
    plt.savefig(os.path.join('../data', 'lowess_plot.png'))
    plt.show()
    
    # =================================================================================
    # FINAL RESULTS
    # =================================================================================
    
    df_results = pd.DataFrame.from_dict(performance, orient='index')
    print("\n" + "="*80)
    print("FINAL MODEL PERFORMANCE SUMMARY (Based on Cross-Validation)")
    print("="*80)
    print(df_results.sort_values(by='Value', ascending=False).to_string())
    
    df_results.to_csv(RESULTS_PATH)
    print(f"\nPerformance summary saved to: {RESULTS_PATH}")

if __name__ == '__main__':
    # Ensure the model directory exists before running the pipeline
    os.makedirs(MODEL_DIR, exist_ok=True) 
    run_ml_pipeline()