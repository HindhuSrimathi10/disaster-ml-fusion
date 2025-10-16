# File Path: Disaster_ML_Fusion/src/preprocessing.py

import pandas as pd
import numpy as np
import os
from datetime import date
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration for visualization
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150 

def preprocess_and_clean_data(file_path):
    """
    Loads, cleans, transforms, and scales the historical disaster data.
    
    Returns:
        pd.DataFrame: The cleaned and scaled dataframe.
    """
    try:
        # Use the explicit file path passed to the function
        df = pd.read_csv("C:\\Users\\indhu\\Disaster_ML_Fusion - Copy\\data\\Disaster2021.csv")
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}. Please check your path.")
        return pd.DataFrame()

    # --- 1. Filtering and Initial Cleaning (Remains the same) ---
    df_cleaned = df[df['Year'] >= 2010].copy()
    
    high_missing_cols = [
        'Recon_Costs', 'Aid_Contribution', 'OFDA_Response', 'Appeal', 'Local_Time',
        'Associated_Dis2', 'Insured_Damages', 'River_Basin', 'Disaster_Subsubtype',
        'No_Homeless', 'Declaration', 'Event_Name', 'Associated_Dis', 'Total_Damages',
        'Origin', 'End_Year', 'End_Month', 'End_Day', 'Seq', 'Year'
    ]
    df_cleaned.drop(columns=high_missing_cols, inplace=True, errors='ignore')

    # --- 2. Temporal & Target Imputation/Engineering (Remains the same) ---
    df_cleaned['Start_Month'] = df_cleaned['Start_Month'].fillna(1).astype(int)
    df_cleaned['Start_Day'] = df_cleaned['Start_Day'].fillna(1).astype(int)
    
    def create_date(row):
        try:
            return date(int(row['Start_Year']), row['Start_Month'], row['Start_Day'])
        except ValueError:
            return date(int(row['Start_Year']), row['Start_Month'], 1)
            
    df_cleaned['Start_Date'] = pd.to_datetime(df_cleaned.apply(create_date, axis=1))
    df_cleaned.drop(columns=['Start_Year', 'Start_Month', 'Start_Day'], inplace=True)

    impact_cols = ['Total_Deaths', 'No_Injured', 'No_Affected', 'Total_Affected']
    for col in impact_cols:
        df_cleaned[col] = df_cleaned[col].fillna(0)
    
    df_cleaned['Severity_Level'] = np.where(
        (df_cleaned['Total_Affected'] > 1000) | (df_cleaned['Total_Deaths'] > 10),
        1, 0
    )

    # --- 3. Spatial/Numerical Imputation (FIXED: Removed inplace=True for robust code) ---
    df_cleaned['Latitude'] = pd.to_numeric(df_cleaned['Latitude'], errors='coerce')
    df_cleaned['Longitude'] = pd.to_numeric(df_cleaned['Longitude'], errors='coerce')
    
    numerical_to_impute = ['Latitude', 'Longitude', 'CPI', 'Dis_Mag_Value']
    for col in numerical_to_impute:
        median_val = df_cleaned[col].median()
        # FIX: Assign back to the column instead of using inplace=True
        df_cleaned[col] = df_cleaned[col].fillna(median_val) 

    # --- 4. Feature Scaling (Standardization) (Remains the same) ---
    scaling_cols = ['Total_Deaths', 'No_Injured', 'No_Affected', 'Total_Affected', 'Dis_Mag_Value', 'CPI']
    scaler = StandardScaler()
    df_cleaned[scaling_cols] = scaler.fit_transform(df_cleaned[scaling_cols])

    # --- 5. Final Feature Selection (Remains the same) ---
    final_cols = [
        'ID_No', 'Start_Date', 'Country', 'ISO', 'Disaster_Subgroup', 'Disaster_Type',
        'Disaster_Subtype', 'Location', 'Latitude', 'Longitude', 
        'Dis_Mag_Value', 'CPI', 'Total_Deaths', 'Total_Affected', 'Severity_Level'
    ]
    df_final = df_cleaned[final_cols].copy()
    
    return df_final

if __name__ == '__main__':
    # --- PATH FIX: Create robust absolute paths for both input and output files ---
    base_dir = os.path.dirname(__file__)
    
    input_file_path = os.path.abspath(os.path.join(base_dir, '../data/Disaster2021.csv'))
    output_scaled_path = os.path.abspath(os.path.join(base_dir, '../data/Disaster2021_Scaled.csv'))
    
    df_preprocessed = preprocess_and_clean_data(input_file_path)
    
    if df_preprocessed.empty:
        print("Cannot run EDA: Preprocessing failed or returned an empty DataFrame.")
    else:
        # --- Save the Cleaned and Scaled Data ---
        df_preprocessed.to_csv(output_scaled_path, index=False)
        print("\n" + "="*70)
        print("PREPROCESSING & SCALING COMPLETE. Data saved for Geocoding step.")
        print(f"Data saved to: {output_scaled_path}")
        print("="*70)

        # --- EXPLORATORY DATA ANALYSIS (VISUALIZATION) ---
        
        # 1. Visualization Setup: Load RAW data using the ABSOLUTE PATH
        try:
            df_viz = pd.read_csv(input_file_path) # FIX IS HERE! Using the resolved path.
        except FileNotFoundError:
            print(f"FATAL ERROR: Could not read raw data for visualization at {input_file_path}. Please check if Disaster2021.csv exists in the data folder.")
            exit()
            
        df_viz = df_viz[df_viz['Year'] >= 2010].copy()
        
        numerical_cols_raw = ['Total_Deaths', 'Total_Affected']
        for col in numerical_cols_raw:
            df_viz[col] = df_viz[col].fillna(0)
            
        # Ensure the target variable aligns after filtering
        df_viz['Severity_Level'] = df_preprocessed['Severity_Level'].reset_index(drop=True) 
        
        # 2. Impact Distribution (Histograms/Boxplots)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, col in enumerate(numerical_cols_raw):
            sns.histplot(np.log1p(df_viz[col]), kde=True, ax=axes[i])
            axes[i].set_title(f'Log Distribution of {col}')
            axes[i].set_xlabel(f'log(1 + {col})')
        plt.suptitle("Impact Distribution (Log-Transformed)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(base_dir, '../data/eda_impact_distribution.png'))
        plt.show()

        # 3. Target Distribution and Relationship
        plt.figure(figsize=(5, 5))
        df_viz['Severity_Level'].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                                     colors=['#2ecc71', '#e74c3c'], 
                                                     labels=['0: Low Severity', '1: High Severity'])
        plt.ylabel('')
        plt.title('Target Variable Distribution (Severity_Level)')
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, '../data/eda_target_distribution.png'))
        plt.show()

        plt.figure(figsize=(8, 6))
        severity_rate = df_preprocessed.groupby('Disaster_Subgroup')['Severity_Level'].mean().sort_values(ascending=False)
        sns.barplot(x=severity_rate.index, y=severity_rate.values, palette='viridis')
        plt.title('High Severity Rate by Disaster Subgroup')
        plt.ylabel('Mean Severity Level (P(Severity=1))')
        plt.xlabel('Disaster Subgroup')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, '../data/eda_severity_rate_by_subgroup.png'))
        plt.show()
        
        print("\n--- Visualizations Complete ---")
        print("Plots saved as PNG files in the '../data/' directory.")