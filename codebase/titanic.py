import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

#------------------------------
# Defiene colors
#------------------------------
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CAYAN = "\033[96m"
MAGNETA = "\033[95m"
YELLOW = "\033[93m"
RESET = "\033[0m"

#------------------------------
# Configuration
# ------------------------------
dataset_folder = "/home/wael/Main Vault/Personal Project/Titanc/datasets"
os.makedirs(dataset_folder, exist_ok=True)
url = "https://raw.githubusercontent.com/sitmbadept/sitmbadept.github.io/main/BDTM/R/titanic.csv"
output_path = os.path.join(dataset_folder, "titanic.csv")

# ------------------------------
# Download Dataset
# ------------------------------
resp = requests.get(url)
resp.raise_for_status()  # error if download failed

with open(output_path, "wb") as f:
    f.write(resp.content)

print(f"Downloaded Titanic dataset → {output_path}")

# ✅ Load and show first rows
df = pd.read_csv(output_path)


# ------------------------------
# Helper Functions
# ------------------------------
def check_csv_files(folder):
    """Check if CSV files already exist in the folder"""
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                return os.path.join(root, file)
    return None

def plots_exist(plots_folder, numeric_cols, has_scatterplot):
    """Check if all plots already exist"""
    # Check distribution plots
    for col in numeric_cols:
        plot_path = os.path.join(plots_folder, f"{col}_distribution.png")
        if not os.path.exists(plot_path):
            return False
    
    # Check scatterplot
    if has_scatterplot and not os.path.exists(os.path.join(plots_folder, "size_vs_price.png")):
        return False
    
    # Check correlation heatmap
    if not os.path.exists(os.path.join(plots_folder, "correlation_heatmap.png")):
        return False
    
    return True

# ------------------------------
# Data Overview
# ------------------------------
print(f"{BLUE}=== Titanic Dataset Overview ==={RESET}")
print(f"first 5 rows:\n{df.head()}")

print(f"\n{GREEN}Dataset Info: \n")
df.info()

print(f"\n{BLUE}Statistical Summary:\n {df.describe()} {RESET}")

print(f"\n{RED}Columns: {df.columns.tolist()}")



# ------------------------------
# 5. Exploratory Data Analysis (EDA) - Titanic
# ------------------------------
plots_folder = "../codebase/plots/"
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# Check for scatterplot columns (Titanic-specific)
size_columns = ['Age']
price_columns = ['Fare']
size_col = next((col for col in size_columns if col in df.columns), None)
price_col = next((col for col in price_columns if col in df.columns), None)
has_scatterplot = size_col and price_col

if plots_exist(plots_folder, numeric_cols, has_scatterplot):
    print(f"\n{GREEN}✓ All plots already exist in '{plots_folder}'{RESET}")
    print(f"{YELLOW}Skipping plot generation...{RESET}")
else:
    print(f"\n{MAGNETA}--- Creating Visualizations ---{RESET}")
    
    # Distribution plots for numeric columns
    for col in numeric_cols:
        plot_path = os.path.join(plots_folder, f"{col}_distribution.png")
        if not os.path.exists(plot_path):
            plt.figure(figsize=(8,5))
            sns.histplot(df[col].dropna(), bins=30, kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"{GREEN}✓ Created distribution plot: {col}{RESET}")
        else:
            print(f"{CAYAN}✓ Distribution plot already exists: {col}{RESET}")

    # Scatterplot: Age vs Fare
    if has_scatterplot:
        scatter_path = os.path.join(plots_folder, "age_vs_fare.png")
        if not os.path.exists(scatter_path):
            plt.figure(figsize=(8,5))
            sns.scatterplot(x=size_col, y=price_col, data=df)
            plt.title(f'{size_col} vs {price_col}')
            plt.xlabel(size_col)
            plt.ylabel(price_col)
            plt.tight_layout()
            plt.savefig(scatter_path)
            plt.close()
            print(f"{GREEN}✓ Created scatterplot: {size_col} vs {price_col}{RESET}")
        else:
            print(f"{CAYAN}✓ Scatterplot already exists{RESET}")

    # Correlation Heatmap
    corr_path = os.path.join(plots_folder, "correlation_heatmap.png")
    if not os.path.exists(corr_path):
        plt.figure(figsize=(10,8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(corr_path)
        plt.close()
        print(f"{GREEN}✓ Created correlation heatmap{RESET}")
    else:
        print(f"{CAYAN}✓ Correlation heatmap already exists{RESET}")


#  ------------------------------
# Gender vs Survived
# ------------------------------
gender_col = 'Sex'
target_col = 'Survived'

gender_plot_path = os.path.join(plots_folder, "sex_vs_survived.png")
if not os.path.exists(gender_plot_path):
    plt.figure(figsize=(6,5))
    sns.countplot(x=gender_col, hue=target_col, data=df, palette='Set2')
    plt.title('Survival Count by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.legend(title='Survived')
    plt.tight_layout()
    plt.savefig(gender_plot_path)
    plt.close()
    print(f"{GREEN}✓ Created gender vs survived plot{RESET}")
else:
    print(f"{CAYAN}✓ Gender vs survived plot already exists{RESET}")
