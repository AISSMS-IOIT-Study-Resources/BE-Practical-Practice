import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

def load_sample_data():
    data = {
        'feature_1': np.random.normal(loc=50, scale=10, size=100),
        'feature_2': np.random.normal(loc=30, scale=5, size=100),
        'feature_3': np.random.uniform(low=10, high=100, size=100),
        'category': np.random.choice(['A', 'B', 'C'], size=100)
    }

    df = pd.DataFrame(data)
    return df

def plot_histogram(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols].hist(bins=15, figsize=(10, 6), layout=(1, len(numeric_cols)))
    plt.suptitle("Histograms of Numerical Features")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_scatter(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="feature_1", y="feature_2", hue="category", palette="Set2")
    plt.title("Scatter Plot: Feature 1 vs Feature 2")
    plt.show()

def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.show()

def main():
    df = load_sample_data()
    print("Sample Data Head:")
    print(df.head(), "\n")

    plot_histogram(df)
    plot_scatter(df)
    plot_correlation_heatmap(df)

main()