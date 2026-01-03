"""
Customer Segmentation using K-Means Clustering with PyCaret
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.clustering import *
from sklearn.cluster import KMeans

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def load_data(filepath='data/Mall_Customers.csv'):
    """Load customer data from CSV file."""
    df = pd.read_csv(filepath)
    print(f"Data loaded: {len(df)} customers")
    print(f"Columns: {list(df.columns)}")
    return df


def prepare_data(df):
    """Prepare data for clustering by selecting relevant features."""
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    data_for_clustering = df[features].copy()
    return data_for_clustering


def find_optimal_clusters(X, k_range=range(2, 11)):
    """Use Elbow method to find optimal number of clusters."""
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=123, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Plot Elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=10)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.title('Elbow Method - Optimal K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_range)
    plt.tight_layout()
    plt.show()
    
    return inertias


def initialize_pycaret(data):
    """Initialize PyCaret clustering setup with normalization."""
    cluster_setup = setup(
        data=data,
        normalize=True,
        session_id=123
    )
    print("PyCaret setup initialized")
    return cluster_setup


def create_kmeans_model(num_clusters=5):
    """Create and train K-Means clustering model."""
    kmeans_model = create_model(
        model='kmeans',
        num_clusters=num_clusters
    )
    print(f"K-Means model created with {num_clusters} clusters")
    return kmeans_model


def assign_clusters(model, original_df):
    """Assign cluster labels to original dataframe."""
    results = assign_model(model)
    original_df['Cluster'] = results['Cluster']
    
    # Display cluster distribution
    cluster_counts = original_df['Cluster'].value_counts().sort_index()
    print("\nCluster distribution:")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} customers ({count/len(original_df)*100:.1f}%)")
    
    return original_df


def analyze_clusters(df):
    """Analyze cluster characteristics and provide business insights."""
    cluster_summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
    
    print("\n" + "="*80)
    print("CLUSTER CHARACTERISTICS")
    print("="*80)
    print(cluster_summary.round(1))
    print("\n" + "="*80)
    
    # Business interpretation
    print("\nBUSINESS INSIGHTS:")
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster_id]
        avg_age = cluster_data['Age'].mean()
        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_spending = cluster_data['Spending Score (1-100)'].mean()
        count = len(cluster_data)
        
        print(f"\nCluster {cluster_id} ({count} customers, {count/len(df)*100:.1f}%):")
        print(f"  Avg Age: {avg_age:.0f} years")
        print(f"  Avg Income: ${avg_income:.0f}k")
        print(f"  Avg Spending Score: {avg_spending:.0f}/100")
        
        # Segment classification
        if avg_income < 40 and avg_spending < 40:
            segment = "Budget-Conscious"
            strategy = "Discounts, loyalty programs"
        elif avg_income < 40 and avg_spending >= 40:
            segment = "Young Enthusiasts"
            strategy = "Trendy products, installment plans"
        elif avg_income >= 40 and avg_income < 70 and avg_spending >= 40 and avg_spending < 60:
            segment = "Average Customers"
            strategy = "Standard offers, rewards programs"
        elif avg_income >= 70 and avg_spending < 40:
            segment = "Wealthy Cautious"
            strategy = "Premium products with value proposition"
        elif avg_income >= 40 and avg_spending >= 60:
            segment = "High-Value Customers"
            strategy = "Luxury products, VIP treatment"
        else:
            segment = "Mixed Segment"
            strategy = "Diversified approach"
        
        print(f"  Segment: {segment}")
        print(f"  Strategy: {strategy}")
    
    return cluster_summary


def visualize_clusters(df):
    """Create visualizations for cluster analysis."""
    
    # Convert cluster labels to numeric (PyCaret returns 'Cluster 0', 'Cluster 1', etc.)
    cluster_numeric = df['Cluster'].str.replace('Cluster ', '').astype(int)
    
    # 1. Income vs Spending
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        df['Annual Income (k$)'],
        df['Spending Score (1-100)'],
        c=cluster_numeric,
        cmap='viridis',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Annual Income (k$)', fontsize=12)
    plt.ylabel('Spending Score (1-100)', fontsize=12)
    plt.title('Customer Segmentation: Income vs Spending', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 2. Age vs Spending
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        df['Age'],
        df['Spending Score (1-100)'],
        c=cluster_numeric,
        cmap='viridis',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Spending Score (1-100)', fontsize=12)
    plt.title('Customer Segmentation: Age vs Spending', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 3. 3D scatter plot
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        df['Age'],
        df['Annual Income (k$)'],
        df['Spending Score (1-100)'],
        c=cluster_numeric,
        cmap='viridis',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    ax.set_xlabel('Age', fontsize=11, labelpad=10)
    ax.set_ylabel('Annual Income (k$)', fontsize=11, labelpad=10)
    ax.set_zlabel('Spending Score', fontsize=11, labelpad=10)
    ax.set_title('Customer Segmentation - 3D View', fontsize=14, fontweight='bold', pad=20)
    
    plt.colorbar(scatter, label='Cluster', pad=0.1)
    plt.tight_layout()
    plt.show()
    
    # 4. Box plots for feature comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    df.boxplot(column='Age', by='Cluster', ax=axes[0])
    axes[0].set_title('Age by Cluster')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Age (years)')
    
    df.boxplot(column='Annual Income (k$)', by='Cluster', ax=axes[1])
    axes[1].set_title('Income by Cluster')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Annual Income (k$)')
    
    df.boxplot(column='Spending Score (1-100)', by='Cluster', ax=axes[2])
    axes[2].set_title('Spending by Cluster')
    axes[2].set_xlabel('Cluster')
    axes[2].set_ylabel('Spending Score')
    
    plt.suptitle('Feature Comparison Across Clusters', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_pycaret(model):
    """Generate PyCaret visualizations."""
    # Cluster plot (PCA)
    plot_model(model, plot='cluster')
    
    # Elbow plot
    plot_model(model, plot='elbow')
    
    # Silhouette plot
    plot_model(model, plot='silhouette')


def save_results(model, df, model_path='models/customer_segmentation_model', 
                 data_path='data/customers_with_clusters.csv'):
    """Save trained model and data with cluster assignments."""
    # Save model
    save_model(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save data with clusters
    df.to_csv(data_path, index=False)
    print(f"Data with clusters saved to: {data_path}")


def predict_new_customer(model, age, income, spending_score):
    """Predict cluster for a new customer."""
    new_customer = pd.DataFrame({
        'Age': [age],
        'Annual Income (k$)': [income],
        'Spending Score (1-100)': [spending_score]
    })
    
    prediction = predict_model(model, data=new_customer)
    assigned_cluster = prediction['Cluster'].values[0]
    
    print(f"\nNew customer profile:")
    print(f"  Age: {age} years")
    print(f"  Annual Income: ${income}k")
    print(f"  Spending Score: {spending_score}/100")
    print(f"  Assigned to Cluster: {assigned_cluster}")
    
    return assigned_cluster


def main():
    """Main execution pipeline."""
    print("="*80)
    print("CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING")
    print("="*80)
    
    # Load data
    df = load_data('data/Mall_Customers.csv')
    
    # Prepare data
    data_for_clustering = prepare_data(df)
    
    # Initialize PyCaret
    cluster_setup = initialize_pycaret(data_for_clustering)
    
    # Get preprocessed data
    X = get_config('X')
    
    # Find optimal number of clusters
    print("\nFinding optimal number of clusters...")
    find_optimal_clusters(X)
    
    # Create K-Means model (adjust num_clusters based on Elbow plot)
    optimal_k = 5  # Change based on Elbow plot
    kmeans_model = create_kmeans_model(num_clusters=optimal_k)
    
    # Assign clusters
    df = assign_clusters(kmeans_model, df)
    
    # Analyze clusters
    cluster_summary = analyze_clusters(df)
    
    # Visualize clusters
    print("\nGenerating visualizations...")
    visualize_clusters(df)
    
    # PyCaret visualizations
    print("\nGenerating PyCaret visualizations...")
    visualize_pycaret(kmeans_model)
    
    # Save results
    save_results(kmeans_model, df)
    
    # Example: Predict for new customer
    print("\n" + "="*80)
    print("EXAMPLE: Predicting cluster for new customer")
    print("="*80)
    predict_new_customer(kmeans_model, age=28, income=75, spending_score=80)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    main()
