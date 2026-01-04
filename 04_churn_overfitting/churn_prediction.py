"""
Churn Prediction using PyCaret - Classification
Demonstrates overfitting detection using cross-validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def load_data(filepath='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """Load telco customer churn dataset."""
    df = pd.read_csv(filepath)
    print(f"Data loaded: {len(df)} customers, {len(df.columns)} features")
    return df


def explore_data(df):
    """Basic data exploration and visualization."""
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    # Target distribution
    print("\nChurn distribution:")
    print(df['Churn'].value_counts())
    print(f"Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']*100:.1f}%")
    
    # Visualize
    plt.figure(figsize=(8, 5))
    df['Churn'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('Churn Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Churn', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def preprocess_data(df):
    """Clean and prepare data for modeling."""
    # Drop customerID (no predictive value)
    df = df.drop('customerID', axis=1)
    
    # Fix TotalCharges (convert to numeric)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    print("\nData preprocessing completed")
    print(f"Final shape: {df.shape}")
    
    return df


def setup_pycaret(df):
    """Initialize PyCaret classification setup."""
    print("\n" + "="*60)
    print("PYCARET SETUP")
    print("="*60)
    
    clf_setup = setup(
        data=df,
        target='Churn',
        session_id=123,
        train_size=0.8,
        fold=5,  # 5-fold cross-validation
        fix_imbalance=False,
        remove_multicollinearity=False,
        normalize=True
    )
    
    print("\nPyCaret setup completed")
    return clf_setup


def compare_models_cv():
    """Compare multiple models using cross-validation."""
    print("\n" + "="*60)
    print("MODEL COMPARISON (Cross-Validation)")
    print("="*60)
    print("\nTraining and comparing ~15 ML algorithms...")
    print("This may take a few minutes...\n")
    
    # Compare models sorted by AUC
    best_models = compare_models(sort='AUC', n_select=10)
    
    print("\n✅ Model comparison completed")
    print("\nNOTE: Results above are from CROSS-VALIDATION")
    print("Models were tested on data they haven't seen during training")
    
    return best_models


def train_best_model(best_models):
    """Train the best model from comparison."""
    best_model = best_models[0]
    
    print("\n" + "="*60)
    print(f"TRAINING BEST MODEL: {type(best_model).__name__}")
    print("="*60)
    
    final_model = create_model(best_model)
    
    print("\n✅ Model trained successfully")
    return final_model


def demonstrate_overfitting():
    """Create an overfitted model to demonstrate the problem."""
    print("\n" + "="*60)
    print("⚠️  OVERFITTING DEMONSTRATION")
    print("="*60)
    print("\nCreating a highly complex Decision Tree without constraints...")
    print("(This is an ANTI-PATTERN - showing what NOT to do!)\n")
    
    # Create overfitted model
    overfit_tree = DecisionTreeClassifier(
        max_depth=None,  # No depth limit
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=123
    )
    
    overfit_model = create_model(overfit_tree, verbose=False)
    
    # Compare training vs CV accuracy
    X_train = get_config('X_train')
    y_train = get_config('y_train')
    
    # Training accuracy
    train_predictions = overfit_model.predict(X_train)
    train_accuracy = (train_predictions == y_train).mean()
    
    # Cross-validation accuracy
    cv_scores = cross_val_score(
        overfit_tree, 
        X_train, 
        y_train, 
        cv=5,
        scoring='accuracy'
    )
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Analyze results
    print("\n" + "="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)
    print(f"\n1️⃣ Training Accuracy:         {train_accuracy*100:.2f}%")
    print(f"   (Tested on training data)")
    print(f"\n2️⃣ Cross-Validation Accuracy: {cv_mean*100:.2f}% (±{cv_std*100:.2f}%)")
    print(f"   (Tested on unseen data)")
    print(f"   Fold scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
    
    difference = train_accuracy - cv_mean
    print(f"\n⚠️  DIFFERENCE: {difference*100:.2f}%")
    print("="*60)
    
    if difference > 0.10:
        print("\n🚨 OVERFITTING DETECTED!")
        print("\nWhat happened:")
        print("- Model 'memorized' training data instead of learning patterns")
        print("- Performs MUCH WORSE on new data (CV)")
        print("- Would FAIL in production")
        print("\n💡 Solution:")
        print("- Use simpler models or add constraints (max_depth, min_samples)")
        print("- Add regularization")
        print("- Collect more training data")
        print("- ALWAYS evaluate using cross-validation")
    elif difference > 0.05:
        print("\n⚠️  Mild overfitting detected")
    else:
        print("\n✅ Model is OK")
    
    return overfit_model, train_accuracy, cv_mean, difference


def test_model(model):
    """Test model on hold-out test set."""
    print("\n" + "="*60)
    print("TESTING ON HOLD-OUT SET")
    print("="*60)
    
    test_predictions = predict_model(model)
    
    print("\n✅ Testing completed")
    print("\n💡 Compare test results with CV results:")
    print("   Similar scores → Good generalization ✅")
    print("   Test << CV → Overfitting ❌")
    
    return test_predictions


def visualize_model(model):
    """Generate model visualizations."""
    print("\n" + "="*60)
    print("MODEL VISUALIZATIONS")
    print("="*60)
    
    # Confusion Matrix
    print("\n1. Confusion Matrix")
    plot_model(model, plot='confusion_matrix')
    
    # AUC Curve
    print("\n2. ROC AUC Curve")
    plot_model(model, plot='auc')
    
    # Feature Importance
    print("\n3. Feature Importance")
    plot_model(model, plot='feature')


def save_model_file(model, filepath='models/churn_prediction_model'):
    """Save trained model to disk."""
    save_model(model, filepath)
    print(f"\n✅ Model saved to: {filepath}")


def predict_new_customer(model):
    """Example prediction for a new customer."""
    print("\n" + "="*60)
    print("NEW CUSTOMER PREDICTION")
    print("="*60)
    
    # High-risk customer profile
    new_customer = pd.DataFrame({
        'gender': ['Female'],
        'SeniorCitizen': [0],
        'Partner': ['No'],
        'Dependents': ['No'],
        'tenure': [2],  # New customer
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['No'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],  # High risk
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [70.0],
        'TotalCharges': [140.0]
    })
    
    print("\n🆕 Customer Profile:")
    print("  - Tenure: 2 months (NEW)")
    print("  - Contract: Month-to-month (RISK)")
    print("  - Internet: Fiber optic")
    print("  - No additional services\n")
    
    # Predict
    prediction = predict_model(model, data=new_customer)
    
    churn_prediction = prediction['prediction_label'].values[0]
    churn_score = prediction['prediction_score'].values[0]
    
    print(f"\n🎯 PREDICTION: {churn_prediction}")
    print(f"📊 Churn Probability: {churn_score*100:.1f}%")
    
    if churn_prediction == 'Yes':
        print("\n🚨 ALERT! Customer likely to churn")
        print("\n💡 Recommended actions:")
        print("  1. Contact within 48h")
        print("  2. Offer discount for annual contract")
        print("  3. Add free OnlineSecurity for 3 months")
        print("  4. Check service satisfaction")
    else:
        print("\n✅ Customer likely to stay")
        print("\nMonitor: Low tenure + monthly contract = potential risk")


def main():
    """Main execution pipeline."""
    print("="*60)
    print("CHURN PREDICTION - OVERFITTING DETECTION")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Explore
    explore_data(df)
    
    # Preprocess
    df = preprocess_data(df)
    
    # Setup PyCaret
    setup_pycaret(df)
    
    # Compare models (using CV)
    best_models = compare_models_cv()
    
    # Train best model
    final_model = train_best_model(best_models)
    
    # Demonstrate overfitting
    overfit_model, train_acc, cv_acc, diff = demonstrate_overfitting()
    
    # Test on hold-out set
    test_predictions = test_model(final_model)
    
    # Visualizations
    visualize_model(final_model)
    
    # Save model
    save_model_file(final_model)
    
    # Example prediction
    predict_new_customer(final_model)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. Always use Cross-Validation, not training accuracy")
    print("2. Training acc >> CV acc = Overfitting")
    print("3. Simpler models often better in production")
    print("4. Interpretability matters for business decisions")


if __name__ == "__main__":
    main()
