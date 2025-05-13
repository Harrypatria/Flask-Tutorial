"""
Machine Learning Model Training Script
Author: Harry Patria
Version: 1.1.0

Trains and serializes a Multiple Linear Regression model for startup profit prediction.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
from config import MODEL_CONFIG
from pathlib import Path

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_data(filepath: str) -> tuple:
    """Load and preprocess dataset"""
    df = pd.read_csv(filepath)
    
    # Validate dataset structure
    required_columns = ['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataset missing required columns. Expected: {required_columns}")
    
    # Clean column names
    df.columns = df.columns.str.replace(' ', '_')
    return df

def preprocess_data(df: pd.DataFrame) -> tuple:
    """Feature engineering and train-test split"""
    X = df[['R&D_Spend', 'Administration', 'Marketing_Spend', 'State']]
    y = df['Profit']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['R&D_Spend', 'Administration', 'Marketing_Spend']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), ['State'])
        ],
        remainder='drop'
    )
    
    return X, y, preprocessor

def train_model(X, y, preprocessor) -> Pipeline:
    """Train model with preprocessing pipeline"""
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model.fit(X, y)
    return model

def evaluate_model(model, X, y) -> dict:
    """Calculate and return evaluation metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    y_pred = model.predict(X)
    return {
        'r2_score': r2_score(y, y_pred),
        'mae': mean_absolute_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'coefficients': model.named_steps['regressor'].coef_,
        'intercept': model.named_steps['regressor'].intercept_
    }

def save_model_artifacts(model, metrics: dict):
    """Save model and metadata"""
    # Create output directory if not exists
    Path(MODEL_CONFIG['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, MODEL_CONFIG['MODEL_PATH'])
    
    # Save model metadata
    metadata = {
        'training_date': pd.Timestamp.now().isoformat(),
        'model_version': MODEL_CONFIG['VERSION'],
        'features': MODEL_CONFIG['FEATURE_COLUMNS'],
        'state_encoding': MODEL_CONFIG['STATE_ENCODING'],
        'metrics': metrics
    }
    joblib.dump(metadata, MODEL_CONFIG['METADATA_PATH'])

def main():
    try:
        print("🚀 Starting model training pipeline...")
        
        # Data loading
        print("📊 Loading dataset...")
        df = load_data('data/50_Startup.csv')
        
        # Preprocessing
        print("⚙️ Preprocessing data...")
        X, y, preprocessor = preprocess_data(df)
        
        # Model training
        print("🧠 Training model...")
        model = train_model(X, y, preprocessor)
        
        # Model evaluation
        print("📈 Evaluating model...")
        metrics = evaluate_model(model, X, y)
        print(f"✅ Model trained successfully with R² score: {metrics['r2_score']:.3f}")
        
        # Save artifacts
        print("💾 Saving model artifacts...")
        save_model_artifacts(model, metrics)
        
        # Sample prediction
        sample_input = pd.DataFrame([{
            'R&D_Spend': 160000,
            'Administration': 135000,
            'Marketing_Spend': 450000,
            'State': 'California'
        }])
        prediction = model.predict(sample_input)
        print(f"🔮 Sample prediction: ${prediction[0]:,.2f}")
        
    except Exception as e:
        print(f"❌ Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
