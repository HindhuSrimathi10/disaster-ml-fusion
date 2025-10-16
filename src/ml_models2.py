import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
from preprocessing2 import SocialMediaPreprocessor

class SocialMediaDisasterClassifier:
    def __init__(self):
        self.preprocessor = SocialMediaPreprocessor()
        self.models = {}
        self.best_model = None
        self.model_performance = {}
        self.dataset_info = {}
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess social media data - CUSTOMIZED FOR YOUR DATASET"""
        print(f"📥 Loading data from {file_path}")
        
        # Load dataset
        try:
            df = pd.read_csv("C:\\Users\\indhu\\Disaster_ML_Fusion - Copy\\data\\DisasterTweets.csv")
            print(f"✅ Dataset loaded: {df.shape}")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            raise
        
        print(f"📋 Columns: {list(df.columns)}")
        
        # FOR YOUR SPECIFIC DATASET - Use exact column names
        text_column = 'Tweets'  # Your text column
        label_column = 'Disaster'  # Your label column
        
        if text_column not in df.columns:
            available_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['tweet', 'text', 'message'])]
            if available_cols:
                text_column = available_cols[0]
                print(f"⚠️  Using alternative text column: {text_column}")
            else:
                raise ValueError(f"Text column not found. Available columns: {list(df.columns)}")
        
        if label_column not in df.columns:
            available_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['disaster', 'label', 'category', 'type'])]
            if available_cols:
                label_column = available_cols[0]
                print(f"⚠️  Using alternative label column: {label_column}")
            else:
                raise ValueError(f"Label column not found. Available columns: {list(df.columns)}")
        
        print(f"🔤 Using text column: '{text_column}'")
        print(f"🏷️  Using label column: '{label_column}'")
        
        # Display dataset info
        print(f"\n📊 Dataset Info:")
        print(f"   Total samples: {len(df)}")
        print(f"   Label distribution:")
        label_counts = df[label_column].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"     - {label}: {count} ({percentage:.1f}%)")
        
        # Check for missing values
        missing_text = df[text_column].isna().sum()
        missing_labels = df[label_column].isna().sum()
        
        if missing_text > 0:
            print(f"⚠️  Removing {missing_text} rows with missing text...")
            df = df.dropna(subset=[text_column])
        
        if missing_labels > 0:
            print(f"⚠️  Removing {missing_labels} rows with missing labels...")
            df = df.dropna(subset=[label_column])
        
        # Store dataset info
        self.dataset_info = {
            'text_column': text_column,
            'label_column': label_column,
            'label_distribution': df[label_column].value_counts().to_dict(),
            'total_samples': len(df),
            'disaster_types': df[label_column].unique().tolist()
        }
        
        # Preprocess data
        print("🔄 Preprocessing data...")
        processed_df = self.preprocessor.preprocess_data(df, text_column, label_column)
        
        # Vectorize text
        print("🔡 Vectorizing text...")
        X = self.preprocessor.vectorize_text(processed_df['processed_text'], fit=True)
        y = processed_df['encoded_labels']
        
        print(f"✅ Preprocessing complete: X.shape={X.shape}, y.shape={y.shape}")
        return X, y, processed_df
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        print("🤖 Splitting data for training...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Testing set: {X_test.shape[0]} samples")
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='linear', probability=True, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'naive_bayes': MultinomialNB()
        }
        
        best_accuracy = 0
        best_model_name = None
        
        print("\n🏋️ Training Models:")
        print("-" * 50)
        
        for name, model in models.items():
            print(f"   Training {name}...")
            try:
                model.fit(X_train, y_train)
                self.models[name] = model
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                self.model_performance[name] = {
                    'accuracy': accuracy,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                print(f"   ✅ {name} Accuracy: {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = name
                    self.best_model = model
                    
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
        
        if self.best_model is None:
            raise ValueError("All models failed to train!")
        
        print(f"\n🏆 Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
        
        # Save performance metrics
        performance_df = pd.DataFrame({
            'Model': list(self.model_performance.keys()),
            'Accuracy': [perf['accuracy'] for perf in self.model_performance.values()]
        })
        os.makedirs('models', exist_ok=True)
        performance_df.to_csv('models/social_media_model_performance.csv', index=False)
        
        return self.best_model
    
    def predict_disaster(self, text):
        """Predict disaster type from text"""
        if self.best_model is None:
            raise ValueError("No model trained. Please train a model first.")
        
        # Preprocess text
        X, processed_text = self.preprocessor.preprocess_single_text(text)
        
        # Predict
        prediction_encoded = self.best_model.predict(X)[0]
        probabilities = self.best_model.predict_proba(X)[0]
        
        # Get disaster type
        disaster_type = self.preprocessor.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence
        confidence = probabilities[prediction_encoded]
        
        return {
            'disaster_type': disaster_type,
            'confidence': confidence,
            'processed_text': processed_text,
            'all_probabilities': dict(zip(
                self.preprocessor.label_encoder.classes_, 
                probabilities
            ))
        }
    
    def save_model(self, file_path='models/social_media_disaster_model.joblib'):
        """Save the trained model and preprocessor"""
        os.makedirs('models', exist_ok=True)
        model_data = {
            'model': self.best_model,
            'preprocessor': self.preprocessor,
            'label_encoder': self.preprocessor.label_encoder,
            'performance': self.model_performance,
            'dataset_info': self.dataset_info
        }
        joblib.dump(model_data, file_path)
        print(f"💾 Model saved to {file_path}")
    
    def load_model(self, file_path='models/social_media_disaster_model.joblib'):
        """Load a trained model and preprocessor"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} not found")
        
        model_data = joblib.load(file_path)
        self.best_model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.model_performance = model_data['performance']
        self.dataset_info = model_data.get('dataset_info', {})
        print(f"📂 Model loaded from {file_path}")

# Training function specifically for your dataset
def train_social_media(dataset_path=None):
    """Function to train the social media disaster classification model"""
    classifier = SocialMediaDisasterClassifier()
    
    # Use provided dataset path or find it automatically
    if dataset_path is None:
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        dataset_path = project_root / "data" / "DisasterTweets.csv"
    
    print(f"🎯 Training with dataset: {dataset_path}")
    
    try:
        X, y, df = classifier.load_and_preprocess_data(str(dataset_path))
        
        # Train models
        best_model = classifier.train_models(X, y)
        
        # Save model
        classifier.save_model()
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Dataset used: {dataset_path}")
        print(f"🏷️  Labels: {classifier.preprocessor.label_encoder.classes_}")
        
        return classifier
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    train_social_media()