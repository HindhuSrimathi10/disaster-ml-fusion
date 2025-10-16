import joblib
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_model():
    print("🧪 Testing Trained Model")
    print("=" * 40)
    
    model_path = project_root / "models" / "social_media_disaster_model.joblib"
    
    if not model_path.exists():
        print("❌ Model not found! Train the model first.")
        return
    
    try:
        # Load the trained model
        model_data = joblib.load(model_path)
        classifier = model_data['model']
        preprocessor = model_data['preprocessor']
        label_encoder = model_data['label_encoder']
        
        print("✅ Model loaded successfully!")
        print(f"🏷️  Available disaster types: {list(label_encoder.classes_)}")
        
        # Test predictions
        test_texts = [
            "Major drought affecting farm lands across multiple states #drought",
            "Wildfire spreading rapidly in forest area, evacuations underway #wildfire",
            "Earthquake felt in downtown area, buildings shaking #earthquake",
            "Flash floods warning issued for river valley #floods",
            "Hurricane approaching coastal areas, seek shelter #hurricane",
            "Tornado spotted near the city, take cover immediately #tornado",
            "Beautiful sunny day perfect for outdoor activities with friends",
            "Just had a great lunch at the new restaurant downtown"
        ]
        
        print("\n📊 Prediction Results:")
        print("-" * 60)
        
        for text in test_texts:
            # Preprocess
            X, processed_text = preprocessor.preprocess_single_text(text)
            
            # Predict
            prediction_encoded = classifier.predict(X)[0]
            disaster_type = label_encoder.inverse_transform([prediction_encoded])[0]
            confidence = classifier.predict_proba(X)[0][prediction_encoded]
            
            print(f"📝 Input: '{text}'")
            print(f"🎯 Prediction: {disaster_type}")
            print(f"📈 Confidence: {confidence:.1%}")
            print(f"🔧 Processed: {processed_text}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()