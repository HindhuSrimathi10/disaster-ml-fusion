import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml_models2 import SocialMediaDisasterClassifier

def retrain_with_improvements():
    print("🔄 Retraining with Improved Model")
    print("=" * 50)
    
    classifier = SocialMediaDisasterClassifier()
    dataset_path = project_root / "data" / "DisasterTweets.csv"
    
    print(f"📁 Dataset: {dataset_path}")
    
    try:
        # Load and preprocess with new improvements
        X, y, df = classifier.load_and_preprocess_data(str(dataset_path))
        
        # Train models
        classifier.train_models(X, y)
        
        # Save model
        classifier.save_model()
        
        print("\n✅ Improved model trained successfully!")
        
        # Test with problematic examples
        print("\n🧪 Testing Improved Model:")
        test_cases = [
            "Working on exciting project at office today. Productive day!",
            "Beautiful sunny day perfect for picnic in the park",
            "Just had amazing dinner with friends at restaurant",
            "Earthquake shaking buildings in downtown area!",
            "Wildfire spreading rapidly in forest area",
            "Flash flood warning issued for river valley"
        ]
        
        for text in test_cases:
            result = classifier.predict_disaster(text)
            print(f"   '{text}'")
            print(f"   → {result['disaster_type']} ({result['confidence']:.1%} confidence)")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    retrain_with_improvements()