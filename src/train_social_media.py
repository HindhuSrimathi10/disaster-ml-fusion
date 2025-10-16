import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    print("🚀 Starting Social Media Disaster Model Training")
    print("=" * 50)
    
    try:
        # Import the available functions/classes from your ml_models2.py
        from ml_models2 import SocialMediaDisasterClassifier, train_social_media
        
        # Option 1: Use the train_social_media function if it exists
        try:
            classifier = train_social_media()
        except:
            # Option 2: Manual training if function doesn't exist
            print("Using manual training approach...")
            classifier = SocialMediaDisasterClassifier()
            dataset_path = project_root / "data" / "DisasterTweets.csv"
            
            if not dataset_path.exists():
                print(f"❌ Dataset not found: {dataset_path}")
                return None
                
            print(f"📁 Using dataset: {dataset_path}")
            X, y, df = classifier.load_and_preprocess_data(str(dataset_path))
            classifier.train_models(X, y)
            classifier.save_model()
        
        if classifier:
            print("\n✅ Training completed successfully!")
            print("\n📊 Model Information:")
            print(f"   - Best model: {classifier.best_model.__class__.__name__}")
            
            # Get best accuracy
            best_accuracy = 0
            for perf in classifier.model_performance.values():
                if isinstance(perf, dict) and 'accuracy' in perf:
                    best_accuracy = max(best_accuracy, perf['accuracy'])
                elif isinstance(perf, (int, float)):
                    best_accuracy = max(best_accuracy, perf)
            
            print(f"   - Best accuracy: {best_accuracy:.4f}")
            print(f"   - Labels: {list(classifier.preprocessor.label_encoder.classes_)}")
            print(f"   - Dataset: {classifier.dataset_info.get('text_column', 'N/A')} -> {classifier.dataset_info.get('label_column', 'N/A')}")
            
            # Test some predictions
            print("\n🧪 Test Predictions:")
            test_texts = [
                "Earthquake shaking buildings in downtown area",
                "Flood warning issued for river banks", 
                "Wildfire spreading rapidly in forest area",
                "Just had dinner with friends, great weather today"
            ]
            
            for text in test_texts:
                try:
                    result = classifier.predict_disaster(text)
                    print(f"   '{text}'")
                    print(f"   → {result['disaster_type']} ({result['confidence']:.2%} confidence)")
                    print()
                except Exception as e:
                    print(f"   '{text}' -> Error: {e}")
        else:
            print("\n❌ Training failed.")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Available imports from ml_models2:")
        import ml_models2
        available_items = [item for item in dir(ml_models2) if not item.startswith('_')]
        for item in available_items:
            print(f"   - {item}")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()