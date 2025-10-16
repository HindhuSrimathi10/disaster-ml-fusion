import shutil
from pathlib import Path

def copy_dataset():
    """Copy your dataset from Downloads to the project data folder"""
    
    # Source file (your dataset)
    source_path = Path("C:/Users/indhu/Downloads/DisasterTweets.csv")
    
    # Destination in project
    project_root = Path(__file__).parent.parent
    dest_path = project_root / "data" / "DisasterTweets.csv"
    
    print(f"Source: {source_path}")
    print(f"Destination: {dest_path}")
    
    # Check if source exists
    if not source_path.exists():
        print(f"❌ Source file not found: {source_path}")
        print("💡 Please make sure your DisasterTweets.csv is in the Downloads folder")
        return
    
    # Create data directory if it doesn't exist
    dest_path.parent.mkdir(exist_ok=True)
    
    try:
        # Copy the file
        shutil.copy2(source_path, dest_path)
        print(f"✅ Successfully copied dataset to: {dest_path}")
        print(f"📊 File size: {dest_path.stat().st_size} bytes")
    except Exception as e:
        print(f"❌ Error copying file: {e}")

if __name__ == "__main__":
    copy_dataset()