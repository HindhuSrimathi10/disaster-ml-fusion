from pathlib import Path
import os

def check_files():
    """Simple script to check what files exist"""
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    print("📁 Project Structure:")
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Data exists: {data_dir.exists()}")
    
    if data_dir.exists():
        print("\n📊 Files in data directory:")
        for file in data_dir.iterdir():
            if file.is_file():
                print(f"  📄 {file.name} ({file.stat().st_size} bytes)")
            elif file.is_dir():
                print(f"  📁 {file.name}/")
    
    # Check if your specific file exists
    your_file = data_dir / "DisasterTweets.csv"
    if your_file.exists():
        print(f"\n✅ Found your file: {your_file}")
        print(f"   Size: {your_file.stat().st_size} bytes")
    else:
        print(f"\n❌ Your file not found: DisasterTweets.csv")
        print("💡 Please copy your DisasterTweets.csv file to the data/ folder")

if __name__ == "__main__":
    check_files()