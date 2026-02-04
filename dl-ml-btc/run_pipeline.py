# run_pipeline.py - Main script to execute the full pipeline
# -*- coding: utf-8 -*-

import os
import sys
from datetime import datetime

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass  # Fallback if reconfiguration fails

def print_header(text):
    """Displays a stylized header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def run_fetcher(name, script_path):
    """Executes a data fetcher script."""
    print(f"🚀 Launching {name}...")
    print(f"   Script: {script_path}")
    
    if not os.path.exists(script_path):
        print(f"   ❌ Script not found: {script_path}")
        return False
    
    python_exe = sys.executable
    result = os.system(f'"{python_exe}" {script_path}')
    
    if result == 0:
        print(f"   ✅ {name} completed successfully\n")
        return True
    else:
        print(f"   ⚠️  {name} completed with warnings (code: {result})\n")
        return False

def main():
    """Main pipeline."""
    print_header("BTC DATA COLLECTION AND PROCESSING PIPELINE")
    print(f"📅 Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verify we are in the correct directory
    if not os.path.exists('api'):
        print("❌ ERROR: The 'api' folder does not exist.")
        print("   Make sure to run this script from the dl-ml-btc/ directory")
        sys.exit(1)
    
    # Step 1: Data fetching
    print_header("STEP 1/5: FETCHING HISTORICAL OHLCV DATA")
    run_fetcher("Historical Data (Binance)", "api/fetch_historical.py")
    
    print_header("STEP 2/5: FETCHING BLOCKCHAIN METRICS")
    run_fetcher("Blockchain Metrics", "api/fetch_blockchain.py")
    
    print_header("STEP 3/5: FETCHING MACRO INDICATORS")
    run_fetcher("Macro Indicators", "api/fetch_macro.py")
    
    print_header("STEP 4/5: FETCHING SENTIMENT DATA")
    run_fetcher("Sentiment Data", "api/fetch_sentiment.py")
    
    # Step 2: Feature engineering
    print_header("STEP 5/5: FEATURE GENERATION")
    success = run_fetcher("Feature engineering", "api/feature_engine.py")
    
    # Final Summary
    print_header("✅ PIPELINE COMPLETED")
    
    if success:
        print("🎉 All steps completed successfully!")
        print("\n📊 Final dataset available in:")
        print("   → data/features/btc_features_complete.csv")
        print("\nYou can now use this dataset to train your ML/DL models.")
    else:
        print("⚠️  The pipeline completed with warnings.")
        print("   Check the messages above for details.")
    
    print("\n" + "=" * 80 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
