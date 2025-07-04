#!/usr/bin/env python3
"""
Quick viewer for trade export files
"""

import sys
import subprocess
from pathlib import Path

def main():
    export_dir = Path("trade_exports")
    
    if not export_dir.exists():
        print("❌ No trade_exports directory found. Run export_trades.py first.")
        return 1
    
    # Find the most recent Excel file
    excel_files = list(export_dir.glob("*.xlsx"))
    if not excel_files:
        print("❌ No Excel files found in trade_exports directory.")
        return 1
    
    # Sort by modification time, newest first
    excel_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if len(sys.argv) > 1:
        # Open specific alpha if provided
        alpha_name = sys.argv[1]
        matching_files = [f for f in excel_files if alpha_name in f.name]
        if matching_files:
            latest_file = matching_files[0]
        else:
            print(f"❌ No trade files found for {alpha_name}")
            print(f"Available files:")
            for f in excel_files[:5]:
                print(f"   - {f.name}")
            return 1
    else:
        # Open most recent file
        latest_file = excel_files[0]
    
    print(f"📊 Opening: {latest_file.name}")
    
    try:
        # Try to open with default application
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", str(latest_file)])
        elif sys.platform == "win32":  # Windows
            subprocess.run(["start", str(latest_file)], shell=True)
        else:  # Linux
            subprocess.run(["xdg-open", str(latest_file)])
        print("✅ File opened successfully!")
    except Exception as e:
        print(f"❌ Failed to open file: {e}")
        print(f"📁 File location: {latest_file.absolute()}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 