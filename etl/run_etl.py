import sys
sys.path.append(r"D:\Desktop\project\etl")

import os
from extract import load_all
from transform import transform_all

# ─────────────────────────────────────────────
# Configuration: Switch between demo and full data
# ─────────────────────────────────────────────
MODE = "demo"   # Switch to "full" to view actual data

PATHS = {
    "demo": {
        "raw":     r"D:\Desktop\project\data\raw",
        "cleaned": r"D:\Desktop\project\data\cleaned",
    },
    "full": {
        "raw":     r"D:\Desktop\project\data\raw_full",
        "cleaned": r"D:\Desktop\project\data\cleaned_full",
    }
}

RAW_DIR     = PATHS[MODE]["raw"]
CLEANED_DIR = PATHS[MODE]["cleaned"]

# ─────────────────────────────────────────────

def save_cleaned(tables):
    os.makedirs(CLEANED_DIR, exist_ok=True)
    for name, df in tables.items():
        out_path = os.path.join(CLEANED_DIR, f"{name}.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[Save] {name}.csv → {len(df)} rows")

LINEAGE_PATH = r"D:\Desktop\project\lineage\lineage.json"

def reset_lineage():
    """Clear lineage.json before each run to prevent duplicate accumulation."""
    os.makedirs(os.path.dirname(LINEAGE_PATH), exist_ok=True)
    with open(LINEAGE_PATH, 'w', encoding='utf-8') as f:
        import json
        json.dump({"lineage_records": []}, f)
    print("[Lineage] lineage.json reset sucessfully")

if __name__ == "__main__":
    reset_lineage()
    print(f"MODE: {MODE.upper()} | raw → {RAW_DIR}")
    print("=" * 40)
    print("Step 1: Extract")
    print("=" * 40)
    raw = load_all(RAW_DIR)

    print("\n" + "=" * 40)
    print("Step 2: Transform")
    print("=" * 40)
    cleaned = transform_all(raw)

    print("\n" + "=" * 40)
    print("Step 3: Save to cleaned/")
    print("=" * 40)
    save_cleaned(cleaned)

    print("\nETL Completed")
    print(f"   MODE    : {MODE.upper()}")
    print(f"   Cleaning → {CLEANED_DIR}")
    print(f"   Lineage  → D:\\Desktop\\project\\lineage\\lineage.json")