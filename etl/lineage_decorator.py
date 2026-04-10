import functools
import json
import os
from datetime import datetime

LINEAGE_OUTPUT_PATH = r"D:\Desktop\DAMG 7374\healthcare_lineagetracking\lineage\lineage.json"

TRANSFORM_TYPE_MAP = {
    "fill_missing":        "IMPUTATION",
    "fill_missing_median": "IMPUTATION",
    "replace_unknown":     "STANDARDIZATION",
    "concat_version_suffix": "STANDARDIZATION",
}

def _load_lineage():
    if os.path.exists(LINEAGE_OUTPUT_PATH):
        with open(LINEAGE_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("lineage_records", [])
    return []

def _save_lineage(records):
    os.makedirs(os.path.dirname(LINEAGE_OUTPUT_PATH), exist_ok=True)
    with open(LINEAGE_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"lineage_records": records}, f, ensure_ascii=False, indent=2)

def capture_lineage(sources, target, transformation, description="", script_ref="transform.py"):
    """
    Decorator: Automatic capture lineage metadata of the transfromation process

    Arguments:
        sources        : list, imput field, format 'table.field'
        target         : str, output field, format 'table.field'
        transformation : str, transformation type name
        description    : str, conversion Logic Explanation
        script_ref     : str, corresponding ETL script filename
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = datetime.now()

            # Record the number of input lines
            df_in = args[0] if args else None
            records_before = len(df_in) if df_in is not None else 0

            result = func(*args, **kwargs)

            elapsed = (datetime.now() - start).total_seconds() * 1000
            records_after = len(result) if result is not None else 0

            # Generate transformation_id
            existing = _load_lineage()
            t_id = f"t_{str(len(existing) + 1).zfill(3)}"

            # source/target fields convert into f_{table}_{field} 
            def to_field_id(field_str):
                parts = field_str.split(".")
                if len(parts) == 2:
                    return f"f_{parts[0]}_{parts[1]}"
                return f"f_{field_str}"

            source_field_ids = [to_field_id(s) for s in sources]
            target_field_id  = to_field_id(target)

            transform_type = TRANSFORM_TYPE_MAP.get(transformation, "DERIVATION")

            record = {
                "transformation_id":  t_id,
                "transform_type":     transform_type,
                "source_fields":      source_field_ids,
                "target_fields":      [target_field_id],
                "logic":              description,
                "script_ref":         script_ref,
                "timestamp":          start.strftime("%Y-%m-%d %H:%M:%S"),
                "records_affected":   records_before,
                "records_rejected":   max(0, records_before - records_after)
            }

            existing.append(record)
            _save_lineage(existing)

            print(f"[Lineage] {t_id} {func.__name__} → {target_field_id} ({elapsed:.1f}ms)")
            return result
        return wrapper
    return decorator