"""Consolidates all 37,210 brain MRI scans across all cohorts in E:\\Brain tumor\\data
into a clean, unified 4-class folder structure:
  E:\\Brain tumor\\data_4class\\
  ├── glioma/
  ├── meningioma/
  ├── no_tumor/
  └── pituitary/

Uses high-speed NTFS hard-links (instant creation, 0 extra disk space, fallback to copy).
"""

import os
from pathlib import Path
import shutil
from tqdm import tqdm

DATA_DIR = Path(r"E:\Brain tumor\data")
TARGET_DIR = Path(r"E:\Brain tumor\data_4class")

CLASS_MAPPING = {
    "glioma": "glioma", "gl": "glioma",
    "meningioma": "meningioma", "me": "meningioma",
    "pituitary": "pituitary", "pi": "pituitary",
    "no_tumor": "no_tumor", "notumor": "no_tumor", "normal": "no_tumor", "healthy": "no_tumor",
}

def consolidate_dataset():
    print(f"Source Directory: {DATA_DIR.resolve()}")
    print(f"Target Directory: {TARGET_DIR.resolve()}")

    # 1. Create target directories for the 4 classes
    for cls in ["glioma", "meningioma", "no_tumor", "pituitary"]:
        (TARGET_DIR / cls).mkdir(parents=True, exist_ok=True)

    # 2. Gather all non-mask scans
    print("\n[1/3] Scanning source folders for MRI scans...")
    all_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for p in DATA_DIR.rglob(ext):
            parts_lower = [part.lower() for part in p.parts]
            # Skip masks
            if "mask" in p.name.lower() or "masks" in parts_lower:
                continue

            matched_class = None
            stem_lower = p.stem.lower()
            for k, v in CLASS_MAPPING.items():
                if any(k == part or k in part for part in parts_lower) or f"_{k}_" in stem_lower or stem_lower.startswith(f"{k}_"):
                    matched_class = v
                    break

            if matched_class:
                all_files.append((p, matched_class))

    print(f"Discovered {len(all_files):,} valid MRI scans across all subfolders.")

    # 3. Consolidate into 4 folders
    print("\n[2/3] Consolidating scans into unified 4-class directories...")
    existing_dest_names = {c: set(f.name for f in (TARGET_DIR / c).glob("*.*")) for c in ["glioma", "meningioma", "no_tumor", "pituitary"]}

    success_count = {c: 0 for c in ["glioma", "meningioma", "no_tumor", "pituitary"]}
    
    for src_p, cls_name in tqdm(all_files, desc="Consolidating", unit="files"):
        dest_folder = TARGET_DIR / cls_name
        dest_name = src_p.name

        # If duplicate name exists, prefix with cohort to avoid overwriting
        if dest_name in existing_dest_names[cls_name]:
            parts = src_p.parts
            data_idx = parts.index("data") if "data" in parts else 0
            cohort_prefix = parts[data_idx + 1] if data_idx + 1 < len(parts) else "sub"
            dest_name = f"{cohort_prefix}_{src_p.name}"

        dest_file = dest_folder / dest_name
        existing_dest_names[cls_name].add(dest_name)

        if not dest_file.exists():
            try:
                os.link(src_p, dest_file)
            except Exception:
                shutil.copy2(src_p, dest_file)
        success_count[cls_name] += 1

    # 4. Summary & Verification
    print("\n[3/3] Verification of Consolidated 4-Class Datafolder:")
    total_consolidated = 0
    for cls in ["glioma", "meningioma", "no_tumor", "pituitary"]:
        count = len(list((TARGET_DIR / cls).glob("*.*")))
        total_consolidated += count
        print(f"  📁 {TARGET_DIR.name}\\{cls:<12} : {count:,} images")

    print(f"\n=======================================================")
    print(f"  SUCCESSFULLY CONSOLIDATED: {total_consolidated:,} SCANS")
    print(f"  LOCATION: {TARGET_DIR.resolve()}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    consolidate_dataset()

