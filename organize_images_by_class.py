import pandas as pd
import os
import shutil

# ──► 1.  Absolute paths (edit if you move folders) ◄──
CSV_FILE      = r"C:\Users\Student\OneDrive\Desktop\Data Science\From Model to Production\archive\myntradataset\styles.csv"
IMAGE_FOLDER  = r"C:\Users\Student\OneDrive\Desktop\Data Science\From Model to Production\archive\images"
OUTPUT_FOLDER = r"C:\Users\Student\OneDrive\Desktop\Data Science\From Model to Production\output"

# ──► 2.  Make sure output folder exists ◄──
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ──► 3.  Load CSV safely ◄──
df = pd.read_csv(CSV_FILE, on_bad_lines='skip')   # replaces deprecated error_bad_lines
df = df.dropna(subset=['id', 'articleType'])      # keep rows that have both columns

# ──► 4.  Counters for quick stats ◄──
copied, missing = 0, 0

# ──► 5.  Main loop ◄──
for _, row in df.iterrows():
    filename = f"{int(row['id'])}.jpg"            # id → str + .jpg
    category = row['articleType'].strip().replace('/', '_')

    src_path = os.path.join(IMAGE_FOLDER, filename)
    dst_dir  = os.path.join(OUTPUT_FOLDER, category)
    dst_path = os.path.join(dst_dir, filename)

    if not os.path.exists(src_path):
        missing += 1
        print(f"[missing] {filename} not found in images folder")
        continue                                   # skip to next row

    os.makedirs(dst_dir, exist_ok=True)

    try:
        shutil.copy(src_path, dst_path)
        copied += 1
    except Exception as e:
        print(f"[error]  Couldn’t copy {filename}: {e}")

# ──► 6.  Summary ◄──
print(f"\n✅ Finished!  Copied: {copied}   Missing: {missing}")
