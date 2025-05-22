import os
import requests
import csv
import glob

API_URL = "http://127.0.0.1:5000/predict"
INPUT_FOLDER = r"C:\Users\Student\OneDrive\Desktop\Data Science\From Model to Production\Batch_Images"
OUTPUT_CSV = r"C:\Users\Student\OneDrive\Desktop\Data Science\From Model to Production\Batch_Results\results.csv"

output_dir = os.path.dirname(OUTPUT_CSV)
os.makedirs(output_dir, exist_ok=True)

csv_header = ["filename", "predicted_class"]

results = []

for filepath in glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")):
    filename = os.path.basename(filepath)

    try:
        with open(filepath, "rb") as f:
            response = requests.post(API_URL, files={"file": f})
            data = response.json()

            predicted = data.get("class", "UNKNOWN")
            print(f"✔️ {filename}: {predicted}")

            results.append([filename, predicted])
    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")
        results.append([filename, "ERROR"])

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(results)

print(f"\n✅ Finished. {len(results)} files written to {OUTPUT_CSV}")
