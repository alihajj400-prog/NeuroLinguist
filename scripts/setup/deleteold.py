import os
import shutil

# Folders to delete
folders_to_clean = [
    r"C:\FYP\Pitt\raw",
    r"C:\FYP\PittAudio",
]

print("=" * 50)
print("DELETING OLD CORRUPTED FILES")
print("=" * 50)

for folder in folders_to_clean:
    if os.path.exists(folder):
        # Count files before deletion
        file_count = sum(len(files) for _, _, files in os.walk(folder))
        
        # Delete the folder and all contents
        shutil.rmtree(folder)
        
        # Recreate empty folder
        os.makedirs(folder, exist_ok=True)
        
        print(f"✓ Deleted {file_count} files from: {folder}")
    else:
        print(f"⚠ Folder not found: {folder}")

# Recreate the folder structure for raw
raw_groups = ["Control", "Dementia"]
tasks = ["cookie", "fluency", "recall", "sentence"]

for group in raw_groups:
    for task in tasks:
        path = os.path.join(r"C:\FYP\Pitt\raw", group, task)
        os.makedirs(path, exist_ok=True)

print("\n✓ Recreated empty folder structure")
print("=" * 50)
print("DONE - Ready for fresh download")
print("=" * 50)