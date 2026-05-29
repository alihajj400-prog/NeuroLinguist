import os

# Base path
base_path = r"C:\FYP\Pitt\processed"

# Groups
groups = ["Control", "Dementia"]

# Tasks
tasks = ["cookie", "fluency", "recall", "sentence"]

# Create all folder combinations
for group in groups:
    for task in tasks:
        folder_path = os.path.join(base_path, group, task)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created: {folder_path}")

print("\n✓ All folders created!")