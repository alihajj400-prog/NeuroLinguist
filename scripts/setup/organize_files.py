import os
import shutil
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

RAW_PATH = Path(r"C:\FYP\Pitt\raw")

FOLDERS = [
    ("https://media.talkbank.org/dementia/English/Pitt/Control/cookie/", "Control", "cookie"),
    ("https://media.talkbank.org/dementia/English/Pitt/Control/fluency/", "Control", "fluency"),
    ("https://media.talkbank.org/dementia/English/Pitt/Control/recall/", "Control", "recall"),
    ("https://media.talkbank.org/dementia/English/Pitt/Control/sentence/", "Control", "sentence"),
    ("https://media.talkbank.org/dementia/English/Pitt/Dementia/cookie/", "Dementia", "cookie"),
    ("https://media.talkbank.org/dementia/English/Pitt/Dementia/fluency/", "Dementia", "fluency"),
    ("https://media.talkbank.org/dementia/English/Pitt/Dementia/recall/", "Dementia", "recall"),
    ("https://media.talkbank.org/dementia/English/Pitt/Dementia/sentence/", "Dementia", "sentence"),
]


def setup_driver():
    options = Options()
    options.add_experimental_option("detach", True)
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def get_clean_filename(filename):
    """Remove (1), (2), etc. from filename"""
    name = filename.replace(".mp3", "")
    for suffix in ["(9)", "(8)", "(7)", "(6)", "(5)", "(4)", "(3)", "(2)", "(1)"]:
        name = name.replace(f" {suffix}", "").replace(suffix, "")
    return name.strip() + ".mp3"


def main():
    print("=" * 50)
    print("ORGANIZE FILES BY CHECKING WEBSITE")
    print("=" * 50)
    
    # Create all target folders
    for _, group, task in FOLDERS:
        (RAW_PATH / group / task).mkdir(parents=True, exist_ok=True)
    
    # Get all downloaded MP3 files in raw folder
    downloaded_files = {}
    for f in RAW_PATH.iterdir():
        if f.is_file() and f.suffix.lower() == ".mp3":
            clean_name = get_clean_filename(f.name)
            if clean_name not in downloaded_files:
                downloaded_files[clean_name] = []
            downloaded_files[clean_name].append(f)
    
    print(f"\nFound {len(downloaded_files)} unique files ({sum(len(v) for v in downloaded_files.values())} total with duplicates)")
    
    driver = setup_driver()
    
    print("\n[STEP 1] Opening TalkBank...")
    print("         Please log in manually.")
    print("         After logging in, press ENTER here.\n")
    
    driver.get("https://media.talkbank.org/dementia/English/Pitt/Control/cookie/")
    input(">>> Press ENTER after you have logged in... ")
    
    print("\n[STEP 2] Scanning folders and organizing files...")
    print("=" * 50)
    
    moved = 0
    not_found = 0
    
    for url, group, task in FOLDERS:
        print(f"\n[FOLDER] {group}/{task}")
        print("-" * 40)
        
        driver.get(url)
        time.sleep(2)
        
        try:
            # Get all MP3 filenames from website
            links = driver.find_elements(By.XPATH, "//a[contains(@href, '.mp3')]")
            website_files = set()
            for link in links:
                href = link.get_attribute("href")
                filename = href.split("/")[-1]
                website_files.add(filename)
            
            print(f"    Website has {len(website_files)} files")
            
            # Match and move files
            target_folder = RAW_PATH / group / task
            
            for web_filename in website_files:
                clean_name = get_clean_filename(web_filename)
                
                if clean_name in downloaded_files and len(downloaded_files[clean_name]) > 0:
                    # Take one file from the list
                    source_file = downloaded_files[clean_name].pop(0)
                    target_path = target_folder / clean_name
                    
                    if not target_path.exists():
                        shutil.move(str(source_file), str(target_path))
                        print(f"    [OK] {clean_name}")
                        moved += 1
                    else:
                        # Already exists, delete duplicate
                        source_file.unlink()
                        print(f"    [SKIP] {clean_name} (already exists)")
                else:
                    print(f"    [NOT FOUND] {clean_name}")
                    not_found += 1
                    
        except Exception as e:
            print(f"    [ERROR] {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("ORGANIZATION COMPLETE")
    print("=" * 50)
    print(f"✅ Moved: {moved}")
    print(f"⚠  Not found: {not_found}")
    
    # Count remaining files in raw
    remaining = len([f for f in RAW_PATH.iterdir() if f.is_file() and f.suffix.lower() == ".mp3"])
    print(f"📁 Remaining in raw/: {remaining}")
    
    # Count files in each folder
    print("\n📁 Files per folder:")
    for _, group, task in FOLDERS:
        folder = RAW_PATH / group / task
        count = len(list(folder.glob("*.mp3")))
        print(f"   {group}/{task}: {count} files")
    
    print("\n>>> Browser will stay open. Close it manually when done.")


if __name__ == "__main__":
    main()