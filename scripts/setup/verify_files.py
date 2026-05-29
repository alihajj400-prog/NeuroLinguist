import os
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


def main():
    print("=" * 60)
    print("FILE COUNT VERIFICATION")
    print("=" * 60)
    
    driver = setup_driver()
    
    print("\n[STEP 1] Opening TalkBank...")
    print("         Please log in manually.")
    print("         After logging in, press ENTER here.\n")
    
    driver.get("https://media.talkbank.org/dementia/English/Pitt/Control/cookie/")
    input(">>> Press ENTER after you have logged in... ")
    
    print("\n[STEP 2] Comparing website vs local files...")
    print("=" * 60)
    print(f"{'Folder':<25} {'Website':<10} {'Local':<10} {'Status':<15}")
    print("-" * 60)
    
    total_website = 0
    total_local = 0
    missing_files = {}
    
    for url, group, task in FOLDERS:
        folder_name = f"{group}/{task}"
        local_folder = RAW_PATH / group / task
        
        # Get website count
        driver.get(url)
        time.sleep(2)
        
        try:
            links = driver.find_elements(By.XPATH, "//a[contains(@href, '.mp3')]")
            website_files = set()
            for link in links:
                href = link.get_attribute("href")
                filename = href.split("/")[-1]
                website_files.add(filename)
            website_count = len(website_files)
        except:
            website_count = 0
            website_files = set()
        
        # Get local count
        if local_folder.exists():
            local_files = set(f.name for f in local_folder.glob("*.mp3"))
            local_count = len(local_files)
        else:
            local_files = set()
            local_count = 0
        
        # Compare
        total_website += website_count
        total_local += local_count
        
        if website_count == local_count:
            status = "✅ OK"
        elif local_count > website_count:
            status = "⚠️  EXTRA"
        else:
            status = f"❌ MISSING {website_count - local_count}"
            # Track missing files
            missing = website_files - local_files
            if missing:
                missing_files[folder_name] = missing
        
        print(f"{folder_name:<25} {website_count:<10} {local_count:<10} {status:<15}")
    
    # Also count files in raw/ root (unorganized)
    raw_root_files = [f for f in RAW_PATH.iterdir() if f.is_file() and f.suffix.lower() == ".mp3"]
    
    print("-" * 60)
    print(f"{'TOTAL':<25} {total_website:<10} {total_local:<10}")
    print(f"\n📁 Unorganized files in raw/: {len(raw_root_files)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if total_website == total_local and len(raw_root_files) == 0:
        print("✅ All files accounted for and organized!")
    else:
        if len(raw_root_files) > 0:
            print(f"⚠️  {len(raw_root_files)} files still need to be organized")
        if total_local < total_website:
            print(f"❌ Missing {total_website - total_local} files total")
        if total_local > total_website:
            print(f"⚠️  {total_local - total_website} extra files (possible duplicates)")
    
    # Show missing files if any
    if missing_files:
        print("\n" + "=" * 60)
        print("MISSING FILES BY FOLDER")
        print("=" * 60)
        for folder, files in missing_files.items():
            print(f"\n{folder}:")
            for f in sorted(files)[:10]:  # Show first 10
                print(f"   - {f}")
            if len(files) > 10:
                print(f"   ... and {len(files) - 10} more")
    
    print("\n>>> Browser will stay open. Close it manually when done.")


if __name__ == "__main__":
    main()