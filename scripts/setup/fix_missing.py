import os
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

BASE_TARGET = Path(r"C:\FYP\Pitt\raw")

FOLDERS = [
    ("https://media.talkbank.org/dementia/English/Pitt/Dementia/recall/", "Dementia", "recall"),
    ("https://media.talkbank.org/dementia/English/Pitt/Dementia/sentence/", "Dementia", "sentence"),
]

BATCH_SIZE = 90
PAUSE_SECONDS = 120  # 2 minutes


def create_driver_for_folder(download_path):
    options = Options()
    
    prefs = {
        "download.default_directory": str(download_path),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    options.add_experimental_option("prefs", prefs)
    options.add_experimental_option("detach", True)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    return driver


def main():
    print("=" * 50)
    print("FIX: Download Missing Files (with rate limit pause)")
    print("=" * 50)
    
    first_url, first_group, first_task = FOLDERS[0]
    first_download_path = BASE_TARGET / first_group / first_task
    
    driver = create_driver_for_folder(first_download_path)
    
    print("\n[STEP 1] Opening TalkBank...")
    print("         Please log in manually.")
    print("         After logging in, press ENTER here.\n")
    
    driver.get(first_url)
    input(">>> Press ENTER after you have logged in... ")
    
    total_downloaded = 0
    batch_count = 0
    
    for url, group, task in FOLDERS:
        download_path = BASE_TARGET / group / task
        
        print(f"\n[FOLDER] {group}/{task}")
        print("-" * 40)
        
        driver.execute_cdp_cmd("Page.setDownloadBehavior", {
            "behavior": "allow",
            "downloadPath": str(download_path)
        })
        
        # Get existing files
        existing_files = set(f.name for f in download_path.glob("*.mp3") if f.stat().st_size > 10000)
        print(f"    Already have: {len(existing_files)} files")
        
        driver.get(url)
        time.sleep(2)
        
        try:
            links = driver.find_elements(By.XPATH, "//a[contains(@href, '.mp3')]")
            mp3_urls = [link.get_attribute("href") for link in links]
            
            # Filter to only missing
            missing_urls = [u for u in mp3_urls if u.split("/")[-1] not in existing_files]
            
            print(f"    Website has: {len(mp3_urls)} files")
            print(f"    Missing: {len(missing_urls)} files")
            print("-" * 40)
            
            for i, mp3_url in enumerate(missing_urls):
                filename = mp3_url.split("/")[-1]
                
                # Pause every 90 downloads
                if batch_count > 0 and batch_count % BATCH_SIZE == 0:
                    print(f"\n    ⏸ PAUSING {PAUSE_SECONDS}s to avoid rate limit...")
                    time.sleep(PAUSE_SECONDS)
                    driver.refresh()
                    time.sleep(2)
                    print("    ▶ Resuming...\n")
                
                print(f"    [DOWNLOADING] {filename}...", end=" ", flush=True)
                
                driver.execute_script(f"""
                    var a = document.createElement('a');
                    a.href = '{mp3_url}';
                    a.download = '{filename}';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                """)
                
                time.sleep(0.5)
                total_downloaded += 1
                batch_count += 1
                print("OK")
                
        except Exception as e:
            print(f"    [ERROR] {e}")
    
    print("\n" + "=" * 50)
    print(f"DONE! Initiated {total_downloaded} downloads")
    print("=" * 50)
    print("\n>>> Browser will stay open.")
    print(">>> Wait for downloads to complete before closing!")


if __name__ == "__main__":
    main()