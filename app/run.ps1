cd D:\FYP\app\neurolinguist-app
Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install --no-cache-dir -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
python app.py
