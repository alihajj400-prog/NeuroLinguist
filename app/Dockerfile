FROM python:3.11-slim

WORKDIR /app

# System dependencies: ffmpeg for audio decoding, libsndfile for parselmouth.
# Praat is bundled inside parselmouth itself so no separate apt install needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy 3.7's `spacy download` command points to a GitHub release URL that 404s.
# Install the model wheel directly to avoid that broken download path.
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Pre-download the Sentence-BERT model into the image so first inference doesn't
# need to hit huggingface from the running container.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
