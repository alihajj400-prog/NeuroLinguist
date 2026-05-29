"""
NeuroLinguist - Alzheimer's Detection Web Application
Flask backend for multi-modal speech analysis

Feature extraction mirrors the training pipeline (extract_linguistic_features.py,
extract_semantic_features.py, extract_acoustic_features_fast.py, transcript
preprocessing). Any deviation here produces predictions inconsistent with the
trained models, so this file is structured to match those scripts line-by-line.
"""

import os
import re
import json
import math
import tempfile
import numpy as np
import joblib
import spacy
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
# Use the OS's actual temp directory (works on Windows, Linux, macOS).
# /tmp/uploads was Linux-specific and silently broke on Windows.
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'neurolinguist_uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================
# LOAD MODELS AND CONFIG
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

with open(os.path.join(MODEL_DIR, 'config.json'), 'r') as f:
    config = json.load(f)

cookie_models = {}
noncookie_models = {}
for name in config['model_names']:
    cookie_models[name] = joblib.load(os.path.join(MODEL_DIR, f'cookie_{name}.joblib'))
    noncookie_models[name] = joblib.load(os.path.join(MODEL_DIR, f'noncookie_{name}.joblib'))

cookie_scaler = joblib.load(os.path.join(MODEL_DIR, 'cookie_scaler.joblib'))
noncookie_scaler = joblib.load(os.path.join(MODEL_DIR, 'noncookie_scaler.joblib'))

# Build per-feature training-distribution stats from scaler.mean_ and scaler.scale_.
# These are used for the analysis page (z-scoring user values, percentile bars, and
# population-distribution shading). Stored as {feature_name: (mean, std)}.
def _build_feature_stats():
    stats = {}
    for i, f in enumerate(config['cookie_numeric']):
        stats[f] = (float(cookie_scaler.mean_[i]), float(cookie_scaler.scale_[i]))
    # Non-cookie scaler covers some features the cookie one doesn't have a direct
    # entry for, but for shared features cookie stats are fine — they are derived
    # from a superset of samples. We only fall back to noncookie for features
    # absent from the cookie numeric list, which doesn't happen in practice.
    for i, f in enumerate(config['noncookie_numeric']):
        if f not in stats:
            stats[f] = (float(noncookie_scaler.mean_[i]), float(noncookie_scaler.scale_[i]))
    return stats

FEATURE_STATS = _build_feature_stats()

# spaCy
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')
nlp.max_length = 2_000_000

# Sentence-BERT (lazy: imported on first call to avoid slowing startup if unused)
_sbert_model = None
def get_sbert():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sbert_model

print("All models loaded successfully!")


# ============================================================
# WORD LISTS (verbatim from extract_linguistic_features.py)
# ============================================================
FILLERS = {'uh', 'um', 'er', 'ah', 'mm', 'hm', 'hmm', 'mhm', 'uhuh', 'uhhuh', 'oh', 'eh', 'huh'}

FUNCTION_WORDS = {
    'a', 'an', 'the', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its',
    'our', 'their', 'some', 'any', 'no', 'every', 'each', 'all', 'both', 'half', 'either',
    'neither', 'much', 'many', 'more', 'most', 'few', 'fewer', 'little', 'less', 'least',
    'i', 'me', 'mine', 'myself', 'you', 'yours', 'yourself', 'he', 'him', 'himself',
    'she', 'hers', 'herself', 'it', 'itself', 'we', 'us', 'ours', 'ourselves',
    'they', 'them', 'theirs', 'themselves', 'who', 'whom', 'whose', 'which', 'what',
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'up', 'down', 'into', 'out',
    'over', 'under', 'about', 'through', 'between', 'after', 'before', 'above', 'below',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'because', 'although', 'though', 'while',
    'if', 'unless', 'until', 'when', 'where', 'whether', 'as', 'than',
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'must',
    'can', 'could', 'not', "n't", 'there', 'here',
}

HIGH_FREQ_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'can', 'could', 'it', 'this', 'that', 'and',
    'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'i', 'you', 'he', 'she',
    'they', 'we', 'there', 'here', 'what', 'who', 'how', 'when', 'where', 'why',
    'just', 'like', 'know', 'think', 'go', 'get', 'make', 'see', 'want', 'come',
}

COOKIE_THEFT_UNITS = {
    'boy': ['boy', 'son', 'brother', 'child', 'kid', 'little boy', 'young boy', 'he'],
    'girl': ['girl', 'daughter', 'sister', 'child', 'kid', 'little girl', 'young girl', 'she'],
    'woman': ['woman', 'mother', 'mom', 'mommy', 'lady', 'wife', 'her', 'she'],
    'cookie': ['cookie', 'cookies', 'treat', 'treats', 'biscuit'],
    'jar': ['jar', 'cookie jar', 'container', 'lid'],
    'stool': ['stool', 'step stool', 'ladder', 'stepping', 'chair', 'standing on'],
    'sink': ['sink', 'basin', 'kitchen sink'],
    'water': ['water', 'faucet', 'tap', 'running water'],
    'dishes': ['dish', 'dishes', 'plate', 'plates', 'cup', 'cups', 'drying'],
    'window': ['window', 'curtain', 'curtains', 'outside'],
    'cupboard': ['cupboard', 'cabinet', 'counter', 'countertop'],
    'floor': ['floor', 'ground', 'puddle'],
    'stealing': ['steal', 'stealing', 'take', 'taking', 'get', 'getting', 'reach', 'reaching', 'hand'],
    'falling': ['fall', 'falling', 'tip', 'tipping', 'topple', 'toppling', 'off balance', 'gonna fall'],
    'washing': ['wash', 'washing', 'dry', 'drying', 'clean', 'cleaning', 'wipe', 'wiping'],
    'overflowing': ['overflow', 'overflowing', 'spill', 'spilling', 'flood', 'flooding', 'running over', 'dripping'],
    'daydreaming': ['daydream', 'not paying attention', 'looking out', 'distracted', 'ignoring'],
    'kitchen': ['kitchen'],
    'outside': ['outside', 'outdoors', 'yard', 'garden', 'backyard', 'tree', 'path', 'walkway'],
}


# ============================================================
# LINGUISTIC FEATURES (mirrors extract_linguistic_features.py)
# ============================================================

def compute_word_frequency_index(tokens):
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t.lower() in HIGH_FREQ_WORDS) / len(tokens)


def compute_pronoun_noun_ratio(doc):
    pronouns = sum(1 for t in doc if t.pos_ == 'PRON')
    nouns = sum(1 for t in doc if t.pos_ in {'NOUN', 'PROPN'})
    if nouns == 0:
        return float(pronouns) if pronouns > 0 else 0.0
    return pronouns / nouns


def compute_content_function_ratio(doc):
    content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
    content_count = function_count = 0
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        word = token.text.lower()
        if word in FILLERS:
            continue
        if token.pos_ in content_pos and word not in FUNCTION_WORDS:
            content_count += 1
        elif word in FUNCTION_WORDS or token.pos_ in {'DET', 'PRON', 'ADP', 'CCONJ', 'SCONJ', 'AUX', 'PART'}:
            function_count += 1
    if function_count == 0:
        return float(content_count) if content_count > 0 else 0.0
    return content_count / function_count


def compute_repetition_rate(tokens):
    if len(tokens) < 2:
        return 0.0
    repetitions = 0
    for i in range(1, len(tokens)):
        window_start = max(0, i - 3)
        if tokens[i].lower() in [t.lower() for t in tokens[window_start:i]]:
            repetitions += 1
    return repetitions / len(tokens)


def compute_syntactic_complexity(doc):
    sentences = list(doc.sents)
    if not sentences:
        return 0.0
    def get_depth(token, depth=0):
        if token.head == token or depth > 50:
            return depth
        return get_depth(token.head, depth + 1)
    depths = []
    for sent in sentences:
        max_depth = 0
        for token in sent:
            if not token.is_punct and not token.is_space:
                d = get_depth(token)
                max_depth = max(max_depth, d)
        if max_depth > 0:
            depths.append(max_depth)
    return sum(depths) / len(depths) if depths else 0.0



def extract_linguistic_features(text):
    if not text or not text.strip():
        return {
            'syntactic_complexity': 0.0, 'pronoun_noun_ratio': 0.0,
            'repetition_rate': 0.0, 'word_frequency_index': 0.0,
            'content_function_ratio': 0.0, 'word_count': 0,
        }

    doc = nlp(text)

    content_tokens = []
    for token in doc:
        if not token.is_punct and not token.is_space:
            word = token.text.lower().strip('.,!?')
            if word and word not in FILLERS:
                content_tokens.append(word)

    if not content_tokens:
        return {
            'syntactic_complexity': 0.0, 'pronoun_noun_ratio': 0.0,
            'repetition_rate': 0.0, 'word_frequency_index': 0.0,
            'content_function_ratio': 0.0, 'word_count': 0,
        }

    return {
        'syntactic_complexity': round(compute_syntactic_complexity(doc), 2),
        'pronoun_noun_ratio': round(compute_pronoun_noun_ratio(doc), 4),
        'repetition_rate': round(compute_repetition_rate(content_tokens), 4),
        'word_frequency_index': round(compute_word_frequency_index(content_tokens), 4),
        'content_function_ratio': round(compute_content_function_ratio(doc), 4),
        'word_count': len(content_tokens),
    }


# ============================================================
# SEMANTIC FEATURES (mirrors extract_semantic_features.py)
# ============================================================

def split_into_sentences(text):
    if not text:
        return []
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip().split()) >= 3]


def count_information_units(text):
    if not text:
        return 0.0
    text_lower = text.lower()
    units_found = 0
    for keywords in COOKIE_THEFT_UNITS.values():
        for kw in keywords:
            if kw in text_lower:
                units_found += 1
                break
    return units_found / len(COOKIE_THEFT_UNITS)


def compute_semantic_coherence(text):
    sentences = split_into_sentences(text)
    if len(sentences) < 2:
        return 0.0
    try:
        emb = get_sbert().encode(sentences, show_progress_bar=False)
        sims = []
        for i in range(len(emb) - 1):
            ni, nj = np.linalg.norm(emb[i]), np.linalg.norm(emb[i+1])
            if ni > 0 and nj > 0:
                sims.append(float(np.dot(emb[i], emb[i+1]) / (ni * nj)))
        return float(np.mean(sims)) if sims else 0.0
    except Exception:
        return 0.0


def compute_global_semantic_drift(text):
    sentences = split_into_sentences(text)
    if len(sentences) < 3:
        return 0.0
    try:
        emb = get_sbert().encode(sentences, show_progress_bar=False)
        centroid = np.mean(emb, axis=0)
        cn = np.linalg.norm(centroid)
        if cn == 0:
            return 0.0
        distances = []
        for v in emb:
            vn = np.linalg.norm(v)
            if vn > 0:
                distances.append(1 - float(np.dot(v, centroid) / (vn * cn)))
        return float(np.std(distances)) if distances else 0.0
    except Exception:
        return 0.0


def compute_story_recall_similarity(text, task_type):
    if 'recall' in task_type.lower():
        return compute_semantic_coherence(text)
    return 0.0


def extract_semantic_features(text, task_type):
    if not text or not text.strip():
        return {
            'semantic_coherence': 0.0, 'information_unit_coverage': 0.0,
            'story_recall_similarity': 0.0, 'global_semantic_drift': 0.0,
        }
    iuc = count_information_units(text) if 'cookie' in task_type.lower() else 0.0
    return {
        'semantic_coherence': round(compute_semantic_coherence(text), 4),
        'information_unit_coverage': round(iuc, 4),
        'story_recall_similarity': round(compute_story_recall_similarity(text, task_type), 4),
        'global_semantic_drift': round(compute_global_semantic_drift(text), 4),
    }


# ============================================================
# BIOMARKERS (counts of disfluency markers in raw transcript)
# ============================================================

def extract_biomarker_features(text, word_count):
    if not text or word_count == 0:
        return {k: 0.0 for k in [
            'filler_rate', 'unfilled_pause_rate', 'repetition_rate_bio',
            'revision_rate', 'fragment_rate', 'utterance_rate', 'words_per_utterance',
        ]}

    tokens = text.split()
    n = max(len(tokens), 1)

    filler_count = sum(1 for t in tokens if t.lower().strip('.,!?') in FILLERS)
    repetition_count = sum(
        1 for i in range(1, len(tokens))
        if tokens[i].lower().strip('.,!?') == tokens[i-1].lower().strip('.,!?')
    )
    fragment_count = sum(1 for t in tokens if t.endswith('-') or t.endswith("'"))
    revision_markers = {'well', 'no', 'wait', 'sorry', 'actually'}
    revision_count = sum(1 for t in tokens if t.lower().strip('.,!?') in revision_markers)
    unfilled_pause_count = text.count('...') + text.count('..') + text.count('. .')

    utterances = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    utterance_count = max(len(utterances), 1)

    return {
        'filler_rate': filler_count / word_count,
        'unfilled_pause_rate': unfilled_pause_count / word_count,
        'repetition_rate_bio': repetition_count / word_count,
        'revision_rate': revision_count / word_count,
        'fragment_rate': fragment_count / word_count,
        'utterance_rate': utterance_count / word_count,
        'words_per_utterance': word_count / utterance_count,
    }


# ============================================================
# ACOUSTIC FEATURES (mirrors extract_acoustic_features_fast.py:
# pitch_floor=50, pitch_ceiling=300, intensity-based pause detection)
# ============================================================

def _default_acoustic():
    """Population-mean defaults used when no audio is provided."""
    return {
        'mean_pause_duration': 0.5,
        'speech_time_ratio': 0.8,
        'pitch_range': 150.0,
        'pitch_variability': 40.0,
        'intensity_range': 50.0,
        'jitter': 0.02,
        'shimmer': 0.12,
    }


def extract_acoustic_features_from_audio(filepath):
    try:
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(filepath)
        duration = snd.get_total_duration()
        if duration < 0.5:
            return _default_acoustic()

        pitch = snd.to_pitch(time_step=0.01, pitch_floor=50, pitch_ceiling=300)
        pv = pitch.selected_array['frequency']
        pv = pv[pv > 0]
        pitch_range = float(np.max(pv) - np.min(pv)) if len(pv) else 0.0
        pitch_variability = float(np.std(pv)) if len(pv) else 0.0

        intensity = snd.to_intensity(minimum_pitch=50)
        iv = intensity.values[0]
        iv = iv[~np.isnan(iv)]
        intensity_range = float(np.max(iv) - np.min(iv)) if len(iv) else 0.0

        if len(iv) > 0:
            thresh = np.percentile(iv, 25)
            time_step = intensity.time_step
            is_pause = iv < thresh
            starts = np.where(np.diff(is_pause.astype(int)) == 1)[0]
            ends = np.where(np.diff(is_pause.astype(int)) == -1)[0]
            if is_pause[0]:
                starts = np.insert(starts, 0, 0)
            if is_pause[-1]:
                ends = np.append(ends, len(is_pause) - 1)
            m = min(len(starts), len(ends))
            if m > 0:
                durs = (ends[:m] - starts[:m]) * time_step
                durs = durs[durs > 0.1]
                mean_pause_duration = float(np.mean(durs)) if len(durs) else 0.0
                speech_time_ratio = max(0.0, min(1.0, 1 - float(np.sum(durs)) / duration))
            else:
                mean_pause_duration = 0.0
                speech_time_ratio = 1.0
        else:
            mean_pause_duration = 0.0
            speech_time_ratio = 1.0

        try:
            pp = call(snd, "To PointProcess (periodic, cc)", 50, 300)
            jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            jitter = float(jitter) if not np.isnan(jitter) else 0.0
            shimmer = float(shimmer) if not np.isnan(shimmer) else 0.0
        except Exception:
            jitter = 0.0
            shimmer = 0.0

        return {
            'mean_pause_duration': round(mean_pause_duration, 4),
            'speech_time_ratio': round(speech_time_ratio, 4),
            'pitch_range': round(pitch_range, 2),
            'pitch_variability': round(pitch_variability, 2),
            'intensity_range': round(intensity_range, 2),
            'jitter': round(jitter, 6),
            'shimmer': round(shimmer, 6),
        }
    except Exception as e:
        print(f"Acoustic extraction error: {e}")
        return _default_acoustic()


# ============================================================
# TASK NORMALIZATION + PREDICTION
# ============================================================

def compute_task_z_scores(features, task_type):
    """Apply training-time per-task standardization. Means and stds come from
    config.json which 10_export_models.py serialised at training time."""
    task_stats = config['task_stats']
    for feat in ['information_unit_coverage', 'story_recall_similarity',
                 'syntactic_complexity', 'semantic_coherence']:
        z_key = f'{feat}_task_z'
        if feat in task_stats and task_type in task_stats[feat]:
            mean = task_stats[feat][task_type]['mean']
            std = task_stats[feat][task_type]['std']
            if std > 1e-6 and feat in features:
                features[z_key] = (features[feat] - mean) / std
            else:
                features[z_key] = 0.0
        else:
            features[z_key] = 0.0
    return features


def predict(features, task_type, acoustic_measured=True):
    if task_type == 'cookie':
        feature_list = config['cookie_features']
        numeric_list = config['cookie_numeric']
        models = cookie_models
        scaler = cookie_scaler
    else:
        feature_list = config['noncookie_features']
        numeric_list = config['noncookie_numeric']
        models = noncookie_models
        scaler = noncookie_scaler
        for tc in config['task_cols']:
            task_name = tc.replace('task_', '')
            features[tc] = 1 if task_name == task_type else 0

    X = np.array([[features.get(f, 0.0) for f in feature_list]], dtype=float)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(numeric_list)
    X[:, :n] = scaler.transform(X[:, :n])

    probs = []
    model_probs = {}
    for name, model in models.items():
        p = float(model.predict_proba(X)[0, 1])
        probs.append(p)
        model_probs[name] = p

    avg_prob = float(np.mean(probs))
    prediction = 'Dementia' if avg_prob >= 0.5 else 'Control'

    # Build enriched feature list. Each entry includes the raw value plus a
    # 0–100 percentile (where the value falls in the training distribution),
    # a z-score, and a flag for skewed/clamped features. The percentile is the
    # right primary signal for the bar widths because it's monotonic in the
    # raw value but bounded — so pitch_range=150 and repetition_rate=0.05
    # become directly comparable on a shared 0–100 axis.
    def _norm_entry(name, raw_value):
        mean, std = FEATURE_STATS.get(name, (0.0, 1.0))
        if std < 1e-9:
            z = 0.0
        else:
            z = (raw_value - mean) / std
        # Clamp at ±4 SD for display sanity (population mass beyond is negligible
        # and would otherwise produce 100% bars on outliers).
        z_clamped = max(-4.0, min(4.0, z))
        # Convert z to percentile via standard normal CDF (math.erf).
        percentile = 0.5 * (1.0 + math.erf(z_clamped / math.sqrt(2.0))) * 100.0
        return {
            'z_score': round(z, 3),
            'percentile': round(percentile, 1),
            'mean': round(mean, 4),
            'std': round(std, 4),
        }

    ACOUSTIC_FEATS = {
        'mean_pause_duration', 'speech_time_ratio', 'pitch_range',
        'pitch_variability', 'intensity_range', 'jitter', 'shimmer',
    }

    top_features = []
    for f in feature_list[:24]:
        # Skip acoustic features entirely when no audio was uploaded — their
        # values are population defaults and would mislead the user.
        if not acoustic_measured and f in ACOUSTIC_FEATS:
            continue
        v = features.get(f, 0.0)
        if abs(v) > 0.001:
            entry = {'name': f, 'value': round(float(v), 4)}
            entry.update(_norm_entry(f, float(v)))
            top_features.append(entry)
    # Sort by absolute z-score so the most distinctive features rise to the top
    top_features.sort(key=lambda e: abs(e['z_score']), reverse=True)

    return {
        'prediction': prediction,
        'confidence': round(avg_prob * 100 if prediction == 'Dementia' else (1 - avg_prob) * 100, 1),
        'dementia_probability': round(avg_prob * 100, 1),
        'model_probabilities': model_probs,
        'top_features': top_features[:10],
        'all_features_norm': top_features,
        'task_type': task_type,
    }


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        transcript = request.form.get('transcript', '').strip()
        task_type = request.form.get('task_type', '').strip().lower()
        audio_file = request.files.get('audio')

        if not transcript:
            return jsonify({'error': 'Please provide a transcript.'}), 400
        if task_type not in {'cookie', 'fluency', 'recall', 'sentence'}:
            return jsonify({'error': 'Please select a cognitive task before analysis.'}), 400

        features = {}
        ling = extract_linguistic_features(transcript)
        features.update(ling)

        sem = extract_semantic_features(transcript, task_type)
        features.update(sem)

        bio = extract_biomarker_features(transcript, ling['word_count'])
        features.update(bio)

        if audio_file and audio_file.filename:
            filename = secure_filename(audio_file.filename) or 'upload.wav'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(filepath)
            saved_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            print(f"[acoustic] saved upload: {filepath}  ({saved_size} bytes)")
            acou = extract_acoustic_features_from_audio(filepath)
            # If extraction returned defaults, the helper already printed the
            # exception. Mark acoustic_measured accordingly so the UI is honest.
            defaults = _default_acoustic()
            acoustic_measured = any(
                abs(acou.get(k, 0.0) - defaults[k]) > 1e-9 for k in defaults
            )
            print(f"[acoustic] measured={acoustic_measured}  values={acou}")
            try:
                os.remove(filepath)
            except OSError:
                pass
        else:
            acou = _default_acoustic()
            acoustic_measured = False
        features.update(acou)

        features = compute_task_z_scores(features, task_type)

        result = predict(features, task_type, acoustic_measured=acoustic_measured)
        result['acoustic_measured'] = acoustic_measured
        result['features_extracted'] = {
            'linguistic': ling,
            'semantic': sem,
            'biomarkers': bio,
            'acoustic': acou,
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analysis')
def analysis_page():
    return render_template('analysis.html')


@app.route('/feature-stats')
def feature_stats_endpoint():
    """Expose training-distribution stats for the analysis page."""
    return jsonify({
        'stats': {k: {'mean': v[0], 'std': v[1]} for k, v in FEATURE_STATS.items()},
        'task_stats': config.get('task_stats', {}),
        'cookie_features': config['cookie_features'],
        'noncookie_features': config['noncookie_features'],
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'models_loaded': True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
