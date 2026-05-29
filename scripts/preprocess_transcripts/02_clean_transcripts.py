
"""
================================================================================
STEP 2: Clean CHAT Annotations + Count Biomarkers
================================================================================

Reads transcripts_raw.csv, cleans CHAT conventions, counts biomarkers.
Outputs: transcripts_clean.csv

CRITICAL BIOMARKERS FOR AD DETECTION (counted BEFORE cleaning):
1. Filled pauses: &-uh, &-um, &-hm, &-mm, &-er, &-ah (hesitations)
2. Unfilled pauses: (.), (..), (...) (silent pauses in transcript)
3. Repetitions: [/] (same word/phrase repeated)
4. Revisions: [//], [///] (self-corrections)
5. Fragments: &+word (abandoned word attempts)

IMPORTANT: Filled pauses are CONVERTED to words (uh, um) and PRESERVED
because they are diagnostic biomarkers for Alzheimer's detection.

Usage:
    python 02_clean_transcripts.py -i "C:/FYP/data/transcripts/transcripts_raw.csv" -o "C:/FYP/data/transcripts"
"""

import os
import re
import csv
import argparse


def count_biomarkers(raw_text):
    """
    Count speech biomarkers in raw CHAT text BEFORE cleaning.
    These are critical diagnostic indicators for Alzheimer's detection.
    
    Returns dict with counts and also returns list of detected fillers for verification.
    """
    biomarkers = {
        'filler_count': 0,       # &-uh, &-um, &-hm, &-mm, &-er, &-ah (hesitations)
        'unfilled_pause_count': 0,  # (.), (..), (...) silent pauses
        'repetition_count': 0,   # [/] markers (same word repeated)
        'revision_count': 0,     # [//] markers (self-corrections)
        'fragment_count': 0,     # &+word (abandoned word attempts)
    }
    
    # ===== FILLED PAUSES =====
    # Standard CHAT notation: &-uh, &-um, &-hm, &-mm, &-er, &-ah, &-mhm, etc.
    # Also catch variations like &-you_know
    filler_pattern = r'&-(?:uh|um|hm|mm|er|ah|mhm|uhuh|uhhuh|oh|eh|huh|hmm|you_know)\b'
    fillers = re.findall(filler_pattern, raw_text, re.IGNORECASE)
    biomarkers['filler_count'] = len(fillers)
    
    # ===== UNFILLED PAUSES =====
    # CHAT notation: (.) short pause, (..) medium pause, (...) long pause
    # Also catch timed pauses like (0.5) or (2.3)
    unfilled_pauses = re.findall(r'\(\.+\)', raw_text)  # (.), (..), (...)
    timed_pauses = re.findall(r'\(\d+\.?\d*\)', raw_text)  # (0.5), (2), etc.
    biomarkers['unfilled_pause_count'] = len(unfilled_pauses) + len(timed_pauses)
    
    # ===== REPETITIONS =====
    # CHAT notation: [/] marks exact repetition
    # e.g., "the the [/] the dog" means "the" was repeated
    repetitions = re.findall(r'\[/\]', raw_text)
    biomarkers['repetition_count'] = len(repetitions)
    
    # ===== REVISIONS (RETRACINGS) =====
    # CHAT notation: [//] marks retracing (self-correction)
    # [///] marks retracing with reformulation
    # e.g., "I want [//] I need water"
    revisions = re.findall(r'\[//+\]', raw_text)
    biomarkers['revision_count'] = len(revisions)
    
    # ===== PHONOLOGICAL FRAGMENTS =====
    # CHAT notation: &+word marks phonological fragment (incomplete word attempt)
    # e.g., "&+un I don't know" (started to say something, abandoned)
    fragments = re.findall(r'&\+\w+', raw_text)
    biomarkers['fragment_count'] = len(fragments)
    
    return biomarkers


def clean_chat_text(raw_text):
    """
    Clean CHAT annotation conventions from a transcript.
    
    CRITICAL: 
    - Preserves filled pauses (converts &-uh → uh) as diagnostic biomarkers
    - Completes incomplete words (fallin(g) → falling)
    - Removes CHAT markers but preserves semantic content
    
    Returns: cleaned_text string
    """
    text = raw_text
    
    # Remove utterance boundary markers we added
    text = text.replace(' ||| ', ' ')
    
    # =========================================================
    # STEP 1: PRESERVE FILLED PAUSES (convert to words)
    # These are critical AD biomarkers - DO NOT REMOVE
    # =========================================================
    filled_pause_map = {
        r'&-uh\b': 'uh',
        r'&-um\b': 'um', 
        r'&-hm\b': 'hm',
        r'&-hmm\b': 'hmm',
        r'&-mm\b': 'mm',
        r'&-er\b': 'er',
        r'&-ah\b': 'ah',
        r'&-oh\b': 'oh',
        r'&-eh\b': 'eh',
        r'&-huh\b': 'huh',
        r'&-mhm\b': 'mhm',
        r'&-uhuh\b': 'uhuh',
        r'&-uhhuh\b': 'uhhuh',
        r'&-you_know\b': 'you know',
    }
    for pattern, replacement in filled_pause_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # =========================================================
    # STEP 2: COMPLETE INCOMPLETE WORDS
    # CHAT notation: fallin(g) → falling, goin(g) → going
    # =========================================================
    text = re.sub(r"(\w+)\((\w+)\)", r"\1\2", text)
    
    # =========================================================
    # STEP 3: REMOVE TIMING MARKERS
    # CHAT uses •0_1234• or \x150_1234\x15 for audio timing
    # =========================================================
    text = re.sub(r'\x15\d+_\d+\x15', '', text)  # Bullet timing
    text = re.sub(r'•\d+_\d+•', '', text)         # Alternative bullet
    text = re.sub(r'\u0015\d+_\d+\u0015', '', text)  # Unicode version
    
    # =========================================================
    # STEP 4: HANDLE REPLACEMENTS [: word]
    # e.g., "childrens [: children]" → "children"
    # =========================================================
    text = re.sub(r'\S+\s*\[:\s*([^\]]+)\]', r'\1', text)
    
    # =========================================================
    # STEP 5: REMOVE CHAT MARKERS (but content is already counted)
    # =========================================================
    
    # Error markers [*], [*p:n], etc.
    text = re.sub(r'\[\*[^\]]*\]', '', text)
    
    # Retracing markers [/], [//], [///] - already counted
    text = re.sub(r'\[/+\]', '', text)
    
    # Completion markers [^ word]
    text = re.sub(r'\[\^[^\]]*\]', '', text)
    
    # Comment markers [% word]
    text = re.sub(r'\[%[^\]]*\]', '', text)
    
    # Action/event markers [=! laughs]
    text = re.sub(r'\[=![^\]]*\]', '', text)
    
    # Explanation markers [= word]
    text = re.sub(r'\[=[^\]]*\]', '', text)
    
    # Best guess markers [?]
    text = re.sub(r'\[\?\]', '', text)
    
    # Excluded utterance markers [+ exc]
    text = re.sub(r'\[\+\s*exc\]', '', text)
    
    # Other bracketed codes [+ word], [- word]
    text = re.sub(r'\[[+\-][^\]]*\]', '', text)
    
    # Grammatical markers [+ gram]
    text = re.sub(r'\[\+[^\]]*\]', '', text)
    
    # Remove remaining square brackets but keep content inside
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    
    # Overlap markers <word word> - keep content
    text = re.sub(r'<([^>]*)>', r'\1', text)
    
    # =========================================================
    # STEP 6: REMOVE SPECIAL UTTERANCE TERMINATORS
    # =========================================================
    text = re.sub(r'\+\.\.\.', '', text)   # +... trailing off
    text = re.sub(r'\+/\.', '', text)       # +/. interruption  
    text = re.sub(r'\+//\.', '', text)      # +//. self-interruption
    text = re.sub(r'\+\.\.\?', '', text)    # +..? trailing off question
    text = re.sub(r'\+["\'!,/]', '', text)  # other special terminators
    text = re.sub(r'\+<', '', text)         # lazy overlap
    text = re.sub(r'\+\^', '', text)        # quick uptake
    
    # =========================================================
    # STEP 7: REMOVE PHONOLOGICAL FRAGMENTS (already counted)
    # =========================================================
    text = re.sub(r'&\+\w+', '', text)
    
    # =========================================================
    # STEP 8: REMOVE REMAINING &- PATTERNS not caught above
    # =========================================================
    text = re.sub(r'&-\w+', '', text)
    
    # =========================================================
    # STEP 9: REMOVE PARALINGUISTIC MARKERS
    # &=laughs, &=coughs, &=sighs, etc.
    # =========================================================
    text = re.sub(r'&=\w+', '', text)
    
    # =========================================================
    # STEP 10: REMOVE UNINTELLIGIBLE/UNTRANSCRIBED MARKERS
    # =========================================================
    text = re.sub(r'\bxxx\b', '', text)  # unintelligible
    text = re.sub(r'\bwww\b', '', text)  # untranscribed  
    text = re.sub(r'\byyy\b', '', text)  # phonological coding
    text = re.sub(r'\b0\b', '', text)    # silence marker in some formats
    
    # =========================================================
    # STEP 11: REMOVE LANGUAGE/SPECIAL MARKERS
    # @s:eng, @q, @l, @wp, etc.
    # =========================================================
    text = re.sub(r'@\w+(?::\w+)?', '', text)
    
    # =========================================================
    # STEP 12: REMOVE PAUSE MARKERS (already counted)
    # =========================================================
    text = re.sub(r'\(\.+\)', '', text)        # (.), (..), (...)
    text = re.sub(r'\(\d+\.?\d*\)', '', text)  # timed pauses (0.5)
    
    # =========================================================
    # STEP 13: REMOVE OTHER PARENTHETICAL CONTENT
    # =========================================================
    text = re.sub(r'\([^)]*\)', '', text)
    
    # =========================================================
    # STEP 14: REMOVE SPECIAL CHAT PUNCTUATION
    # =========================================================
    text = re.sub(r'‡', ' ', text)  # utterance delimiter
    text = re.sub(r'„', '', text)   # quotation marker
    text = re.sub(r'↫', '', text)   # lazy overlap marker
    
    # =========================================================
    # STEP 15: CLEAN UP PUNCTUATION AND WHITESPACE
    # =========================================================
    # Remove multiple punctuation
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\?+', '?', text)
    text = re.sub(r'!+', '!', text)
    
    # Fix punctuation spacing
    text = re.sub(r'\s+([.?!,;:])', r'\1', text)
    text = re.sub(r'([.?!])\s*([.?!])', r'\1', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove isolated punctuation at start/end
    text = re.sub(r'^[.?!,;:\s]+', '', text)
    text = re.sub(r'[.?!,;:\s]+$', '', text)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def process_transcripts(input_path, output_dir):
    """
    Read transcripts_raw.csv, clean each transcript, output transcripts_clean.csv
    """
    output_path = os.path.join(output_dir, 'transcripts_clean.csv')
    records = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Cleaning {len(rows)} transcripts...")
    print(f"  Preserving filled pauses as words (uh, um, etc.)")
    print(f"  Counting biomarkers before cleaning")
    
    # Track totals for summary
    total_biomarkers = {
        'filler_count': 0,
        'unfilled_pause_count': 0,
        'repetition_count': 0,
        'revision_count': 0,
        'fragment_count': 0,
    }
    
    empty_count = 0
    
    for i, row in enumerate(rows):
        raw_text = row.get('raw_text', '')
        
        # Count biomarkers BEFORE cleaning
        biomarkers = count_biomarkers(raw_text)
        
        # Clean the text (preserving fillers as words)
        cleaned_text = clean_chat_text(raw_text)
        
        if not cleaned_text.strip():
            empty_count += 1
        
        # Accumulate totals
        for key in total_biomarkers:
            total_biomarkers[key] += biomarkers[key]
        
        record = {
            'participant_id': row['participant_id'],
            'session': row['session'],
            'folder_group': row['folder_group'],
            'diagnosis': row['diagnosis'],
            'task_folder': row['task_folder'],
            'task_detail': row['task_detail'],
            'cha_path': row.get('cha_path', ''),
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'utterance_count': row['utterance_count'],
            # Biomarker counts
            'filler_count': biomarkers['filler_count'],
            'unfilled_pause_count': biomarkers['unfilled_pause_count'],
            'repetition_count': biomarkers['repetition_count'],
            'revision_count': biomarkers['revision_count'],
            'fragment_count': biomarkers['fragment_count'],
        }
        records.append(record)
        
        if (i + 1) % 200 == 0:
            print(f"  Cleaned {i + 1}/{len(rows)}")
    
    # Summary
    total_all = sum(total_biomarkers.values())
    
    print(f"\n{'='*60}")
    print(f"CLEANING COMPLETE")
    print(f"{'='*60}")
    print(f"  Records: {len(records)}")
    print(f"  Empty after cleaning: {empty_count}")
    
    print(f"\n📊 BIOMARKER COUNTS (diagnostic indicators for AD):")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │ Filled pauses (&-uh, &-um):    {total_biomarkers['filler_count']:>10,} │")
    print(f"  │ Unfilled pauses (.), (..):     {total_biomarkers['unfilled_pause_count']:>10,} │")
    print(f"  │ Repetitions [/]:               {total_biomarkers['repetition_count']:>10,} │")
    print(f"  │ Revisions [//]:                {total_biomarkers['revision_count']:>10,} │")
    print(f"  │ Fragments &+word:              {total_biomarkers['fragment_count']:>10,} │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │ TOTAL BIOMARKERS:              {total_all:>10,} │")
    print(f"  └─────────────────────────────────────────────┘")
    
    if len(records) > 0:
        avg_fillers = total_biomarkers['filler_count'] / len(records)
        print(f"\n  Average fillers per transcript: {avg_fillers:.2f}")
    
    # Word count stats
    non_empty = [r for r in records if r['cleaned_text'].strip()]
    if non_empty:
        word_counts = [len(r['cleaned_text'].split()) for r in non_empty]
        print(f"\n  Word count stats (after cleaning):")
        print(f"    Average: {sum(word_counts)/len(word_counts):.1f}")
        print(f"    Min: {min(word_counts)}")
        print(f"    Max: {max(word_counts)}")
    
    # Write CSV
    os.makedirs(output_dir, exist_ok=True)
    fieldnames = [
        'participant_id', 'session', 'folder_group', 'diagnosis',
        'task_folder', 'task_detail', 'cha_path',
        'raw_text', 'cleaned_text', 'utterance_count',
        'filler_count', 'unfilled_pause_count', 'repetition_count',
        'revision_count', 'fragment_count'
    ]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    print(f"\n✅ Saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Clean CHAT annotations from transcripts')
    parser.add_argument('--input', '-i', required=True, help='Path to transcripts_raw.csv')
    parser.add_argument('--output_dir', '-o', required=True, help='Output directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: input not found: {args.input}")
        return
    
    process_transcripts(args.input, args.output_dir)


if __name__ == '__main__':
    main()
