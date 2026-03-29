# Whisper Persian Audio Transcription Evaluation

Evaluates [OpenAI Whisper](https://github.com/openai/whisper) transcription accuracy on Persian (Farsi) audio using two segmentation strategies.

---

## Scripts

- **`timestamp_segmentation.py`** — Splits audio using timestamps from the reference CSV for a clean 1-to-1 segment evaluation.
- **`silence_segmentation.py`** — Splits audio by detecting silence (5–30s segments), then uses fuzzy time alignment to match segments to reference entries.

---

## CSV Format

Alternating timestamp and text lines:

```
0:05
Text for the first segment
0:23
Text for the second segment
```

Supported formats: `M:SS`, `MM:SS`, `H:MM:SS`

---

## Installation

```bash
pip install -r requirements.txt
# Optional: for better fuzzy matching in silence_segmentation.py
pip install rapidfuzz
```

---

## Usage

```bash
# Timestamp-based
python timestamp_segmentation.py

# Silence-based
python silence_segmentation.py
```

Configure the episode list, Whisper model (`tiny`/`base`/`small`/`medium`/`large`), and segmentation parameters (min/max segment length, silence threshold) inside each script's `main()` function.

---

## Output

Each run prints a WER/CER summary to stdout and saves a detailed per-segment JSON file (`evaluation_results_<episode>_<model>.json`).

**Metrics:**
- **WER** (Word Error Rate) — substitutions + deletions + insertions over total reference words
- **CER** (Character Error Rate) — same, at the character level

---

## Data

Reference transcripts are located in the `kouman_transcripts/` and `roud_transcripts/` directories. The corresponding audio files can be found on Google Drive:

- **Kouman:** [Google Drive](https://drive.google.com/drive/folders/1tekt97xA0nEAJtw36oqwtHVgs4F-Y7ui?usp=sharing)
- **Roud:** [Google Drive](https://drive.google.com/drive/folders/1g2gUlJljKPEuYtKHFvHoRwnRKUV6ysb-?usp=sharing)
