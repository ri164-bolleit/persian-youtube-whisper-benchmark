# Segmentation Strategy Matters: Benchmarking Whisper on Persian YouTube Content

This repository contains the official implementation and evaluation scripts for the paper **"Segmentation Strategy Matters: Benchmarking Whisper on Persian YouTube Content"**, published at **SilkRoadNLP EACL 2026 (First Workshop on NLP and LLMs for the Iranian Language Family)**.

[Read the Paper](https://aclanthology.org/2026.silkroadnlp-1.13/)

---

## Overview

This project benchmarks the performance of OpenAI's Whisper models on Persian YouTube data (Kouman and Roud podcasts). We specifically investigate how different audio segmentation approaches affect Word Error Rate (WER) and Character Error Rate (CER):

1. **Timestamp-based Segmentation:** Uses ground-truth timestamps to isolate segments.
2. **Silence-based Segmentation:** A more "in-the-wild" approach that detects pauses to segment audio, requiring fuzzy alignment to match transcripts.

---

## Scripts

- **`src/timestamp_segmentation.py`** — Splits audio using timestamps from the reference CSV for a clean 1-to-1 segment evaluation.
- **`src/silence_segmentation.py`** — Splits audio by detecting silence (5–30s segments), then uses fuzzy time alignment to match segments to reference entries.

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


## Citation

If you use this code or our findings in your research, please cite our paper:

```bibtex
@inproceedings{iranmanesh-etal-2026-segmentation,
    title = "Segmentation Strategy Matters: Benchmarking Whisper on {P}ersian {Y}ou{T}ube Content",
    author = "Iranmanesh, Reihaneh  and
      Ziaei, Rojin  and
      Garman, Joe",
    editor = "Merchant, Rayyan  and
      Megerdoomian, Karine",
    booktitle = "The Proceedings of the First Workshop on {NLP} and {LLM}s for the {I}ranian Language Family",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.silkroadnlp-1.13/",
    doi = "10.18653/v1/2026.silkroadnlp-1.13",
    pages = "121--130"
}
