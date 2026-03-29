#!/usr/bin/env python3
"""
Silence-Based Audio Segmentation and Whisper Transcription Evaluation (with fuzzy time matching)
- Segments audio on silence with per-segment length constraints (5s–30s)
- Transcribes each segment with Whisper
- Fuzzily aligns each segment to CSV timestamped references (overlap-first, nearest-time fallback)
- Computes WER/CER overall and per-segment
"""

import csv
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

import numpy as np
from pydub import AudioSegment, silence
import whisper
from jiwer import wer, cer

try:
    # Optional: improves fuzzy text tie-breakers if installed
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False


# ----------------------------
# Data structures & parsing
# ----------------------------

class TranscriptSegment:
    """Represents a single segment of the transcript."""
    def __init__(self, start_time: float, end_time: float, text: str, index: int):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.index = index
        self.duration = max(0.0, end_time - start_time)

    def __repr__(self):
        return f"Segment({self.index}: {self.start_time:.2f}s-{self.end_time:.2f}s, '{self.text[:30]}...')"


def parse_timestamp(ts: str) -> float:
    """
    Convert timestamp string to seconds.
    Supports: M:SS, MM:SS, H:MM:SS
    """
    parts = ts.strip().split(':')
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    raise ValueError(f"Invalid timestamp format: {ts}")


def load_transcript_csv(csv_path: str, audio_duration: Optional[float] = None) -> List[TranscriptSegment]:
    """
    Load and parse a transcript CSV that alternates:
      timestamp line, then text line, repeated...
    End times are inferred from the next timestamp; the last one uses audio_duration if provided.
    """
    segments: List[TranscriptSegment] = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        # keep non-empty lines
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Collect (timestamp_sec, text)
    pairs: List[Tuple[float, str]] = []
    i = 0
    while i < len(lines) - 1:
        if re.match(r'^\d+:\d+', lines[i]):
            start = parse_timestamp(lines[i])
            text = lines[i + 1]
            pairs.append((start, text))
            i += 2
        else:
            i += 1

    # Build segments with end times from next timestamp
    for idx, (st, tx) in enumerate(pairs):
        if idx + 1 < len(pairs):
            et = pairs[idx + 1][0]
        else:
            et = audio_duration if audio_duration is not None else st + 10.0
        segments.append(TranscriptSegment(st, et, tx, idx))

    return segments


# ----------------------------
# Silence-based segmentation
# ----------------------------

def segment_audio_by_silence(
    audio_path: str,
    output_dir: str,
    min_segment_ms: int = 5000,
    max_segment_ms: int = 30000,
    min_silence_len: int = 700,
    silence_thresh_db: Optional[int] = None,
    keep_silence_ms: int = 200
) -> List[Tuple[str, float, float]]:
    """
    Split audio into segments by silence with duration constraints.
    Returns list of (segment_path, start_sec, end_sec).
    """
    os.makedirs(output_dir, exist_ok=True)

    audio = AudioSegment.from_wav(audio_path)
    duration_ms = len(audio)

    # Pick threshold if not provided: a bit above the noise floor
    if silence_thresh_db is None:
        silence_thresh_db = int(audio.dBFS - 14)  # heuristic, tweak as needed

    # Initial nonsilent spans
    nonsilent = silence.detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh_db
    )
    # If nothing detected, treat whole file as one span
    if not nonsilent:
        nonsilent = [[0, duration_ms]]

    # Merge near-by spans (if small silences in between)
    merged: List[List[int]] = []
    gap_merge_ms = 300  # merge if silence between spans is tiny
    for start, end in nonsilent:
        if not merged:
            merged.append([start, end])
        else:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= gap_merge_ms:
                merged[-1][1] = end
            else:
                merged.append([start, end])

    # Enforce min/max duration by merging short spans and splitting long ones
    normalized_spans: List[Tuple[int, int]] = []
    buffer_start, buffer_end = merged[0]
    for s, e in merged[1:] + [[None, None]]:  # sentinel to flush last
        if s is not None:
            # decide whether to merge with buffer to reach min length
            if (buffer_end - buffer_start) < min_segment_ms:
                # merge buffer with next span
                buffer_end = e
                continue
        # Flush buffer into normalized chunks of <= max_segment_ms
        chunk_start = buffer_start
        while (buffer_end - chunk_start) > max_segment_ms:
            cut = chunk_start + max_segment_ms
            normalized_spans.append((chunk_start, cut))
            chunk_start = cut
        normalized_spans.append((chunk_start, buffer_end))
        # start a new buffer
        if s is not None:
            buffer_start, buffer_end = s, e

    # Re-add a little silence padding at edges (bounded)
    final_spans: List[Tuple[int, int]] = []
    for st, et in normalized_spans:
        st2 = max(0, st - keep_silence_ms)
        et2 = min(duration_ms, et + keep_silence_ms)
        # clamp to [5s, 30s] in the final pass if padding expanded too far
        seg_len = et2 - st2
        if seg_len < min_segment_ms:
            # expand right if possible
            need = min_segment_ms - seg_len
            et2 = min(duration_ms, et2 + need)
        if (et2 - st2) > max_segment_ms:
            et2 = st2 + max_segment_ms
        final_spans.append((st2, et2))

    # Export
    segment_paths: List[Tuple[str, float, float]] = []
    for idx, (st, et) in enumerate(final_spans):
        clip = audio[st:et]
        out_path = os.path.join(output_dir, f"segment_{idx:04d}.wav")
        clip.export(out_path, format="wav")
        segment_paths.append((out_path, st / 1000.0, et / 1000.0))

    print(f"Created {len(segment_paths)} segments (5–30s) in {output_dir}")
    return segment_paths


# ----------------------------
# Transcription
# ----------------------------

def transcribe_segments(segment_paths: List[str], model_name: str = "base", language: Optional[str] = "fa") -> List[str]:
    """
    Transcribe audio segments using Whisper.
    """
    print(f"\nLoading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    transcriptions: List[str] = []
    print(f"\nTranscribing {len(segment_paths)} segments...")
    for i, seg in enumerate(segment_paths):
        print(f"  Transcribing {i}: {Path(seg).name}")
        result = model.transcribe(seg, language=language) if language else model.transcribe(seg)
        txt = result.get("text", "").strip()
        transcriptions.append(txt)
        print(f"    -> {txt[:60]}...")
    return transcriptions


# ----------------------------
# Fuzzy time alignment
# ----------------------------

def build_reference_for_window(
    window_start: float,
    window_end: float,
    ref_segments: List[TranscriptSegment],
    nearest_tolerance_s: float = 6.0,
    whisper_text_hint: Optional[str] = None
) -> str:
    """
    Construct a reference text for an audio window [window_start, window_end] by:
      1) concatenating all CSV segments whose [start,end] overlaps the window
      2) if none overlap, pick the *nearest* CSV segment by start time within nearest_tolerance_s
      3) optional: if multiple ties remain and rapidfuzz is available, break ties by text similarity
    """
    # 1) Overlap set
    overlaps = []
    for seg in ref_segments:
        if not (seg.end_time <= window_start or seg.start_time >= window_end):
            overlaps.append(seg)

    if overlaps:
        return " ".join(s.text for s in overlaps)

    # 2) Nearest-start fallback
    starts = np.array([s.start_time for s in ref_segments], dtype=float)
    idx = int(np.argmin(np.abs(starts - window_start)))
    nearest = ref_segments[idx]
    if abs(nearest.start_time - window_start) <= nearest_tolerance_s:
        return nearest.text

    # 3) Optional text similarity (if hint provided and rapidfuzz available)
    if whisper_text_hint and HAVE_RAPIDFUZZ:
        best_score, best_text = -1.0, ""
        for seg in ref_segments:
            score = fuzz.partial_ratio(whisper_text_hint, seg.text)
            if score > best_score:
                best_score, best_text = score, seg.text
        return best_text

    # Give *something* deterministic: nearest regardless of tolerance
    return nearest.text


def evaluate_fuzzy_time_aligned(
    ref_segments: List[TranscriptSegment],
    pred_windows: List[Tuple[float, float]],   # [(start,end) in seconds]
    whisper_texts: List[str]
) -> Dict:
    """
    For each predicted window, build a reference by fuzzy time alignment, then compute WER/CER.
    Also compute overall WER/CER on the concatenated text.
    """
    assert len(pred_windows) == len(whisper_texts)

    per_seg_results = []
    assembled_refs = []
    for i, ((st, et), hyp) in enumerate(zip(pred_windows, whisper_texts)):
        ref_text = build_reference_for_window(
            window_start=st,
            window_end=et,
            ref_segments=ref_segments,
            nearest_tolerance_s=6.0,
            whisper_text_hint=hyp
        )
        assembled_refs.append(ref_text)

        seg_wer = wer(ref_text, hyp)
        seg_cer = cer(ref_text, hyp)
        per_seg_results.append({
            "segment_index": i,
            "start_time": st,
            "end_time": et,
            "duration": max(0.0, et - st),
            "reference": ref_text,
            "whisper": hyp,
            "wer": seg_wer,
            "cer": seg_cer,
        })

    overall_ref = " ".join(assembled_refs)
    overall_hyp = " ".join(whisper_texts)

    return {
        "overall_metrics": {
            "word_error_rate": wer(overall_ref, overall_hyp),
            "character_error_rate": cer(overall_ref, overall_hyp),
            "total_segments": len(per_seg_results),
        },
        "segment_metrics": per_seg_results,
    }


def print_evaluation_summary(results: Dict):
    """Print a summary of the evaluation results."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    overall = results["overall_metrics"]
    print(f"\nOverall Metrics:")
    print(f"  Word Error Rate (WER):      {overall['word_error_rate']:.2%}")
    print(f"  Character Error Rate (CER): {overall['character_error_rate']:.2%}")
    print(f"  Total Segments:             {overall['total_segments']}")

    if results["segment_metrics"]:
        segment_wers = [seg["wer"] for seg in results["segment_metrics"]]
        segment_cers = [seg["cer"] for seg in results["segment_metrics"]]

        print(f"\nPer-Segment Statistics:")
        print(f"  Average WER:  {np.mean(segment_wers):.2%}")
        print(f"  Std Dev WER:  {np.std(segment_wers):.2%}")
        print(f"  Average CER:  {np.mean(segment_cers):.2%}")
        print(f"  Std Dev CER:  {np.std(segment_cers):.2%}")

        worst_wer_idx = int(np.argmax(segment_wers))
        best_wer_idx = int(np.argmin(segment_wers))

        print(f"\nBest Segment (#{best_wer_idx}): WER={segment_wers[best_wer_idx]:.2%}")
        print(f"  Ref: {results['segment_metrics'][best_wer_idx]['reference'][:80]}")
        print(f"  Hyp: {results['segment_metrics'][best_wer_idx]['whisper'][:80]}")

        print(f"\nWorst Segment (#{worst_wer_idx}): WER={segment_wers[worst_wer_idx]:.2%}")
        print(f"  Ref: {results['segment_metrics'][worst_wer_idx]['reference'][:80]}")
        print(f"  Hyp: {results['segment_metrics'][worst_wer_idx]['whisper'][:80]}")

    print("\n" + "=" * 80)


def save_detailed_results(results: Dict, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


# ----------------------------
# Pipeline
# ----------------------------

def run_episode(
    ep: int,
    audio_path: str,
    csv_path: str,
    whisper_models: List[str],
    min_seg_s: int = 5,
    max_seg_s: int = 30,
    min_silence_len_ms: int = 700,
    silence_thresh_db: Optional[int] = None
):
    print("\n" + "-" * 80)
    print(f"Episode {ep}")
    print("Audio:", audio_path)
    print("CSV:  ", csv_path)

    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}")
        return

    audio = AudioSegment.from_wav(audio_path)
    audio_dur_s = len(audio) / 1000.0

    # Load CSV transcript (with end times inferred from next timestamps)
    print("\nStep 1: Loading transcript CSV...")
    ref_segments = load_transcript_csv(csv_path, audio_duration=audio_dur_s)
    print(f"Loaded {len(ref_segments)} reference entries")

    # Silence-based segmentation (5–30s)
    print("\nStep 2: Segmenting audio by silence...")
    seg_dir = f"audio_segments_{ep}"
    seg_triplets = segment_audio_by_silence(
        audio_path=audio_path,
        output_dir=seg_dir,
        min_segment_ms=min_seg_s * 1000,
        max_segment_ms=max_seg_s * 1000,
        min_silence_len=min_silence_len_ms,
        silence_thresh_db=silence_thresh_db,
        keep_silence_ms=200
    )
    seg_paths = [p for (p, _, _) in seg_triplets]
    seg_windows = [(st, et) for (_, st, et) in seg_triplets]

    for mname in whisper_models:
        results_path = f"evaluation_results_{ep}_{mname}.json"
        print("\n" + "-" * 40)
        print(f"Using Whisper model: {mname}")

        # Transcribe
        print("\nStep 3: Transcribing segments...")
        hyp_texts = transcribe_segments(seg_paths, model_name=mname, language="fa")

        # Evaluate with fuzzy time alignment
        print("\nStep 4: Fuzzy time-aligned evaluation...")
        results = evaluate_fuzzy_time_aligned(
            ref_segments=ref_segments,
            pred_windows=seg_windows,
            whisper_texts=hyp_texts
        )

        # Summaries + Save
        print_evaluation_summary(results)
        save_detailed_results(results, results_path)


def main():
    EPISODES = ["amazon", "eshgh", "hoosh", "madrese", "mia",
            "nooshabe", "norooz", "safar", "soal", "youtuber"]
    WHISPER_MODELS = ["medium", "large"]

    print("=" * 80)
    print("WHISPER (Silence-Seg) EVALUATION PIPELINE")
    print("=" * 80)

    for ep in EPISODES:
        AUDIO_PATH = f"kouman/{ep}.wav"
        CSV_PATH = f"kouman_transcript/{ep}.csv"

        run_episode(
            ep=ep,
            audio_path=AUDIO_PATH,
            csv_path=CSV_PATH,
            whisper_models=WHISPER_MODELS,
            min_seg_s=5,
            max_seg_s=30,
            min_silence_len_ms=700,     # tweak based on material
            silence_thresh_db=None       # auto pick ~ (dBFS - 14)
        )

    print("\nDone! Check the output JSON files for detailed results.")


if __name__ == "__main__":
    main()
