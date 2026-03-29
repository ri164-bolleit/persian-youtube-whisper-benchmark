#!/usr/bin/env python3
"""
Audio Segmentation and Whisper Transcription Evaluation
This script segments an audio file based on a transcript CSV, transcribes each segment
with Whisper, and evaluates the accuracy.
"""

import csv
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
import json

import numpy as np
from pydub import AudioSegment
import whisper
from jiwer import wer, cer


class TranscriptSegment:
    """Represents a single segment of the transcript."""
    
    def __init__(self, start_time: float, end_time: float, text: str, index: int):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.index = index
        self.duration = end_time - start_time
        
    def __repr__(self):
        return f"Segment({self.index}: {self.start_time:.2f}s-{self.end_time:.2f}s, '{self.text[:30]}...')"


def parse_timestamp(timestamp_str: str) -> float:
    """
    Convert timestamp string to seconds.
    Supports formats: M:SS, MM:SS, H:MM:SS
    
    Args:
        timestamp_str: Timestamp string (e.g., "0:05", "1:23", "1:23:45")
    
    Returns:
        Time in seconds as float
    """
    parts = timestamp_str.strip().split(':')
    
    if len(parts) == 2:  # M:SS or MM:SS
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    elif len(parts) == 3:  # H:MM:SS
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")


def load_transcript_csv(csv_path: str) -> List[TranscriptSegment]:
    """
    Load and parse the transcript CSV file.
    
    Expected format:
    Line 1: timestamp
    Line 2: text
    Line 3: timestamp
    Line 4: text
    ...
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of TranscriptSegment objects
    """
    segments = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Parse alternating timestamp and text lines
    i = 0
    segment_index = 0
    
    while i < len(lines) - 1:
        timestamp_line = lines[i]
        text_line = lines[i + 1]
        
        # Check if this looks like a timestamp
        if re.match(r'^\d+:\d+', timestamp_line):
            start_time = parse_timestamp(timestamp_line)
            
            # Find the next timestamp to determine end time
            if i + 2 < len(lines) and re.match(r'^\d+:\d+', lines[i + 2]):
                end_time = parse_timestamp(lines[i + 2])
            else:
                # Last segment - we'll need to adjust this later with audio duration
                end_time = start_time + 10.0  # placeholder
            
            segment = TranscriptSegment(
                start_time=start_time,
                end_time=end_time,
                text=text_line,
                index=segment_index
            )
            segments.append(segment)
            segment_index += 1
            
            i += 2
        else:
            i += 1
    
    return segments


def segment_audio(audio_path: str, segments: List[TranscriptSegment], 
                  output_dir: str = "audio_segments") -> List[str]:
    """
    Segment the audio file based on transcript timestamps.
    
    Args:
        audio_path: Path to the input audio file (.wav)
        segments: List of TranscriptSegment objects
        output_dir: Directory to save audio segments
        
    Returns:
        List of paths to segmented audio files
    """
    print(f"Loading audio file: {audio_path}")
    audio = AudioSegment.from_wav(audio_path)
    audio_duration_seconds = len(audio) / 1000.0
    
    # Update the last segment's end time with actual audio duration
    if segments:
        segments[-1].end_time = audio_duration_seconds
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    segment_paths = []
    
    print(f"\nSegmenting audio ({audio_duration_seconds:.2f}s total)...")
    for segment in segments:
        start_ms = int(segment.start_time * 1000)
        end_ms = int(segment.end_time * 1000)
        
        # Extract segment
        audio_segment = audio[start_ms:end_ms]
        
        # Save segment
        output_path = os.path.join(output_dir, f"segment_{segment.index:04d}.wav")
        audio_segment.export(output_path, format="wav")
        segment_paths.append(output_path)
        
        print(f"  Segment {segment.index}: {segment.start_time:.2f}s - {segment.end_time:.2f}s "
              f"({segment.duration:.2f}s) -> {output_path}")
    
    return segment_paths


def transcribe_segments(segment_paths: List[str], model_name: str = "base") -> List[str]:
    """
    Transcribe audio segments using Whisper.
    
    Args:
        segment_paths: List of paths to audio segment files
        model_name: Whisper model name (tiny, base, small, medium, large)
        
    Returns:
        List of transcribed texts
    """
    print(f"\nLoading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    transcriptions = []
    
    print(f"\nTranscribing {len(segment_paths)} segments...")
    for i, segment_path in enumerate(segment_paths):
        print(f"  Transcribing segment {i}...")
        
        # Transcribe with Whisper
        result = model.transcribe(segment_path, language="fa")  # fa = Farsi/Persian
        transcription = result["text"].strip()
        transcriptions.append(transcription)
        
        print(f"    -> {transcription[:60]}...")
    
    return transcriptions


def evaluate_transcriptions(reference_segments: List[TranscriptSegment], 
                           whisper_transcriptions: List[str]) -> Dict:
    """
    Evaluate Whisper transcriptions against reference transcript.
    
    Args:
        reference_segments: List of reference TranscriptSegment objects
        whisper_transcriptions: List of Whisper transcription strings
        
    Returns:
        Dictionary with evaluation metrics
    """
    reference_texts = [seg.text for seg in reference_segments]
    
    # Calculate overall metrics
    reference_combined = " ".join(reference_texts)
    whisper_combined = " ".join(whisper_transcriptions)
    
    overall_wer = wer(reference_combined, whisper_combined)
    overall_cer = cer(reference_combined, whisper_combined)
    
    # Calculate per-segment metrics
    segment_metrics = []
    
    for i, (ref_seg, whisper_text) in enumerate(zip(reference_segments, whisper_transcriptions)):
        seg_wer = wer(ref_seg.text, whisper_text)
        seg_cer = cer(ref_seg.text, whisper_text)
        
        segment_metrics.append({
            "segment_index": i,
            "start_time": ref_seg.start_time,
            "end_time": ref_seg.end_time,
            "duration": ref_seg.duration,
            "reference": ref_seg.text,
            "whisper": whisper_text,
            "wer": seg_wer,
            "cer": seg_cer,
        })
    
    results = {
        "overall_metrics": {
            "word_error_rate": overall_wer,
            "character_error_rate": overall_cer,
            "total_segments": len(reference_segments),
        },
        "segment_metrics": segment_metrics,
    }
    
    return results


def print_evaluation_summary(results: Dict):
    """Print a summary of the evaluation results."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    overall = results["overall_metrics"]
    print(f"\nOverall Metrics:")
    print(f"  Word Error Rate (WER):      {overall['word_error_rate']:.2%}")
    print(f"  Character Error Rate (CER): {overall['character_error_rate']:.2%}")
    print(f"  Total Segments:             {overall['total_segments']}")
    
    # Calculate segment statistics
    segment_wers = [seg["wer"] for seg in results["segment_metrics"]]
    segment_cers = [seg["cer"] for seg in results["segment_metrics"]]
    
    print(f"\nPer-Segment Statistics:")
    print(f"  Average WER:  {np.mean(segment_wers):.2%}")
    print(f"  Std Dev WER:  {np.std(segment_wers):.2%}")
    print(f"  Average CER:  {np.mean(segment_cers):.2%}")
    print(f"  Std Dev CER:  {np.std(segment_cers):.2%}")
    
    # Find best and worst segments
    worst_wer_idx = np.argmax(segment_wers)
    best_wer_idx = np.argmin(segment_wers)
    
    print(f"\nBest Segment (Segment {best_wer_idx}):")
    print(f"  WER: {segment_wers[best_wer_idx]:.2%}")
    print(f"  Reference: {results['segment_metrics'][best_wer_idx]['reference'][:80]}")
    print(f"  Whisper:   {results['segment_metrics'][best_wer_idx]['whisper'][:80]}")
    
    print(f"\nWorst Segment (Segment {worst_wer_idx}):")
    print(f"  WER: {segment_wers[worst_wer_idx]:.2%}")
    print(f"  Reference: {results['segment_metrics'][worst_wer_idx]['reference'][:80]}")
    print(f"  Whisper:   {results['segment_metrics'][worst_wer_idx]['whisper'][:80]}")
    
    print("\n" + "="*80)


def save_detailed_results(results: Dict, output_path: str = "evaluation_results_large.json"):
    """Save detailed evaluation results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


def main():

    """Main execution function."""
    # Configuration
    audios = ["amazon.wav", "eshgh.wav", "hoosh.wav", "madrese.wav", "mia.wav",
            "nooshabe.wav", "norooz.wav", "safar.wav", "soal.wav", "youtuber.wav"]

    WHISPER_MODEL = "large"  # Options: medium, large

    for audio in audios:
        # Set paths for current audio file
        AUDIO_PATH = audio
        audio_name = audio.replace(".wav", "")  # Get name without extension
        CSV_PATH = f"{audio_name}.csv"
        
        # Create directory structure for this audio file
        output_dir = audio_name
        SEGMENT_DIR = os.path.join(output_dir, "audio_segments_timestamp")
        RESULTS_PATH = os.path.join(output_dir, "evaluation_results_"+audio_name+"_large.json")

        # Create directories if they don't exist
        os.makedirs(SEGMENT_DIR, exist_ok=True)
        
        print("="*80)
        print("WHISPER TRANSCRIPTION EVALUATION PIPELINE")
        print("="*80)
        
        # Step 1: Load transcript
        print("\nStep 1: Loading transcript CSV...")
        segments = load_transcript_csv(CSV_PATH)
        print(f"Loaded {len(segments)} segments")
        
        # Step 2: Segment audio
        print("\nStep 2: Segmenting audio...")
        if not os.path.exists(AUDIO_PATH):
            print(f"\nERROR: Audio file not found: {AUDIO_PATH}")
            print("Please update the AUDIO_PATH variable in the script with your .wav file path")
            return
        
        segment_paths = segment_audio(AUDIO_PATH, segments, SEGMENT_DIR)
        
        # Step 3: Transcribe with Whisper
        print("\nStep 3: Transcribing with Whisper...")
        whisper_transcriptions = transcribe_segments(segment_paths, WHISPER_MODEL)
        
        # Step 4: Evaluate
        print("\nStep 4: Evaluating transcriptions...")
        results = evaluate_transcriptions(segments, whisper_transcriptions)
        
        # Step 5: Display and save results
        print_evaluation_summary(results)
        save_detailed_results(results, RESULTS_PATH)
        
        print("\nDone! Check the output files for detailed results.")


if __name__ == "__main__":
    main()
