"""Core audio processing functionality."""

import os
import ffmpeg
import logging
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import multiprocessing

from ..utils.loaders import load_audio_modules, load_audio_processing_modules
from ..utils.constants import SUPPORTED_FORMATS, SUPPORTED_CODECS

# Determine optimal number of workers based on CPU cores
DEFAULT_MAX_WORKERS = max(
    1, min(multiprocessing.cpu_count() - 1, 4)
)  # Leave 1 core free, max 4 workers


def get_relative_path(file_path: str, input_folder: str) -> str:
    """Get path relative to input folder."""
    try:
        return str(Path(file_path).relative_to(Path(input_folder)))
    except ValueError:
        return str(Path(file_path))


def load_processed_files(tracking_file: str) -> set:
    """Load the set of previously processed files."""
    try:
        if os.path.exists(tracking_file):
            with open(tracking_file, "r") as f:
                return set(json.load(f))
        else:
            # Create the tracking file with an empty list if it doesn't exist
            os.makedirs(os.path.dirname(tracking_file), exist_ok=True)
            with open(tracking_file, "w") as f:
                json.dump([], f, indent=2)
            return set()
    except Exception as e:
        logging.warning(f"Error loading tracking file: {e}")
    return set()


def save_processed_files(tracking_file: str, processed_files: set) -> None:
    """Save the set of processed files."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(tracking_file), exist_ok=True)
        # Convert any Path objects to strings before serialization
        processed_files_str = {str(path) for path in processed_files}
        with open(tracking_file, "w") as f:
            # Sort the files for better readability
            json.dump(sorted(list(processed_files_str)), f, indent=2)
    except Exception as e:
        logging.error(f"Error saving tracking file: {e}")


def process_file(file_info: tuple) -> bool:
    """Process a single audio file."""
    (
        input_path,
        duration_to_remove,
        make_backup,
        output_dir,
        dry_run,
        from_end,
        quality,
        smart_trim,
        fade_duration,
        output_format,
        segments,
        backup_dir,
        tracking_file,
        force_process,
        input_folder,
    ) = file_info

    input_path = Path(input_path)
    rel_path = get_relative_path(str(input_path), input_folder)
    filename = input_path.name
    logger = logging.getLogger("audio_trimmer")

    # Check if file was already processed using relative path
    if not force_process and tracking_file:
        processed_files = load_processed_files(tracking_file)
        if rel_path in processed_files:
            logger.info(f"Skipping already processed file: {rel_path}")
            return True

    if dry_run:
        logger.info(f"[DRY RUN] Would process: {rel_path}")
        return True

    try:
        # Determine output path with possible format conversion
        output_ext = output_format or input_path.suffix
        output_filename = input_path.stem + output_ext

        # Create output directory if it doesn't exist
        if output_dir:
            output_dir = Path(output_dir)
            # Maintain folder structure by getting relative path from input root
            input_root = Path(input_path).parent
            while "IntroRemover" in input_root.parts:
                input_root = input_root.parent
            try:
                rel_path = Path(input_path).parent.relative_to(input_root)
            except ValueError:
                # If no common root, just use the filename
                rel_path = Path()

            # Create full output path maintaining structure
            full_output_dir = output_dir / rel_path
            full_output_dir.mkdir(parents=True, exist_ok=True)

            output_path = full_output_dir / output_filename
            temp_output_path = full_output_dir / f"temp_{output_filename}"
        else:
            output_path = input_path.parent / output_filename
            temp_output_path = input_path.parent / f"temp_{output_filename}"

        # Create backup if requested
        if make_backup and backup_dir:
            backup_dir = Path(backup_dir)
            # Use same relative path structure for backups
            backup_rel_path = (
                Path(input_path).parent.relative_to(input_root)
                if output_dir
                else Path()
            )
            full_backup_dir = backup_dir / backup_rel_path
            full_backup_dir.mkdir(parents=True, exist_ok=True)

            backup_path = full_backup_dir / (input_path.name + ".backup")
            if not os.path.exists(backup_path):
                try:
                    import shutil

                    shutil.copy2(str(input_path), str(backup_path))
                    logger.info(f"Created backup: {backup_path}")
                except Exception as e:
                    logger.error(f"Failed to create backup for {filename}: {str(e)}")
                    return False

        # Get audio metadata and duration
        probe = ffmpeg.probe(str(input_path))
        duration = float(probe["streams"][0]["duration"])

        # Smart trim using silence detection
        if smart_trim:
            nonsilent_ranges = detect_silence(str(input_path))
            if nonsilent_ranges:
                start_time = nonsilent_ranges[0][0] / 1000.0  # Convert to seconds
            else:
                start_time = duration_to_remove
        else:
            start_time = duration_to_remove

        if duration <= start_time:
            logger.warning(f"{filename} is shorter than the trim duration. Skipping.")
            return False

        # Prepare ffmpeg command
        input_stream = ffmpeg.input(str(input_path))

        # Handle multi-segment trimming
        if segments:
            filter_complex = []
            parts = []
            last_end = 0

            for i, (start, end) in enumerate(segments):
                if start > last_end:
                    part = input_stream.filter("atrim", start=last_end, end=start)
                    filter_complex.append(part)
                last_end = end

            if last_end < duration:
                part = input_stream.filter("atrim", start=last_end, end=duration)
                filter_complex.append(part)

            # Concatenate all parts
            output_stream = ffmpeg.concat(*filter_complex, v=0, a=1)
        else:
            if from_end:
                duration_to_keep = duration - duration_to_remove
                output_stream = input_stream.output(
                    str(temp_output_path), t=duration_to_keep, **quality
                )
            else:
                output_stream = input_stream.output(
                    str(temp_output_path), ss=start_time, **quality
                )

        # Add fade effects if specified
        if fade_duration:
            output_stream = output_stream.filter("afade", t="in", d=fade_duration)
            if from_end:
                output_stream = output_stream.filter("afade", t="out", d=fade_duration)

        # Set output codec based on format
        if output_format and output_format in SUPPORTED_FORMATS:
            quality["acodec"] = SUPPORTED_CODECS[output_format]

        output_stream = output_stream.overwrite_output()

        # Run ffmpeg
        output_stream.run(capture_stdout=True, capture_stderr=True)

        # Move the temp file to final destination
        if os.path.exists(str(temp_output_path)):
            if os.path.exists(str(output_path)):
                os.remove(str(output_path))
            os.rename(str(temp_output_path), str(output_path))
            logger.info(f"Successfully processed: {filename}")

            if tracking_file and not dry_run:
                processed_files = load_processed_files(tracking_file)
                processed_files.add(rel_path)
                save_processed_files(tracking_file, processed_files)
                logger.info(f"Added to tracking file: {rel_path}")

            return True
        else:
            logger.error(f"Failed to process {filename}: Output file not created")
            return False

    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error processing {filename}: {e.stderr.decode()}")
        if os.path.exists(str(temp_output_path)):
            os.remove(str(temp_output_path))
        return False
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        if os.path.exists(str(temp_output_path)):
            os.remove(str(temp_output_path))
        return False


def detect_silence(
    audio_path: str, min_silence_len: int = 1000, silence_thresh: int = -50
) -> list:
    """Detect silence in audio file and return non-silent segments."""
    _, _, AudioSegment, detect_nonsilent = load_audio_modules()
    if not all([AudioSegment, detect_nonsilent]):
        raise ImportError(
            "Required modules not found. Please install pydub: pip install pydub"
        )

    audio = AudioSegment.from_file(audio_path)
    nonsilent_ranges = detect_nonsilent(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )
    return nonsilent_ranges


def preview_audio(audio_path: str, start_time: float = 0, duration: float = 10) -> None:
    """Play a preview of the audio file."""
    sd, sf, _, _ = load_audio_modules()
    if not all([sd, sf]):
        raise ImportError(
            "Required modules not found. Please install sounddevice and soundfile: pip install sounddevice soundfile"
        )

    try:
        # First get the file info to get the sample rate
        info = sf.info(audio_path)
        samplerate = info.samplerate

        # Calculate number of frames to read
        frames_to_read = int(duration * samplerate)
        start_frame = int(start_time * samplerate)

        # Read the audio data
        data, _ = sf.read(
            audio_path,
            start=start_frame,
            frames=frames_to_read,
        )

        # Play the audio
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"Error previewing audio: {e}")
        raise


def compute_audio_fingerprint(audio_path: str, sr: int = 22050) -> np.ndarray:
    """Compute a fingerprint for an audio file."""
    librosa, _, _ = load_audio_processing_modules()
    if not librosa:
        raise ImportError("Required module not found. Please install librosa")

    # Load audio with optimized parameters
    y, sr = librosa.load(
        audio_path, sr=sr, duration=30
    )  # Only load first 30 seconds for fingerprint

    # Compute mel spectrogram with optimized parameters
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=64,  # Reduced number of mel bands
        n_fft=1024,  # Smaller FFT window
        hop_length=512,
    )

    # Convert to log scale
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_db


def find_intro_match(
    audio_path: str, intro_templates: Dict[str, np.ndarray], threshold: float = 0.85
) -> Tuple[Optional[str], float, float]:
    """Find if any of the intro templates match the beginning of the audio file."""
    librosa, _, _ = load_audio_processing_modules()
    if not librosa:
        raise ImportError("Required module not found. Please install librosa")

    # Load the target audio
    y_target, sr = librosa.load(audio_path, sr=22050)
    mel_target = librosa.feature.melspectrogram(y=y_target, sr=sr)
    mel_db_target = librosa.power_to_db(mel_target, ref=np.max)

    best_match = None
    best_score = -1
    best_duration = 0

    for intro_name, intro_fingerprint in intro_templates.items():
        # Get the duration of the intro template
        intro_duration = intro_fingerprint.shape[1]

        # Compare with the beginning of the target audio
        target_segment = mel_db_target[:, :intro_duration]
        if target_segment.shape[1] < intro_duration:
            continue

        # Compute correlation score
        score = np.corrcoef(intro_fingerprint.flatten(), target_segment.flatten())[0, 1]

        if score > threshold and score > best_score:
            best_score = score
            best_match = intro_name
            best_duration = intro_duration

    if best_match:
        # Convert mel spectrogram frames to seconds
        duration_seconds = librosa.frames_to_time(best_duration, sr=sr)
        return best_match, best_score, duration_seconds

    return None, 0, 0


def learn_intro_templates(template_folder: str) -> Dict[str, np.ndarray]:
    """Learn fingerprints from a folder of intro templates."""
    templates = {}
    for ext in SUPPORTED_FORMATS:
        for file_path in Path(template_folder).glob(f"*{ext}"):
            try:
                fingerprint = compute_audio_fingerprint(str(file_path))
                templates[file_path.name] = fingerprint
                logging.info(f"Learned template from: {file_path.name}")
            except Exception as e:
                logging.error(f"Error learning template {file_path}: {str(e)}")
    return templates


def remove_audio_duration(
    input_folder: str,
    duration_to_remove: float,
    make_backup: bool = False,
    max_workers: Optional[int] = None,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    recursive: bool = True,
    from_end: bool = False,
    quality: Optional[Dict] = None,
    smart_trim: bool = False,
    fade_duration: Optional[float] = None,
    output_format: Optional[str] = None,
    batch_config: Optional[str] = None,
    segments: Optional[List[Tuple[float, float]]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    backup_dir: Optional[str] = None,
    tracking_file: Optional[str] = None,
    force_process: bool = False,
) -> None:
    """Remove duration from audio files."""
    logger = logging.getLogger("audio_trimmer")
    input_folder = Path(input_folder)

    if not input_folder.exists():
        raise ValueError(f"The folder {input_folder} does not exist.")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if backup_dir:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

    if tracking_file:
        tracking_file = str(Path(tracking_file))
        os.makedirs(os.path.dirname(tracking_file), exist_ok=True)

    if duration_to_remove <= 0:
        raise ValueError("Duration to remove must be positive.")

    # Load batch configuration if provided
    batch_durations = {}
    if batch_config and os.path.exists(batch_config):
        with open(batch_config) as f:
            batch_durations = json.load(f)

    # Collect audio files
    audio_files = []
    if recursive:
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(input_folder.rglob(f"*{ext}"))
    else:
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(input_folder.glob(f"*{ext}"))

    if not audio_files:
        logger.warning(f"No supported audio files found in {input_folder}")
        return

    total_files = len(audio_files)
    processed_files = 0

    # Use optimized number of workers if not specified
    if max_workers is None:
        max_workers = DEFAULT_MAX_WORKERS

    # Process files with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for f in audio_files:
            future = executor.submit(
                process_file,
                (
                    str(f),
                    batch_durations.get(f.name, duration_to_remove),
                    make_backup,
                    output_dir,
                    dry_run,
                    from_end,
                    quality or {},
                    smart_trim,
                    fade_duration,
                    output_format,
                    segments,
                    backup_dir,
                    tracking_file,
                    force_process,
                    str(input_folder),
                ),
            )
            futures.append(future)

        for future in futures:
            try:
                future.result()
                processed_files += 1
                if progress_callback:
                    progress_callback(processed_files, total_files)
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")


def remove_detected_intros(
    input_folder: str,
    template_folder: str,
    make_backup: bool = False,
    max_workers: Optional[int] = None,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    recursive: bool = True,
    quality: Optional[Dict] = None,
    match_threshold: float = 0.85,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    backup_dir: Optional[str] = None,
    tracking_file: Optional[str] = None,
    force_process: bool = False,
) -> None:
    """Remove intros by matching against template files."""
    logger = logging.getLogger("audio_trimmer")

    # Check for required modules
    librosa, sf, signal = load_audio_processing_modules()
    if not all([librosa, sf, signal]):
        raise ImportError(
            "Required modules not found. Please install: pip install librosa scipy"
        )

    # Learn intro templates
    logger.info("Learning intro templates...")
    intro_templates = learn_intro_templates(template_folder)

    if not intro_templates:
        raise ValueError(f"No valid intro templates found in {template_folder}")

    logger.info(f"Learned {len(intro_templates)} intro templates")

    # Collect audio files
    input_folder = Path(input_folder)
    audio_files = []
    if recursive:
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(input_folder.rglob(f"*{ext}"))
    else:
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(input_folder.glob(f"*{ext}"))

    if not audio_files:
        logger.warning(f"No supported audio files found in {input_folder}")
        return

    total_files = len(audio_files)
    processed_files = 0

    # Process files with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for f in audio_files:
            future = executor.submit(
                process_file_with_template_matching,
                (
                    str(f),
                    intro_templates,
                    make_backup,
                    output_dir,
                    dry_run,
                    quality or {},
                    match_threshold,
                    backup_dir,
                    tracking_file,
                    force_process,
                    str(input_folder),
                ),
            )
            futures.append(future)

        for future in futures:
            try:
                future.result()
                processed_files += 1
                if progress_callback:
                    progress_callback(processed_files, total_files)
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")


def process_file_with_template_matching(file_info: tuple) -> bool:
    """Process a file using template matching."""
    (
        input_path,
        intro_templates,
        make_backup,
        output_dir,
        dry_run,
        quality,
        match_threshold,
        backup_dir,
        tracking_file,
        force_process,
        input_folder,
    ) = file_info

    filename = os.path.basename(input_path)
    logger = logging.getLogger("audio_trimmer")

    if dry_run:
        logger.info(f"[DRY RUN] Would process: {filename}")
        return True

    try:
        # Find matching intro
        intro_match, score, duration = find_intro_match(
            input_path, intro_templates, match_threshold
        )

        if not intro_match:
            logger.info(f"No intro match found for: {filename}")
            return False

        logger.info(
            f"Found intro match in {filename}: {intro_match} (score: {score:.2f}, duration: {duration:.2f}s)"
        )

        # Process the rest similar to the original process_file function
        return process_file(
            (
                input_path,
                duration,
                make_backup,
                output_dir,
                dry_run,
                False,  # from_end
                quality,
                False,  # smart_trim
                None,  # fade_duration
                None,  # output_format
                None,  # segments
                backup_dir,
                tracking_file,
                force_process,
                input_folder,
            )
        )

    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return False
