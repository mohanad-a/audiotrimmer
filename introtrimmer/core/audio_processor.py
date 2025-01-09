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

from ..utils.loaders import load_audio_modules, load_audio_processing_modules
from ..utils.constants import SUPPORTED_FORMATS, SUPPORTED_CODECS


def get_relative_path(file_path: str, input_folder: str) -> str:
    """Get path relative to input folder."""
    try:
        file_path = Path(file_path).resolve()
        input_folder = Path(input_folder).resolve()
        try:
            rel_path = file_path.relative_to(input_folder)
            # Convert to string with forward slashes for consistency
            return str(rel_path).replace(os.sep, "/")
        except ValueError:
            # If we can't get relative path, return the filename
            return file_path.name
    except Exception:
        # In case of any other error, return the filename
        return Path(file_path).name


def load_processed_files(tracking_file: str) -> set:
    """Load the set of previously processed files."""
    if not tracking_file:
        return set()

    try:
        if os.path.exists(tracking_file):
            if os.path.getsize(tracking_file) == 0:
                # File exists but is empty, initialize it
                with open(tracking_file, "w") as f:
                    json.dump([], f, indent=2)
                return set()

            with open(tracking_file, "r") as f:
                try:
                    return set(json.load(f))
                except json.JSONDecodeError:
                    # If file is corrupted, backup and start fresh
                    if os.path.exists(tracking_file + ".bak"):
                        os.remove(tracking_file + ".bak")
                    os.rename(tracking_file, tracking_file + ".bak")
                    with open(tracking_file, "w") as f:
                        json.dump([], f, indent=2)
                    return set()
        else:
            # Create the tracking file with an empty list if it doesn't exist
            os.makedirs(os.path.dirname(tracking_file), exist_ok=True)
            with open(tracking_file, "w") as f:
                json.dump([], f, indent=2)
            return set()
    except Exception as e:
        logging.warning(f"Error accessing tracking file: {e}")
        return set()


def save_processed_files(tracking_file: str, processed_files: set) -> None:
    """Save the set of processed files."""
    if not tracking_file:
        return

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(tracking_file), exist_ok=True)

        # Create a temporary file for atomic write
        temp_file = tracking_file + ".tmp"
        # Convert all paths to strings before saving
        processed_list = sorted([str(path) for path in processed_files])

        with open(temp_file, "w") as f:
            json.dump(processed_list, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk

        # Atomic rename for safer file writing
        if os.path.exists(tracking_file):
            os.replace(temp_file, tracking_file)
        else:
            os.rename(temp_file, tracking_file)

    except Exception as e:
        logging.error(f"Error saving tracking file: {e}")
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


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

    try:
        input_path = Path(input_path).resolve()
        input_folder = Path(input_folder).resolve()
        tracking_rel_path = get_relative_path(str(input_path), str(input_folder))
        filename = tracking_rel_path  # Use relative path for logging
        logger = logging.getLogger("audio_trimmer")

        # Check if file was already processed using relative path
        if not force_process and tracking_file:
            processed_files = load_processed_files(tracking_file)
            if tracking_rel_path in processed_files:
                logger.info(f"Skipping already processed file: {tracking_rel_path}")
                return True

        if dry_run:
            logger.info(f"[DRY RUN] Would process: {tracking_rel_path}")
            return True

        # Determine output path with possible format conversion
        output_ext = output_format or input_path.suffix
        output_filename = input_path.stem + output_ext

        # Create output directory if it doesn't exist
        if output_dir:
            output_dir = Path(output_dir)
            # Use the same relative path structure as the input
            output_rel_path = Path(tracking_rel_path).parent

            # Create full output path maintaining structure
            full_output_dir = output_dir / output_rel_path
            full_output_dir.mkdir(parents=True, exist_ok=True)

            output_path = full_output_dir / output_filename
            temp_output_path = full_output_dir / f"temp_{output_filename}"
        else:
            output_path = input_path.parent / output_filename
            temp_output_path = input_path.parent / f"temp_{output_filename}"

        # Create backup if requested
        if make_backup and backup_dir:
            backup_dir = Path(backup_dir)
            # Use the same relative path structure as the input
            backup_rel_path = Path(tracking_rel_path).parent
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
        try:
            probe = ffmpeg.probe(str(input_path))
            duration = float(probe["streams"][0]["duration"])
        except ffmpeg.Error as e:
            logger.error(f"Failed to probe {filename}: {e.stderr.decode()}")
            return False
        except Exception as e:
            logger.error(f"Failed to probe {filename}: {str(e)}")
            return False

        # Smart trim using silence detection
        if smart_trim:
            try:
                nonsilent_ranges = detect_silence(str(input_path))
                if nonsilent_ranges:
                    start_time = nonsilent_ranges[0][0] / 1000.0  # Convert to seconds
                else:
                    start_time = duration_to_remove
            except Exception as e:
                logger.error(f"Failed to detect silence in {filename}: {str(e)}")
                return False
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
            stream = ffmpeg.concat(*filter_complex, v=0, a=1)
        else:
            # Apply trimming
            if from_end:
                duration_to_keep = duration - duration_to_remove
                stream = input_stream.filter("atrim", duration=duration_to_keep)
            else:
                stream = input_stream.filter("atrim", start=start_time)

        # Add fade effects if specified
        if fade_duration:
            stream = stream.filter("afade", t="in", d=fade_duration)
            if from_end:
                stream = stream.filter("afade", t="out", d=fade_duration)

        # Set output codec based on format
        if output_format and output_format in SUPPORTED_FORMATS:
            quality["acodec"] = SUPPORTED_CODECS[output_format]

        # Create the final output stream
        output_stream = stream.output(str(temp_output_path), **quality)
        output_stream = output_stream.overwrite_output()

        # Run ffmpeg with detailed error capture
        try:
            # Print the ffmpeg command for debugging
            logger.debug(f"FFmpeg command: {' '.join(output_stream.get_args())}")
            output_stream.run(capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error processing {filename}: {error_message}")
            if os.path.exists(str(temp_output_path)):
                os.remove(str(temp_output_path))
            return False

        # Move the temp file to final destination
        if os.path.exists(str(temp_output_path)):
            try:
                if os.path.exists(str(output_path)):
                    os.remove(str(output_path))
                os.rename(str(temp_output_path), str(output_path))
                logger.info(f"Successfully processed: {filename}")

                if tracking_file and not dry_run:
                    processed_files = load_processed_files(tracking_file)
                    processed_files.add(tracking_rel_path)
                    save_processed_files(tracking_file, processed_files)
                    logger.info(f"Added to tracking file: {tracking_rel_path}")

                return True
            except Exception as e:
                logger.error(f"Failed to move temp file for {filename}: {str(e)}")
                if os.path.exists(str(temp_output_path)):
                    try:
                        os.remove(str(temp_output_path))
                    except:
                        pass
                return False
        else:
            logger.error(
                f"Failed to process {filename}: Output file was not created by FFmpeg"
            )
            return False

    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg error processing {filename}: {error_message}")
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

    # Load audio with a fixed duration if it's an intro template
    y, sr = librosa.load(audio_path, sr=sr)

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

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
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
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

    # Calculate number of workers for each pool
    import multiprocessing

    total_cpus = multiprocessing.cpu_count()
    available_cpus = max(1, total_cpus - 2)
    match_workers = max(1, available_cpus // 3)
    process_workers = max(1, available_cpus - match_workers)

    logger.info(
        f"Using {match_workers} workers for matching and {process_workers} workers for processing"
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
    matched_files = 0
    processed_files = 0
    files_to_process = 0

    def check_file_match(f):
        """Check a single file for intro match."""
        try:
            input_path = Path(f).resolve()
            tracking_rel_path = get_relative_path(str(input_path), str(input_folder))

            # Check if already processed
            if not force_process and tracking_file:
                processed_files_set = load_processed_files(tracking_file)
                if tracking_rel_path in processed_files_set:
                    logger.info(f"Skipping already processed file: {tracking_rel_path}")
                    return None

            # Find intro match
            intro_match, score, duration = find_intro_match(
                str(input_path), intro_templates, match_threshold
            )

            if not intro_match:
                logger.info(f"No intro match found for: {tracking_rel_path}")
                return None

            logger.info(
                f"Found intro match in {tracking_rel_path}: {intro_match} (score: {score:.2f}, duration: {duration:.2f}s)"
            )

            return (input_path, duration, tracking_rel_path)
        except Exception as e:
            logger.error(f"Error checking file {tracking_rel_path}: {str(e)}")
            return None

    # Start matching files in parallel while processing
    with ThreadPoolExecutor(
        max_workers=match_workers
    ) as match_executor, ThreadPoolExecutor(
        max_workers=process_workers
    ) as process_executor:

        # Submit all files for matching
        match_futures = [
            match_executor.submit(check_file_match, f) for f in audio_files
        ]
        process_futures = []

        # Process results as they come in
        for future in match_futures:
            try:
                result = future.result()
                matched_files += 1
                if progress_callback:
                    progress_callback(matched_files, total_files, "matching")

                if result:
                    files_to_process += 1
                    input_path, duration, tracking_rel_path = result
                    # Submit for processing
                    process_future = process_executor.submit(
                        process_file,
                        (
                            str(input_path),
                            duration,
                            make_backup,
                            output_dir,
                            dry_run,
                            False,  # from_end
                            quality or {},
                            False,  # smart_trim
                            None,  # fade_duration
                            None,  # output_format
                            None,  # segments
                            backup_dir,
                            tracking_file,
                            force_process,
                            str(input_folder),
                        ),
                    )
                    process_futures.append((process_future, tracking_rel_path))
            except Exception as e:
                logger.error(f"Error in matching: {str(e)}")

        logger.info(f"Matching complete. Processing {files_to_process} files...")

        # Wait for all processing to complete
        for future, rel_path in process_futures:
            try:
                success = future.result()
                if not success:
                    logger.error(f"Failed to process file: {rel_path}")
            except Exception as e:
                logger.error(f"Error processing file {rel_path}: {str(e)}")
            finally:
                processed_files += 1
                if progress_callback:
                    progress_callback(processed_files, files_to_process, "processing")
