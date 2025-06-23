"""Core audio processing functionality."""

import os
import ffmpeg
import logging
import json
import gc
import psutil
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from ..utils.loaders import load_audio_modules, load_audio_processing_modules
from ..utils.constants import SUPPORTED_FORMATS, SUPPORTED_CODECS

# Memory management constants
MAX_AUDIO_DURATION_SECONDS = 300  # Limit audio loading to 5 minutes
MEMORY_THRESHOLD_GB = 1  # Alert when available memory drops below 500MB
BATCH_SIZE = 10  # Process files in smaller batches to prevent memory buildup
CLEANUP_AFTER_EACH_FILE = True  # Clean memory after each file (can be disabled for speed)


def set_cleanup_per_file(enabled: bool):
    """Set whether to perform aggressive memory cleanup after each file."""
    global CLEANUP_AFTER_EACH_FILE
    CLEANUP_AFTER_EACH_FILE = enabled


def get_cleanup_per_file() -> bool:
    """Get current cleanup per file setting."""
    return CLEANUP_AFTER_EACH_FILE


def check_memory_usage():
    """Check current memory usage and trigger cleanup if needed."""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    if available_gb < MEMORY_THRESHOLD_GB:
        logging.warning(f"Low memory detected: {available_gb:.2f}GB available. Triggering garbage collection.")
        gc.collect()
        # Force cleanup of numpy arrays
        import numpy as np
        np.seterr(all='ignore')  # Suppress numpy warnings during cleanup
        return True
    return False


def safe_load_audio(audio_path: str, sr: int = 22050, max_duration: float = None) -> Tuple[np.ndarray, int]:
    """Safely load audio with memory management."""
    librosa, _, _ = load_audio_processing_modules()
    if not librosa:
        raise ImportError("Required module not found. Please install librosa")
    
    try:
        # Limit duration to prevent memory issues
        duration = max_duration or MAX_AUDIO_DURATION_SECONDS
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        return y, sr
    except Exception as e:
        logging.error(f"Error loading audio {audio_path}: {str(e)}")
        raise


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
        preserve_original_quality,
    ) = file_info

    try:
        # Check memory before processing each file
        check_memory_usage()
        
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

        # Get audio metadata and duration with memory check
        try:
            check_memory_usage()  # Check before ffmpeg operations
            probe = ffmpeg.probe(str(input_path))
            duration = float(probe["streams"][0]["duration"])
            
            # Get original codec and bitrate
            original_codec = probe["streams"][0].get("codec_name", "")
            original_bitrate = probe["streams"][0].get("bit_rate", "")
            
            # If we have original codec and bitrate, use them instead of quality preset
            if preserve_original_quality and original_codec and original_bitrate:
                quality = {
                    "acodec": original_codec,
                    "audio_bitrate": original_bitrate
                }
                logger.info(f"Using original codec: {original_codec} with bitrate: {original_bitrate}")
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
            
            # Aggressive cleanup after FFmpeg operation
            if CLEANUP_AFTER_EACH_FILE:
                # Clean up FFmpeg objects
                del output_stream, stream, input_stream
                gc.collect()
                
        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error processing {filename}: {error_message}")
            if os.path.exists(str(temp_output_path)):
                os.remove(str(temp_output_path))
            # Clean up on error
            if CLEANUP_AFTER_EACH_FILE:
                try:
                    del output_stream, stream, input_stream
                except:
                    pass
                gc.collect()
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

                # Aggressive memory cleanup after each file
                if CLEANUP_AFTER_EACH_FILE:
                    gc.collect()
                    memory = psutil.virtual_memory()
                    logger.debug(f"Memory after processing {filename}: {memory.percent:.1f}% used ({memory.available / (1024**3):.1f}GB available)")

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
        # Cleanup on error too
        if CLEANUP_AFTER_EACH_FILE:
            gc.collect()
        return False
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        if os.path.exists(str(temp_output_path)):
            os.remove(str(temp_output_path))
        # Cleanup on error too
        if CLEANUP_AFTER_EACH_FILE:
            gc.collect()
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
    try:
        # Use safe loading with duration limit for templates
        y, sr = safe_load_audio(audio_path, sr=sr, max_duration=60)  # Limit templates to 60 seconds
        
        # Check memory before processing
        check_memory_usage()
        
        # Compute mel spectrogram
        librosa, _, _ = load_audio_processing_modules()
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        
        # Convert to log scale
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Clean up intermediate variables
        del y, mel_spec
        gc.collect()
        
        return mel_db
    except Exception as e:
        logging.error(f"Error computing fingerprint for {audio_path}: {str(e)}")
        raise


def find_intro_match(
    audio_path: str, intro_templates: Dict[str, np.ndarray], threshold: float = 0.85
) -> Tuple[Optional[str], float, float]:
    """Find if any of the intro templates match the beginning of the audio file."""
    try:
        # Load only the beginning of the target audio (first 2 minutes should be enough)
        y_target, sr = safe_load_audio(audio_path, sr=22050, max_duration=120)
        
        # Check memory before processing
        check_memory_usage()
        
        librosa, _, _ = load_audio_processing_modules()
        mel_target = librosa.feature.melspectrogram(y=y_target, sr=sr)
        mel_db_target = librosa.power_to_db(mel_target, ref=np.max)
        
        # Clean up target audio data
        del y_target, mel_target
        gc.collect()
        
        best_match = None
        best_score = -1
        best_duration = 0
        
        for intro_name, intro_fingerprint in intro_templates.items():
            try:
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
                    
            except Exception as e:
                logging.warning(f"Error matching template {intro_name}: {str(e)}")
                continue
        
        # Clean up
        del mel_db_target
        gc.collect()
        
        if best_match:
            # Convert mel spectrogram frames to seconds
            duration_seconds = librosa.frames_to_time(best_duration, sr=sr)
            return best_match, best_score, duration_seconds
        
        return None, 0, 0
        
    except Exception as e:
        logging.error(f"Error finding intro match for {audio_path}: {str(e)}")
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
    preserve_original_quality: bool = False,
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
    
    # Limit max workers based on available memory
    if max_workers is None:
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        # Use fewer workers if memory is limited
        max_workers = min(4, max(1, int(available_gb // 2)))
        logger.info(f"Auto-setting max_workers to {max_workers} based on available memory ({available_gb:.1f}GB)")

    # Process files in batches to prevent memory buildup
    batch_size = min(BATCH_SIZE, len(audio_files))
    logger.info(f"Processing {total_files} files in batches of {batch_size}")

    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(audio_files) + batch_size - 1)//batch_size}")
        
        # Check memory before processing each batch
        if check_memory_usage():
            logger.warning("Low memory detected before batch processing. Consider reducing batch size or max_workers.")
        
        # Process files with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for f in batch_files:
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
                        preserve_original_quality,
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
        
        # Force garbage collection between batches
        gc.collect()
        
        # Log memory usage
        memory = psutil.virtual_memory()
        logger.info(f"Batch complete. Memory usage: {memory.percent:.1f}% ({memory.available / (1024**3):.1f}GB available)")

    logger.info(f"Processing complete. Processed {processed_files}/{total_files} files.")


def remove_detected_intros(
    input_folder: str,
    template_folder: str,
    make_backup: bool = False,
    max_workers: Optional[tuple[int, int]] = None,  # (match_workers, process_workers)
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    recursive: bool = True,
    quality: Optional[Dict] = None,
    match_threshold: float = 0.85,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    backup_dir: Optional[str] = None,
    tracking_file: Optional[str] = None,
    force_process: bool = False,
    preserve_original_quality: bool = False,
) -> None:
    """Remove intros by matching against template files."""
    logger = logging.getLogger("audio_trimmer")

    # Check for required modules
    librosa, sf, signal = load_audio_processing_modules()
    if not all([librosa, sf, signal]):
        raise ImportError(
            "Required modules not found. Please install: pip install librosa scipy"
        )

    # Get worker counts with memory considerations
    if max_workers is None:
        # Calculate default worker counts
        import multiprocessing
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        total_cpus = multiprocessing.cpu_count()
        available_cpus = max(1, total_cpus - 2)  # Reserve 2 CPUs by default
        
        # Reduce workers if memory is limited
        if available_gb < 4.0:  # Less than 4GB available
            available_cpus = min(available_cpus, 2)
            logger.warning(f"Limited memory detected ({available_gb:.1f}GB). Reducing CPU usage to prevent memory issues.")
            
        match_workers = max(1, available_cpus // 3)
        process_workers = max(1, available_cpus - match_workers)
    else:
        match_workers, process_workers = max_workers

    logger.info(
        f"Using {match_workers} workers for matching and {process_workers} workers for processing"
    )

    # Learn intro templates with memory management
    logger.info("Learning intro templates...")
    try:
        intro_templates = learn_intro_templates(template_folder)
        
        if not intro_templates:
            raise ValueError(f"No valid intro templates found in {template_folder}")

        logger.info(f"Learned {len(intro_templates)} intro templates")
        
        # Check memory usage after loading templates
        check_memory_usage()
        
    except Exception as e:
        logger.error(f"Error learning templates: {str(e)}")
        raise

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

            # Find intro match with memory monitoring
            check_memory_usage()
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

    # Process files in batches to manage memory
    batch_size = min(BATCH_SIZE * 2, len(audio_files))  # Slightly larger batches for matching
    logger.info(f"Processing {total_files} files in batches of {batch_size}")
    
    for batch_start in range(0, len(audio_files), batch_size):
        batch_end = min(batch_start + batch_size, len(audio_files))
        batch_files = audio_files[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(audio_files) + batch_size - 1)//batch_size}")
        
        # Check memory before each batch
        if check_memory_usage():
            logger.warning("Low memory detected. Consider reducing batch size or worker count.")

        # Start matching files in parallel while processing
        with ThreadPoolExecutor(
            max_workers=match_workers
        ) as match_executor, ThreadPoolExecutor(
            max_workers=process_workers
        ) as process_executor:

            # Submit batch files for matching
            match_futures = [
                match_executor.submit(check_file_match, f) for f in batch_files
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
                                preserve_original_quality,
                            ),
                        )
                        process_futures.append((process_future, tracking_rel_path))
                except Exception as e:
                    logger.error(f"Error in matching: {str(e)}")

            # Wait for all processing in this batch to complete
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
        
        # Force garbage collection between batches
        gc.collect()
        
        # Log memory usage
        memory = psutil.virtual_memory()
        logger.info(f"Batch complete. Memory usage: {memory.percent:.1f}% ({memory.available / (1024**3):.1f}GB available)")

    logger.info(f"Template matching complete. Processed {files_to_process} matching files out of {total_files} total files.")
