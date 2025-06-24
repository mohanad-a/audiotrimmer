"""Core audio processing functionality."""

import os
import ffmpeg
import logging
import json
import gc
import psutil
import signal
import platform
import tempfile
import multiprocessing
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Callable
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ..utils.loaders import load_audio_modules, load_audio_processing_modules
from ..utils.constants import SUPPORTED_FORMATS, SUPPORTED_CODECS

# Memory management constants
MAX_AUDIO_DURATION_SECONDS = 300  # Limit audio loading to 5 minutes
MEMORY_THRESHOLD_GB = 1  # Alert when available memory drops below 500MB
BATCH_SIZE = 10  # Process files in smaller batches to prevent memory buildup

# Platform-specific settings
IS_MACOS = platform.system() == 'Darwin'

# Main process signal handler for bus errors
def main_bus_error_handler(signum, frame):
    """Handle bus errors gracefully in the main process."""
    logging.error(f"Main process bus error detected (Signal {signum}). Cleaning up and exiting...")
    # Force garbage collection
    gc.collect()
    # Exit with error code
    os._exit(1)

# Install signal handler for main process
if hasattr(signal, 'SIGBUS'):
    signal.signal(signal.SIGBUS, main_bus_error_handler)
if hasattr(signal, 'SIGSEGV'):
    signal.signal(signal.SIGSEGV, main_bus_error_handler)


def get_optimal_worker_count(available_memory_gb: float, is_macos: bool = False, override_workers: int = None) -> int:
    """Calculate optimal worker count based on system resources."""
    if override_workers is not None:
        # Allow manual override but still cap at reasonable limits
        return min(max(1, override_workers), multiprocessing.cpu_count())
    
    cpu_count = multiprocessing.cpu_count()
    
    # Base worker count on CPU cores
    if is_macos:
        # macOS: Be less conservative but still cautious
        # Allow up to 50% of cores or 4 workers, whichever is smaller
        max_workers = max(2, min(4, cpu_count // 2))
    else:
        max_workers = max(1, cpu_count - 1)  # Reserve 1 CPU for system
    
    # Adjust based on available memory (assume ~500MB per worker)
    memory_based_workers = max(1, int(available_memory_gb // 0.5))
    
    # Use the minimum of CPU and memory constraints
    optimal_workers = min(max_workers, memory_based_workers)
    
    # Cap at 8 workers to avoid overwhelming the system (increased from 4)
    return min(optimal_workers, 8)


def init_worker():
    """Initialize worker process with signal handlers and environment."""
    # Set up signal handlers for worker processes
    def worker_bus_error_handler(signum, frame):
        logging.error(f"Worker process bus error (Signal {signum}). Exiting worker gracefully.")
        gc.collect()
        os._exit(1)
    
    # Install signal handlers for worker processes
    if hasattr(signal, 'SIGBUS'):
        signal.signal(signal.SIGBUS, worker_bus_error_handler)
    if hasattr(signal, 'SIGSEGV'):
        signal.signal(signal.SIGSEGV, worker_bus_error_handler)
    
    # Configure environment for worker processes on macOS
    if IS_MACOS:
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'


def process_file_wrapper(args):
    """Wrapper function for multiprocessing that handles individual file processing."""
    try:
        # Initialize logging for the worker process
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        # Call the actual process_file function
        result = process_file(args)
        
        # Force cleanup in worker process
        gc.collect()
        return result
        
    except Exception as e:
        logging.error(f"Worker process error: {str(e)}")
        # Force cleanup on error
        gc.collect()
        return False


def process_batch_multiprocess(batch_tasks, max_workers=None, batch_id=None):
    """Process a batch of files using multiprocessing for better isolation."""
    logger = logging.getLogger("audio_trimmer")
    
    if batch_id:
        logger.info(f"Processing batch {batch_id} with {len(batch_tasks)} files using {max_workers} processes")
    
    successful_tasks = 0
    failed_tasks = 0
    
    try:
        # Use ProcessPoolExecutor for better isolation
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker
        ) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_file_wrapper, task): task 
                for task in batch_tasks
            }
            
            # Process results as they complete
            for future in future_to_task:
                try:
                    result = future.result(timeout=600)  # 10 minute timeout per file
                    if result:
                        successful_tasks += 1
                    else:
                        failed_tasks += 1
                        logger.warning(f"File processing returned False for a task in batch {batch_id}")
                except Exception as e:
                    failed_tasks += 1
                    task = future_to_task[future]
                    logger.error(f"Process failed for task in batch {batch_id}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Batch processing error for batch {batch_id}: {str(e)}")
        failed_tasks = len(batch_tasks)
    
    logger.info(f"Batch {batch_id} complete: {successful_tasks} successful, {failed_tasks} failed")
    return successful_tasks, failed_tasks


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


def safe_ffmpeg_probe(audio_path: str, max_retries: int = 3) -> Optional[Dict]:
    """Safely probe audio file with retries and error handling."""
    for attempt in range(max_retries):
        try:
            # On macOS, add specific flags to prevent bus errors
            if IS_MACOS:
                # Use a separate process to isolate potential crashes
                import subprocess
                cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=duration,codec_name,bit_rate', '-of', 'json', str(audio_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    import json
                    probe = json.loads(result.stdout)
                else:
                    raise ffmpeg.Error('ffprobe', result.stdout, result.stderr)
            else:
                probe = ffmpeg.probe(str(audio_path))
            return probe
        except ffmpeg.Error as e:
            logging.warning(f"FFmpeg probe attempt {attempt + 1} failed for {audio_path}: {e}")
            if attempt == max_retries - 1:
                raise
            # Wait before retry
            import time
            time.sleep(0.1)
        except Exception as e:
            logging.error(f"Unexpected error during probe attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise
    return None


def safe_load_audio(audio_path: str, sr: int = 22050, max_duration: float = None) -> Tuple[np.ndarray, int]:
    """Safely load audio with memory management and bus error prevention."""
    librosa, _, _ = load_audio_processing_modules()
    if not librosa:
        raise ImportError("Required module not found. Please install librosa")
    
    try:
        # Limit duration to prevent memory issues
        duration = max_duration or MAX_AUDIO_DURATION_SECONDS
        
        # On macOS, use additional safety measures
        if IS_MACOS:
            # Set NumPy to use single-threaded BLAS to prevent bus errors
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
            os.environ['OMP_NUM_THREADS'] = '1'
        
        # Load with offset to prevent bus errors on certain files
        y, sr = librosa.load(audio_path, sr=sr, duration=duration, offset=0.0)
        
        # Ensure array is properly aligned for macOS
        if IS_MACOS and not y.flags.aligned:
            y = np.copy(y)
        
        return y, sr
    except Exception as e:
        logging.error(f"Error loading audio {audio_path}: {str(e)}")
        raise


def safe_ffmpeg_run(output_stream, max_retries: int = 2) -> bool:
    """Safely run FFmpeg with retries and bus error prevention."""
    for attempt in range(max_retries):
        try:
            # Use ffmpeg-python's run method with additional safety on macOS
            if IS_MACOS:
                # On macOS, add environment variables to prevent bus errors
                old_env = {}
                env_vars = {
                    'OPENBLAS_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1', 
                    'NUMEXPR_NUM_THREADS': '1',
                    'OMP_NUM_THREADS': '1'
                }
                
                # Backup and set environment variables
                for key, value in env_vars.items():
                    old_env[key] = os.environ.get(key)
                    os.environ[key] = value
                
                try:
                    # Run with timeout and capture output
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("FFmpeg process timed out")
                    
                    # Set up timeout signal (5 minutes) - only if not in multiprocessing context
                    timeout_set = False
                    old_handler = None
                    try:
                        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(300)  # 5 minutes
                        timeout_set = True
                    except (ValueError, OSError):
                        # Signal handling not available in this context (e.g., worker process)
                        pass
                    
                    try:
                        output_stream.run(capture_stdout=True, capture_stderr=True)
                    finally:
                        if timeout_set:
                            signal.alarm(0)  # Cancel the alarm
                            signal.signal(signal.SIGALRM, old_handler)
                
                finally:
                    # Restore environment variables
                    for key, value in old_env.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
            else:
                # Non-macOS: use standard approach
                output_stream.run(capture_stdout=True, capture_stderr=True)
            
            return True
        except ffmpeg.Error as e:
            logging.warning(f"FFmpeg run attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            # Clean up and retry
            gc.collect()
        except TimeoutError as e:
            logging.error(f"FFmpeg run attempt {attempt + 1} timed out: {e}")
            if attempt == max_retries - 1:
                raise
        except Exception as e:
            logging.error(f"Unexpected error during FFmpeg run attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise
    return False


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
            probe = safe_ffmpeg_probe(str(input_path))
            if not probe:
                logger.error(f"Failed to probe {filename}: Unable to get file information")
                return False
            
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
            logger.error(f"Failed to probe {filename}: {e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else str(e)}")
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
            if not safe_ffmpeg_run(output_stream):
                logger.error(f"FFmpeg processing failed for {filename}")
                if os.path.exists(str(temp_output_path)):
                    os.remove(str(temp_output_path))
                return False
        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else str(e)
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


def safe_correlation(x, y):
    """Safely compute correlation to prevent bus errors."""
    try:
        # Ensure arrays are properly aligned
        if IS_MACOS:
            if not x.flags.aligned:
                x = np.copy(x)
            if not y.flags.aligned:
                y = np.copy(y)
        
        # Flatten arrays to prevent shape issues
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # Ensure arrays have the same length
        min_len = min(len(x_flat), len(y_flat))
        x_flat = x_flat[:min_len]
        y_flat = y_flat[:min_len]
        
        # Compute correlation with error handling
        if len(x_flat) == 0 or len(y_flat) == 0:
            return 0.0
        
        # Use manual correlation calculation to avoid numpy bus errors
        x_mean = np.mean(x_flat)
        y_mean = np.mean(y_flat)
        
        numerator = np.sum((x_flat - x_mean) * (y_flat - y_mean))
        x_var = np.sum((x_flat - x_mean) ** 2)
        y_var = np.sum((y_flat - y_mean) ** 2)
        
        if x_var == 0 or y_var == 0:
            return 0.0
        
        correlation = numerator / np.sqrt(x_var * y_var)
        return correlation
        
    except Exception as e:
        logging.warning(f"Error in correlation calculation: {e}")
        return 0.0


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
                score = safe_correlation(intro_fingerprint, target_segment)
                
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


def match_intro_wrapper(args):
    """Wrapper function for multiprocessing intro matching."""
    try:
        # Initialize logging for the worker process
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        f, intro_templates, match_threshold, input_folder, tracking_file, force_process = args
        
        # Import required modules in worker process
        from pathlib import Path
        
        input_path = Path(f).resolve()
        input_folder = Path(input_folder).resolve()
        tracking_rel_path = get_relative_path(str(input_path), str(input_folder))

        # Check if already processed
        if not force_process and tracking_file:
            processed_files_set = load_processed_files(tracking_file)
            if tracking_rel_path in processed_files_set:
                return None

        # Find intro match with memory monitoring
        check_memory_usage()
        intro_match, score, duration = find_intro_match(
            str(input_path), intro_templates, match_threshold
        )

        if not intro_match:
            return None

        return (input_path, duration, tracking_rel_path, intro_match, score)
        
    except Exception as e:
        logging.error(f"Worker matching error: {str(e)}")
        return None


def process_matching_batch(match_tasks, max_workers=None, batch_id=None):
    """Process a batch of intro matching using multiprocessing."""
    logger = logging.getLogger("audio_trimmer")
    
    if batch_id:
        logger.info(f"Matching batch {batch_id} with {len(match_tasks)} files using {max_workers} processes")
    
    matched_results = []
    
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker
        ) as executor:
            # Submit all matching tasks
            futures = [executor.submit(match_intro_wrapper, task) for task in match_tasks]
            
            # Process results as they complete
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per match
                    if result:
                        matched_results.append(result)
                except Exception as e:
                    logger.error(f"Matching process failed: {str(e)}")
    
    except Exception as e:
        logger.error(f"Batch matching error for batch {batch_id}: {str(e)}")
    
    logger.info(f"Matching batch {batch_id} complete: {len(matched_results)} matches found")
    return matched_results


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
        max_workers = get_optimal_worker_count(available_gb, IS_MACOS)
        logger.info(f"Auto-setting max_workers to {max_workers} based on available memory ({available_gb:.1f}GB) and platform (macOS: {IS_MACOS})")
    elif isinstance(max_workers, int):
        # Manual override - validate the count
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        recommended_workers = get_optimal_worker_count(available_gb, IS_MACOS)
        if max_workers > recommended_workers:
            logger.warning(f"Manual override: Using {max_workers} workers. Recommended: {recommended_workers} for your system. Monitor for stability issues.")
    else:
        # Validate user-provided worker count
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        recommended_workers = get_optimal_worker_count(available_gb, IS_MACOS)
        if max_workers > recommended_workers:
            logger.warning(f"Requested {max_workers} workers exceeds recommended {recommended_workers} for your system. Consider reducing for better stability.")

    # Process files in batches to prevent memory buildup
    batch_size = min(BATCH_SIZE, len(audio_files))
    logger.info(f"Processing {total_files} files in batches of {batch_size} using multiprocessing")

    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i + batch_size]
        batch_id = f"{i//batch_size + 1}/{(len(audio_files) + batch_size - 1)//batch_size}"
        
        # Check memory before processing each batch
        if check_memory_usage():
            logger.warning("Low memory detected before batch processing. Consider reducing batch size or max_workers.")
        
        # Prepare batch tasks
        batch_tasks = []
        for f in batch_files:
            task = (
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
            )
            batch_tasks.append(task)
        
        # Process batch using multiprocessing
        successful, failed = process_batch_multiprocess(
            batch_tasks, 
            max_workers=max_workers, 
            batch_id=batch_id
        )
        
        processed_files += successful
        if progress_callback:
            progress_callback(processed_files, total_files)
        
        # Force garbage collection between batches
        gc.collect()
        
        # Log memory usage
        memory = psutil.virtual_memory()
        logger.info(f"Batch {batch_id} complete. Memory usage: {memory.percent:.1f}% ({memory.available / (1024**3):.1f}GB available)")

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
        # Calculate default worker counts using the optimal calculation
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        optimal_workers = get_optimal_worker_count(available_gb, IS_MACOS)
        
        # Split workers between matching and processing (1/3 for matching, 2/3 for processing)
        match_workers = max(1, optimal_workers // 3)
        process_workers = max(1, optimal_workers - match_workers)
        
        logger.info(f"Auto-setting workers: {match_workers} for matching, {process_workers} for processing (total: {optimal_workers})")
    else:
        match_workers, process_workers = max_workers
        # Validate user-provided worker counts
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        recommended_total = get_optimal_worker_count(available_gb, IS_MACOS)
        total_requested = match_workers + process_workers
        if total_requested > recommended_total:
            logger.warning(f"Manual override: Using {total_requested} total workers. Recommended: {recommended_total} for your system. Monitor for stability issues.")

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

    # Process files in batches to manage memory
    batch_size = min(BATCH_SIZE * 2, len(audio_files))  # Slightly larger batches for matching
    logger.info(f"Processing {total_files} files in batches of {batch_size} using multiprocessing")
    
    for batch_start in range(0, len(audio_files), batch_size):
        batch_end = min(batch_start + batch_size, len(audio_files))
        batch_files = audio_files[batch_start:batch_end]
        batch_id = f"{batch_start//batch_size + 1}/{(len(audio_files) + batch_size - 1)//batch_size}"
        
        # Check memory before each batch
        if check_memory_usage():
            logger.warning("Low memory detected. Consider reducing batch size or worker count.")

        # Prepare matching tasks
        match_tasks = []
        for f in batch_files:
            match_task = (f, intro_templates, match_threshold, str(input_folder), tracking_file, force_process)
            match_tasks.append(match_task)
        
        # Process matching batch
        matched_results = process_matching_batch(
            match_tasks, 
            max_workers=match_workers, 
            batch_id=f"{batch_id}-match"
        )
        
        matched_files += len(batch_files)
        if progress_callback:
            progress_callback(matched_files, total_files, "matching")
        
        # Prepare processing tasks for matched files
        if matched_results:
            process_tasks = []
            for input_path, duration, tracking_rel_path, intro_match, score in matched_results:
                logger.info(
                    f"Found intro match in {tracking_rel_path}: {intro_match} (score: {score:.2f}, duration: {duration:.2f}s)"
                )
                
                process_task = (
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
                )
                process_tasks.append(process_task)
            
            # Process the matched files
            if process_tasks:
                files_to_process += len(process_tasks)
                successful, failed = process_batch_multiprocess(
                    process_tasks, 
                    max_workers=process_workers, 
                    batch_id=f"{batch_id}-process"
                )
                
                processed_files += successful
                if progress_callback:
                    progress_callback(processed_files, files_to_process, "processing")
        
        # Force garbage collection between batches
        gc.collect()
        
        # Log memory usage
        memory = psutil.virtual_memory()
        logger.info(f"Batch {batch_id} complete. Memory usage: {memory.percent:.1f}% ({memory.available / (1024**3):.1f}GB available)")

    logger.info(f"Template matching complete. Processed {files_to_process} matching files out of {total_files} total files.")
