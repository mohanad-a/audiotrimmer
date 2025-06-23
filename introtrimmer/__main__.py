"""Main entry point for the audio intro trimmer."""

import argparse
import logging
import sys
from pathlib import Path

from .core.audio_processor import remove_audio_duration, remove_detected_intros, set_cleanup_per_file, set_ffmpeg_threads
from .utils.constants import QUALITY_PRESETS


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Audio Intro Trimmer")

    # Add GUI mode argument
    parser.add_argument("--gui", action="store_true", help="Launch the GUI interface")

    # CLI arguments
    parser.add_argument("-i", "--input", help="Input folder containing audio files")
    parser.add_argument(
        "-d", "--duration", type=float, help="Duration to remove (in seconds)"
    )
    parser.add_argument(
        "-t", "--template", help="Folder containing intro template files"
    )
    parser.add_argument("-o", "--output", help="Output folder for processed files")
    parser.add_argument(
        "-b", "--backup", action="store_true", help="Create backup of original files"
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Process files in subdirectories"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--from-end",
        action="store_true",
        help="Remove duration from end instead of start",
    )
    parser.add_argument(
        "--quality",
        choices=list(QUALITY_PRESETS.keys()),
        default="high",
        help="Audio quality preset",
    )
    parser.add_argument(
        "--preserve-original-quality",
        action="store_true",
        help="Preserve original codec and bitrate of input files",
    )
    parser.add_argument(
        "--smart-trim",
        action="store_true",
        help="Use silence detection for smarter trimming",
    )
    parser.add_argument(
        "--fade", type=float, help="Add fade in/out effect of specified duration"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Matching threshold for template mode",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Memory management arguments
    parser.add_argument(
        "--cleanup-per-file",
        action="store_true",
        default=True,
        help="Clean memory after each file (default: True, use --no-cleanup-per-file to disable)",
    )
    parser.add_argument(
        "--no-cleanup-per-file",
        action="store_false",
        dest="cleanup_per_file",
        help="Disable memory cleanup after each file (faster but uses more memory)",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable fast mode (larger batches, less frequent cleanup, optimized for speed)",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=2,
        help="Number of threads for FFmpeg operations (default: 2)",
    )

    # CPU allocation arguments
    import multiprocessing

    total_cpus = multiprocessing.cpu_count()

    parser.add_argument(
        "--reserved-cpus",
        type=int,
        default=2,
        help=f"Number of CPUs to reserve for system (0-{total_cpus-1}, default: 2)",
    )
    parser.add_argument(
        "--match-cpu-ratio",
        type=int,
        default=33,
        help="Percentage of available CPUs to use for matching (10-90, default: 33)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Launch GUI if requested
    if args.gui:
        try:
            from .gui.main_window import create_main_window

            create_main_window()
            return
        except ImportError as e:
            logging.error(f"Could not start GUI: {str(e)}")
            logging.error("Please ensure tkinter is installed on your system.")
            sys.exit(1)

    # Validate CLI arguments
    if not args.input:
        parser.error("Input folder is required for CLI mode")

    if not args.duration and not args.template:
        parser.error("Either --duration or --template must be specified")

    if args.duration and args.template:
        parser.error("Cannot use both --duration and --template at the same time")

    # Validate CPU allocation arguments
    if args.reserved_cpus < 0 or args.reserved_cpus >= total_cpus:
        parser.error(f"Reserved CPUs must be between 0 and {total_cpus-1}")
    if args.match_cpu_ratio < 10 or args.match_cpu_ratio > 90:
        parser.error("Match CPU ratio must be between 10 and 90")

    try:
        # Calculate CPU allocation for template mode
        if args.template:
            available_cpus = max(1, total_cpus - args.reserved_cpus)
            match_workers = max(1, int(available_cpus * args.match_cpu_ratio / 100))
            process_workers = max(1, available_cpus - match_workers)
            logging.info(
                f"CPU allocation: {match_workers} matching, {process_workers} processing, {args.reserved_cpus} reserved"
            )

        # Set memory cleanup behavior
        set_cleanup_per_file(args.cleanup_per_file)
        logging.info(f"Memory cleanup per file: {'enabled' if args.cleanup_per_file else 'disabled'}")

        # Configure fast mode if enabled
        if args.fast_mode:
            from .core.audio_processor import ADAPTIVE_CLEANUP, FAST_BATCH_SIZE, FFMPEG_THREAD_COUNT
            # Override settings for maximum speed
            set_cleanup_per_file(False)  # Disable per-file cleanup
            logging.info("Fast mode enabled: Using optimized settings for maximum speed")
            logging.info("- Per-file cleanup: disabled")
            logging.info("- Batch sizes: large")
            logging.info("- Memory monitoring: reduced frequency")
        
        # Set FFmpeg thread count
        if hasattr(args, 'ffmpeg_threads'):
            from .core.audio_processor import set_ffmpeg_threads
            set_ffmpeg_threads(args.ffmpeg_threads)
            logging.info(f"FFmpeg threads: {args.ffmpeg_threads}")

        # Template mode
        if args.template:
            remove_detected_intros(
                input_folder=args.input,
                template_folder=args.template,
                make_backup=args.backup,
                output_dir=args.output,
                dry_run=args.dry_run,
                recursive=args.recursive,
                quality=QUALITY_PRESETS[args.quality],
                match_threshold=args.threshold,
                max_workers=(match_workers, process_workers),
            )
        # Basic mode
        else:
            remove_audio_duration(
                input_folder=args.input,
                duration_to_remove=args.duration,
                make_backup=args.backup,
                output_dir=args.output,
                dry_run=args.dry_run,
                recursive=args.recursive,
                from_end=args.from_end,
                quality=QUALITY_PRESETS[args.quality],
                smart_trim=args.smart_trim,
                fade_duration=args.fade,
                preserve_original_quality=args.preserve_original_quality,
            )
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
