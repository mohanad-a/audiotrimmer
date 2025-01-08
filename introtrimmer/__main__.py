"""Main entry point for the audio intro trimmer."""

import argparse
import logging
import sys
from pathlib import Path

from .core.audio_processor import remove_audio_duration, remove_detected_intros
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

    try:
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
            )
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
