"""Main GUI window for the Audio Intro Remover."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import queue
import logging
import os
import multiprocessing

from ..utils.constants import SUPPORTED_FORMATS, QUALITY_PRESETS
from ..core.audio_processor import process_file, preview_audio


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Audio Intro Remover")
        self.root.geometry("800x600")

        # Set default tracking file in script root directory
        script_dir = Path(
            __file__
        ).parent.parent.parent  # Go up to IntroRemover directory
        self.default_tracking_file = str(script_dir / "processed_files.json")

        # Variables
        # Basic mode variables
        self.basic_input_folder = tk.StringVar()
        self.duration = tk.StringVar(value="8.65")
        self.from_end = tk.BooleanVar(value=False)

        # Template mode variables
        self.template_input_folder = tk.StringVar()
        self.template_folder = tk.StringVar()
        self.match_threshold = tk.StringVar(value="0.85")

        # CPU allocation variables
        total_cpus = multiprocessing.cpu_count()
        self.reserved_cpus = tk.StringVar(value="2")  # Default reserve 2 CPUs
        self.match_cpu_ratio = tk.StringVar(value="33")  # Default 33% for matching

        # Common variables
        self.output_folder = tk.StringVar()
        self.backup_folder = tk.StringVar()
        self.tracking_file = tk.StringVar(
            value=self.default_tracking_file
        )  # Set default tracking file
        self.force_process = tk.BooleanVar(value=False)
        self.quality = tk.StringVar(value="high")
        self.fade_duration = tk.StringVar()
        self.make_backup = tk.BooleanVar(value=True)
        self.recursive = tk.BooleanVar(value=True)
        self.smart_trim = tk.BooleanVar(value=False)

        self._create_widgets()
        self._create_layout()

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Notebook for different modes
        self.notebook = ttk.Notebook(self.root)

        # Basic mode frame
        self.basic_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.basic_frame, text="Basic Mode")

        # Template mode frame
        self.template_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.template_frame, text="Template Mode")

        # Advanced options frame
        self.advanced_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.advanced_frame, text="Advanced Options")

        # Create widgets for basic mode
        self._create_basic_widgets()

        # Create widgets for template mode
        self._create_template_widgets()

        # Create widgets for advanced options
        self._create_advanced_widgets()

        # Create common controls
        self._create_common_controls()

    def _create_basic_widgets(self):
        """Create widgets for basic mode."""
        # Input folder selection
        input_frame = ttk.LabelFrame(
            self.basic_frame, text="Input Selection", padding="5"
        )
        input_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

        ttk.Label(input_frame, text="Input Folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.basic_input_folder, width=50).grid(
            row=0, column=1, padx=5
        )
        ttk.Button(
            input_frame, text="Browse", command=self._select_basic_input_folder
        ).grid(row=0, column=2)

        # Settings frame
        settings_frame = ttk.LabelFrame(
            self.basic_frame, text="Trim Settings", padding="5"
        )
        settings_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

        # Duration input
        ttk.Label(settings_frame, text="Duration to Remove (seconds):").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(settings_frame, textvariable=self.duration, width=10).grid(
            row=0, column=1, sticky="w", padx=5
        )

        # From end checkbox
        ttk.Checkbutton(
            settings_frame,
            text="Remove from end instead of start",
            variable=self.from_end,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=5)

        # Help text
        help_frame = ttk.LabelFrame(self.basic_frame, text="Instructions", padding="5")
        help_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        help_text = ttk.Label(
            help_frame,
            text=(
                "Basic Mode:\n"
                "1. Select the input folder containing audio files you want to process\n"
                "2. Enter the duration (in seconds) you want to remove\n"
                "3. Choose whether to remove from the start or end of files\n"
                "4. Use advanced options tab for quality settings and fade effects\n"
                "5. Click 'Process Files' to start or 'Dry Run' to test"
            ),
            justify=tk.LEFT,
            wraplength=500,
        )
        help_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _create_template_widgets(self):
        """Create widgets for template mode."""
        # Input folder selection
        input_frame = ttk.LabelFrame(
            self.template_frame, text="Input Selection", padding="5"
        )
        input_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

        ttk.Label(input_frame, text="Input Folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.template_input_folder, width=50).grid(
            row=0, column=1, padx=5
        )
        ttk.Button(
            input_frame, text="Browse", command=self._select_template_input_folder
        ).grid(row=0, column=2)

        # Template folder selection
        template_frame = ttk.LabelFrame(
            self.template_frame, text="Template Selection", padding="5"
        )
        template_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

        ttk.Label(template_frame, text="Template Folder:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(template_frame, textvariable=self.template_folder, width=50).grid(
            row=0, column=1, padx=5
        )
        ttk.Button(
            template_frame, text="Browse", command=self._select_template_folder
        ).grid(row=0, column=2)

        # Settings frame
        settings_frame = ttk.LabelFrame(
            self.template_frame, text="Template Settings", padding="5"
        )
        settings_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

        # Match threshold
        ttk.Label(settings_frame, text="Match Threshold:").grid(
            row=0, column=0, sticky="w"
        )
        threshold_entry = ttk.Entry(
            settings_frame, textvariable=self.match_threshold, width=10
        )
        threshold_entry.grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(settings_frame, text="(0.0-1.0, higher = stricter matching)").grid(
            row=0, column=2, sticky="w"
        )

        # Help text
        help_frame = ttk.LabelFrame(
            self.template_frame, text="Instructions", padding="5"
        )
        help_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        help_text = ttk.Label(
            help_frame,
            text=(
                "Template Mode:\n"
                "1. Select the input folder containing audio files to process\n"
                "2. Select the template folder containing your intro audio files\n"
                "3. Adjust match threshold (higher = stricter matching)\n"
                "4. Use advanced options tab for quality settings\n"
                "5. Click 'Process Files' to start or 'Dry Run' to test"
            ),
            justify=tk.LEFT,
            wraplength=500,
        )
        help_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _create_advanced_widgets(self):
        """Create widgets for advanced options."""
        # CPU Settings
        cpu_frame = ttk.LabelFrame(
            self.advanced_frame, text="CPU Settings", padding="5"
        )
        cpu_frame.grid(row=0, column=0, columnspan=4, sticky="ew", pady=5)

        total_cpus = multiprocessing.cpu_count()

        # Reserved CPUs
        ttk.Label(cpu_frame, text="Reserved CPUs:").grid(
            row=0, column=0, sticky="w", padx=5
        )
        reserved_spin = ttk.Spinbox(
            cpu_frame,
            from_=0,
            to=max(0, total_cpus - 1),
            width=5,
            textvariable=self.reserved_cpus,
        )
        reserved_spin.grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(
            cpu_frame, text=f"(0-{total_cpus-1}, higher = more responsive system)"
        ).grid(row=0, column=2, sticky="w", padx=5)

        # Matching CPU ratio
        ttk.Label(cpu_frame, text="Matching CPU %:").grid(
            row=1, column=0, sticky="w", padx=5
        )
        match_spin = ttk.Spinbox(
            cpu_frame, from_=10, to=90, width=5, textvariable=self.match_cpu_ratio
        )
        match_spin.grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(cpu_frame, text="(10-90%, rest used for processing)").grid(
            row=1, column=2, sticky="w", padx=5
        )

        # Quality selection
        ttk.Label(self.advanced_frame, text="Audio Quality:").grid(
            row=1, column=0, sticky="w", pady=5
        )
        quality_combo = ttk.Combobox(
            self.advanced_frame,
            textvariable=self.quality,
            values=list(QUALITY_PRESETS.keys()),
        )
        quality_combo.grid(row=1, column=1, sticky="w", padx=5)

        # Fade duration
        ttk.Label(self.advanced_frame, text="Fade Duration:").grid(
            row=2, column=0, sticky="w"
        )
        ttk.Entry(self.advanced_frame, textvariable=self.fade_duration, width=10).grid(
            row=2, column=1, sticky="w", padx=5
        )
        ttk.Label(self.advanced_frame, text="seconds").grid(row=2, column=2, sticky="w")

        # Smart trim
        ttk.Checkbutton(
            self.advanced_frame,
            text="Use smart trim (detect silence)",
            variable=self.smart_trim,
        ).grid(row=3, column=0, columnspan=2, sticky="w")

        # Output folder
        ttk.Label(self.advanced_frame, text="Output Folder:").grid(
            row=4, column=0, sticky="w"
        )
        ttk.Entry(self.advanced_frame, textvariable=self.output_folder, width=50).grid(
            row=4, column=1, columnspan=2, padx=5
        )
        ttk.Button(
            self.advanced_frame, text="Browse", command=self._select_output_folder
        ).grid(row=4, column=3)

        # Backup folder
        backup_frame = ttk.LabelFrame(
            self.advanced_frame, text="Backup Settings", padding="5"
        )
        backup_frame.grid(row=5, column=0, columnspan=4, sticky="ew", pady=10)

        ttk.Checkbutton(
            backup_frame, text="Create backups", variable=self.make_backup
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(backup_frame, text="Backup Folder:").grid(row=1, column=0, sticky="w")
        ttk.Entry(backup_frame, textvariable=self.backup_folder, width=50).grid(
            row=1, column=1, padx=5
        )
        ttk.Button(
            backup_frame, text="Browse", command=self._select_backup_folder
        ).grid(row=1, column=2)

        # Tracking settings
        tracking_frame = ttk.LabelFrame(
            self.advanced_frame, text="Processing History", padding="5"
        )
        tracking_frame.grid(row=6, column=0, columnspan=4, sticky="ew", pady=10)

        ttk.Label(tracking_frame, text="Tracking File:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(tracking_frame, textvariable=self.tracking_file, width=50).grid(
            row=0, column=1, padx=5
        )
        ttk.Button(
            tracking_frame, text="Browse", command=self._select_tracking_file
        ).grid(row=0, column=2)
        ttk.Button(
            tracking_frame,
            text="Reset to Default",
            command=lambda: self.tracking_file.set(self.default_tracking_file),
        ).grid(row=0, column=3, padx=5)

        ttk.Checkbutton(
            tracking_frame,
            text="Force process all files (ignore tracking)",
            variable=self.force_process,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=5)

    def _create_common_controls(self):
        """Create controls common to all modes."""
        # Create a frame for common controls at the bottom
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Common options
        ttk.Checkbutton(
            control_frame, text="Create backups", variable=self.make_backup
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            control_frame, text="Include subdirectories", variable=self.recursive
        ).pack(side=tk.LEFT)

        # Action buttons
        ttk.Button(control_frame, text="Preview", command=self._preview).pack(
            side=tk.RIGHT, padx=5
        )
        ttk.Button(
            control_frame, text="Process Files", command=self._process_files
        ).pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="Dry Run", command=self._dry_run).pack(
            side=tk.RIGHT, padx=5
        )

    def _create_layout(self):
        """Create the main layout."""
        # Make the notebook expand to fill the window
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure grid weights
        for frame in [self.basic_frame, self.template_frame, self.advanced_frame]:
            frame.grid_columnconfigure(1, weight=1)

    def _select_basic_input_folder(self):
        """Select input folder for basic mode."""
        folder = filedialog.askdirectory(
            title="Select Input Folder",
            initialdir=os.getcwd(),
        )
        if folder:
            self.basic_input_folder.set(folder)

    def _select_template_input_folder(self):
        """Select input folder for template mode."""
        folder = filedialog.askdirectory(
            title="Select Input Folder",
            initialdir=os.getcwd(),
        )
        if folder:
            self.template_input_folder.set(folder)

    def _select_output_folder(self):
        """Select output folder."""
        # Get current tab
        current_tab = self.notebook.index(self.notebook.select())

        # Get the appropriate input folder based on mode
        start_dir = (
            self.template_input_folder.get()
            if current_tab == 1
            else self.basic_input_folder.get()
        ) or os.getcwd()

        folder = filedialog.askdirectory(
            title="Select Output Folder", initialdir=start_dir
        )
        if folder:
            self.output_folder.set(folder)

    def _select_template_folder(self):
        """Select template folder."""
        # Start from template input folder if selected, otherwise current directory
        start_dir = self.template_input_folder.get() or os.getcwd()
        folder = filedialog.askdirectory(
            title="Select Template Folder", initialdir=start_dir
        )
        if folder:
            self.template_folder.set(folder)

    def _select_backup_folder(self):
        """Select backup folder."""
        # Get current tab
        current_tab = self.notebook.index(self.notebook.select())

        # Get the appropriate input folder based on mode
        start_dir = (
            self.template_input_folder.get()
            if current_tab == 1
            else self.basic_input_folder.get()
        ) or os.getcwd()

        folder = filedialog.askdirectory(
            title="Select Backup Folder", initialdir=start_dir
        )
        if folder:
            self.backup_folder.set(folder)

    def _select_tracking_file(self):
        """Select tracking file."""
        # Default to script's root directory
        start_dir = os.path.dirname(self.tracking_file.get())
        filename = filedialog.asksaveasfilename(
            title="Select Tracking File",
            initialdir=start_dir,
            initialfile=os.path.basename(self.tracking_file.get()),
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if filename:
            self.tracking_file.set(filename)

    def _preview(self):
        """Preview the first file."""
        # Get current tab
        current_tab = self.notebook.index(self.notebook.select())

        # Get the appropriate input folder based on mode
        input_folder = (
            self.template_input_folder.get()
            if current_tab == 1
            else self.basic_input_folder.get()
        )

        if not input_folder:
            messagebox.showerror("Error", "Please select an input folder first.")
            return

        # Find the first audio file
        input_path = Path(input_folder)
        audio_files = []
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(input_path.glob(f"*{ext}"))

        if not audio_files:
            messagebox.showerror(
                "Error", "No audio files found in the selected folder."
            )
            return

        try:
            preview_audio(str(audio_files[0]))
        except Exception as e:
            messagebox.showerror("Error", f"Error previewing audio: {str(e)}")

    def _process_files(self, dry_run: bool = False):
        """Process all files."""
        # Get current tab
        current_tab = self.notebook.index(self.notebook.select())

        # Get the appropriate input folder based on mode
        input_folder = (
            self.template_input_folder.get()
            if current_tab == 1
            else self.basic_input_folder.get()
        )

        if not input_folder:
            messagebox.showerror("Error", "Please select an input folder first.")
            return

        # Check backup settings
        if self.make_backup.get() and not self.backup_folder.get():
            messagebox.showerror(
                "Error", "Please select a backup folder or disable backup creation."
            )
            return

        # Create progress window
        progress_window = ProgressWindow(self.root)

        # Create queue for log messages
        log_queue = queue.Queue()

        # Create and add queue handler to logger
        queue_handler = LogHandler(log_queue)
        queue_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger = logging.getLogger("audio_trimmer")
        logger.addHandler(queue_handler)

        def process_thread():
            try:
                # Get current tab
                current_tab = self.notebook.index(self.notebook.select())

                # Template mode requires template folder
                if current_tab == 1 and not self.template_folder.get():
                    progress_window.log("Error: Please select a template folder first.")
                    return

                def update_progress(current, total, phase):
                    progress_window.update_progress(current, total, phase)

                # Process based on mode
                if current_tab == 1:  # Template mode
                    from ..core.audio_processor import remove_detected_intros

                    # Calculate CPU allocation
                    total_cpus = multiprocessing.cpu_count()
                    try:
                        reserved_cpus = int(self.reserved_cpus.get())
                        match_ratio = int(self.match_cpu_ratio.get()) / 100.0
                    except ValueError:
                        reserved_cpus = 2
                        match_ratio = 0.33

                    # Ensure values are within valid ranges
                    reserved_cpus = max(0, min(reserved_cpus, total_cpus - 1))
                    match_ratio = max(0.1, min(match_ratio, 0.9))

                    # Calculate available CPUs and worker distribution
                    available_cpus = max(1, total_cpus - reserved_cpus)
                    match_workers = max(1, int(available_cpus * match_ratio))
                    process_workers = max(1, available_cpus - match_workers)

                    progress_window.log(
                        f"CPU allocation: {match_workers} matching, {process_workers} processing, {reserved_cpus} reserved"
                    )

                    remove_detected_intros(
                        input_folder=self.template_input_folder.get(),
                        template_folder=self.template_folder.get(),
                        make_backup=self.make_backup.get(),
                        output_dir=self.output_folder.get() or None,
                        dry_run=dry_run,
                        recursive=self.recursive.get(),
                        quality=QUALITY_PRESETS.get(self.quality.get()),
                        match_threshold=float(self.match_threshold.get()),
                        progress_callback=update_progress,
                        backup_dir=(
                            self.backup_folder.get() if self.make_backup.get() else None
                        ),
                        tracking_file=self.tracking_file.get() or None,
                        force_process=self.force_process.get(),
                        max_workers=(match_workers, process_workers),  # Pass as tuple
                    )
                else:  # Basic mode
                    from ..core.audio_processor import remove_audio_duration

                    remove_audio_duration(
                        input_folder=self.basic_input_folder.get(),
                        duration_to_remove=float(self.duration.get()),
                        make_backup=self.make_backup.get(),
                        output_dir=self.output_folder.get() or None,
                        dry_run=dry_run,
                        recursive=self.recursive.get(),
                        from_end=self.from_end.get(),
                        quality=QUALITY_PRESETS.get(self.quality.get()),
                        smart_trim=self.smart_trim.get(),
                        fade_duration=(
                            float(self.fade_duration.get())
                            if self.fade_duration.get()
                            else None
                        ),
                        progress_callback=update_progress,
                        backup_dir=(
                            self.backup_folder.get() if self.make_backup.get() else None
                        ),
                        tracking_file=self.tracking_file.get() or None,
                        force_process=self.force_process.get(),
                    )

                progress_window.update_status("Processing completed!")
                if dry_run:
                    progress_window.log("Dry run completed successfully!")
                else:
                    progress_window.log("Processing completed successfully!")

            except Exception as e:
                progress_window.log(f"Error: {str(e)}")
            finally:
                logger.removeHandler(queue_handler)
                self.root.after(2000, progress_window.close)  # Close after 2 seconds

        def update_log():
            """Update log display from queue."""
            try:
                while True:
                    message = log_queue.get_nowait()
                    progress_window.log(message)
            except queue.Empty:
                pass

            if threading.active_count() > 1 and not progress_window.cancelled:
                self.root.after(100, update_log)

        # Start processing thread
        threading.Thread(target=process_thread, daemon=True).start()

        # Start log update loop
        update_log()

    def _dry_run(self):
        """Perform a dry run."""
        self._process_files(dry_run=True)

    def _get_processing_params(self) -> Dict[str, Any]:
        """Get all processing parameters based on current GUI state."""
        # Get current tab
        current_tab = self.notebook.index(self.notebook.select())

        # Get the appropriate input folder based on mode
        input_folder = (
            self.template_input_folder.get()
            if current_tab == 1
            else self.basic_input_folder.get()
        )

        params = {
            "input_folder": input_folder,
            "make_backup": self.make_backup.get(),
            "recursive": self.recursive.get(),
        }

        # Add output folder if specified
        if self.output_folder.get():
            params["output_dir"] = self.output_folder.get()

        # Add quality settings if specified
        if self.quality.get():
            params["quality"] = QUALITY_PRESETS[self.quality.get()]

        # Add mode-specific parameters
        if current_tab == 0:  # Basic mode
            params.update(
                {
                    "duration": float(self.duration.get()),
                    "from_end": self.from_end.get(),
                }
            )
        elif current_tab == 1:  # Template mode
            params.update(
                {
                    "template_folder": self.template_folder.get(),
                    "match_threshold": float(self.match_threshold.get()),
                }
            )

        # Add advanced options
        if self.fade_duration.get():
            params["fade_duration"] = float(self.fade_duration.get())
        params["smart_trim"] = self.smart_trim.get()

        return params

    def _update_template_input_files_list(self, *args):
        """Update the template mode input files list when folder changes."""
        self.input_files_list.delete(0, tk.END)
        folder = self.template_input_folder.get()
        if folder and os.path.exists(folder):
            for ext in SUPPORTED_FORMATS:
                for file in Path(folder).glob(f"*{ext}"):
                    self.input_files_list.insert(tk.END, file.name)

    def _preview_input_file(self):
        """Preview the selected input file."""
        if not self.template_input_folder.get():
            messagebox.showerror("Error", "Please select an input folder first.")
            return

        selection = self.input_files_list.curselection()
        if not selection:
            messagebox.showerror("Error", "Please select a file to preview.")
            return

        filename = self.input_files_list.get(selection[0])
        file_path = Path(self.template_input_folder.get()) / filename

        try:
            preview_audio(str(file_path))
        except Exception as e:
            messagebox.showerror("Error", f"Error previewing file: {str(e)}")


class ProgressWindow:
    """Progress window with log display and progress bar."""

    def __init__(self, parent, title="Processing Files"):
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("600x400")
        self.window.transient(parent)
        self.window.grab_set()

        # Create log display
        self.log_display = scrolledtext.ScrolledText(self.window, height=15, width=70)
        self.log_display.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Create progress bars frame
        progress_frame = ttk.LabelFrame(self.window, text="Progress", padding="5")
        progress_frame.pack(padx=10, pady=5, fill=tk.X)

        # Create matching progress bar
        ttk.Label(progress_frame, text="Matching:").grid(
            row=0, column=0, sticky="w", padx=5
        )
        self.match_progress_var = tk.DoubleVar()
        self.match_progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.match_progress_var,
            maximum=100,
            mode="determinate",
        )
        self.match_progress_bar.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.match_status_label = ttk.Label(progress_frame, text="0/0")
        self.match_status_label.grid(row=0, column=2, padx=5)

        # Create processing progress bar
        ttk.Label(progress_frame, text="Processing:").grid(
            row=1, column=0, sticky="w", padx=5
        )
        self.process_progress_var = tk.DoubleVar()
        self.process_progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.process_progress_var,
            maximum=100,
            mode="determinate",
        )
        self.process_progress_bar.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        self.process_status_label = ttk.Label(progress_frame, text="0/0")
        self.process_status_label.grid(row=1, column=2, padx=5)

        # Configure grid column weights
        progress_frame.columnconfigure(1, weight=1)

        # Create status label
        self.status_label = ttk.Label(self.window, text="Initializing...")
        self.status_label.pack(padx=10, pady=5)

        # Create cancel button
        self.cancel_button = ttk.Button(self.window, text="Cancel", command=self.cancel)
        self.cancel_button.pack(pady=10)

        self.cancelled = False
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)

    def update_progress(self, current: int, total: int, phase: str) -> None:
        """Update progress bar."""
        progress = (current / total) * 100 if total > 0 else 0

        if phase == "matching":
            self.match_progress_var.set(progress)
            self.match_status_label.config(text=f"{current}/{total}")
            self.status_label.config(text=f"Matching files: {current}/{total}")
        else:  # processing
            self.process_progress_var.set(progress)
            self.process_status_label.config(text=f"{current}/{total}")
            self.status_label.config(
                text=f"Processing matched files: {current}/{total}"
            )

        self.window.update_idletasks()

    def log(self, message):
        """Add message to log display."""
        self.log_display.insert(tk.END, message + "\n")
        self.log_display.see(tk.END)
        self.window.update_idletasks()

    def cancel(self):
        """Handle cancel button click."""
        self.cancelled = True
        self.status_label.config(text="Cancelling...")
        self.cancel_button.config(state="disabled")

    def close(self):
        """Close the progress window."""
        self.window.destroy()


class LogHandler(logging.Handler):
    """Custom logging handler that sends logs to a queue."""

    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def emit(self, record):
        self.queue.put(self.format(record))


def create_main_window():
    """Create and run the main window."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Audio Intro Trimmer")

    # Set window size and make it resizable
    root.geometry("800x600")
    root.minsize(600, 400)

    # Create main window instance
    app = MainWindow(root)

    # Start the main event loop
    root.mainloop()
