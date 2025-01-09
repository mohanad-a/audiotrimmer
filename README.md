# Audio Intro Trimmer

A powerful tool for batch processing audio files to remove intros, outros, or specific segments. Supports both duration-based trimming and template-based intro detection.

## Features

- GUI and CLI interfaces
- Two operation modes:
  - Duration-based trimming (remove specific duration from start/end)
  - Template-based intro detection and removal
- Smart trimming with silence detection
- Multi-CPU support for template matching
- Audio quality presets
- Fade in/out effects
- Progress tracking and backup functionality
- Recursive directory processing
- Multiple audio format support

## Installation

1. First, ensure you have Python 3.7+ installed:

   ```bash
   python3 --version
   ```

2. Install FFmpeg on your system:

   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt-get install ffmpeg

   # Windows
   # Download from https://www.ffmpeg.org/download.html
   ```

3. Install Tkinter (required for GUI features):

   ```bash
   # macOS
   brew uninstall --ignore-dependencies python@3.12  # Remove current Python
   brew install python-tk@3.12  # Install Python with Tkinter
   # Or for all Python versions:
   brew install python-tk

   # Ubuntu/Debian
   sudo apt-get install python3-tk

   # Fedora
   sudo dnf install python3-tkinter

   # Windows
   # Reinstall Python and make sure to check "tcl/tk and IDLE" during installation
   ```

4. Set up a Python virtual environment:

   ```bash
   # Create a virtual environment
   python3 -m venv venv

   # Activate the virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate

   # Verify you're in the virtual environment
   which python  # Should show path to venv
   ```

5. Install Python dependencies:

   ```bash
   # Make sure you're in the virtual environment
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### GUI Mode

Launch the graphical interface:

```bash
# Make sure your virtual environment is activated
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate   # On Windows

python -m introtrimmer --gui
```

### CLI Mode

#### Duration-based Trimming

Remove a specific duration from audio files:

```bash
python -m introtrimmer -i input_folder -d 5.5 -o output_folder
```

Options:

- `-d`, `--duration`: Duration to remove in seconds
- `--from-end`: Remove from end instead of start
- `--smart-trim`: Use silence detection for smarter trimming
- `--fade`: Add fade in/out effect (specify duration in seconds)

#### Template-based Intro Detection

Remove intros by matching against template files:

```bash
python -m introtrimmer -i input_folder -t template_folder -o output_folder
```

Options:

- `-t`, `--template`: Folder containing intro template files
- `--threshold`: Matching threshold (default: 0.85)
- `--reserved-cpus`: CPUs to reserve for system (default: 2)
- `--match-cpu-ratio`: Percentage of CPUs for matching (default: 33)

### Common Options

- `-i`, `--input`: Input folder containing audio files
- `-o`, `--output`: Output folder for processed files
- `-b`, `--backup`: Create backup of original files
- `-r`, `--recursive`: Process files in subdirectories
- `--dry-run`: Show what would be done without making changes
- `--quality`: Audio quality preset (high/medium/low)
- `--debug`: Enable debug logging

### Advanced Options

#### Audio Processing

- `--smart-trim`: Use silence detection for smarter trimming
- `--fade`: Add fade in/out effect (specify duration in seconds)
- `--from-end`: Remove duration from end instead of start
- `--segments`: JSON file specifying multiple segments to remove (format: [[start1, end1], [start2, end2]])
- `--batch-config`: JSON file for batch processing with different durations per file

#### Template Matching

- `--threshold`: Matching threshold (0.0-1.0, default: 0.85)
- `--reserved-cpus`: Number of CPUs to reserve for system (default: 2)
- `--match-cpu-ratio`: Percentage of available CPUs for matching (10-90, default: 33)
- `--template-format`: Format of template files (default: auto-detect)

#### Output Control

- `--output-format`: Force output format (.mp3, .wav, etc.)
- `--preserve-metadata`: Keep original file metadata
- `--overwrite`: Overwrite existing files without asking
- `--progress`: Show progress bar during processing
- `--log-file`: Path to save processing log
- `--quiet`: Suppress non-error output

## GUI Features

The graphical interface provides an intuitive way to access all features:

### Main Window

- **Input/Output Selection**

  - Browse button for input folder selection
  - Browse button for output folder selection
  - Option to create output folder if it doesn't exist
  - Toggle for recursive directory processing

- **Operation Mode**
  - Duration-based trimming
  - Template-based intro detection
  - Batch processing with configuration file

### Duration Mode Settings

- Duration input field (seconds)
- Position selector (start/end)
- Smart trim toggle
- Fade effect controls
  - Fade in duration
  - Fade out duration
  - Fade curve type

### Template Mode Settings

- Template folder selection
- Matching threshold slider
- CPU allocation controls
  - Reserved CPUs selector
  - Matching CPU ratio slider
- Template preview window

### Quality Settings

- Quality preset selector (high/medium/low)
- Advanced quality controls:
  - Bitrate
  - Sample rate
  - Channels
  - Codec options

### Processing Options

- Backup toggle
- Format conversion
- Metadata preservation
- Progress display
  - Overall progress bar
  - Current file progress
  - Estimated time remaining
  - Processing speed

### Preview Features

- Audio waveform display
- Trim point markers
- Play/pause controls
- Zoom controls
- Before/after comparison

### Batch Processing

- Batch configuration editor
- File list with individual settings
- Drag-and-drop support
- Import/export configurations

### Additional Features

- Dark/light theme toggle
- Recent folders history
- Processing log viewer
- Error notification system
- Auto-save settings
- Keyboard shortcuts

## Supported Formats

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- AAC (.aac)
- OGG (.ogg)
- FLAC (.flac)

## Examples

1. Remove 10 seconds from the start of all files:

   ```bash
   python -m introtrimmer -i audio_files -d 10 -o processed_files
   ```

2. Remove detected intros using templates:

   ```bash
   python -m introtrimmer -i audio_files -t intro_templates -o processed_files --threshold 0.9
   ```

3. Smart trim with fade effect:

   ```bash
   python -m introtrimmer -i audio_files -d 5 -o processed_files --smart-trim --fade 0.5
   ```

4. High-quality processing with backups:

   ```bash
   python -m introtrimmer -i audio_files -d 8 -o processed_files -b --quality high
   ```

5. Process multiple segments:

   ```bash
   # Remove segments from 0-10s and 30-40s
   python -m introtrimmer -i audio_files --segments segments.json -o processed_files
   ```

6. Batch processing with different durations:

   ```bash
   # Process files with different trim durations
   python -m introtrimmer -i audio_files --batch-config config.json -o processed_files
   ```

7. High-quality MP3 conversion with metadata:

   ```bash
   python -m introtrimmer -i audio_files -d 5 -o processed_files --output-format .mp3 --quality high --preserve-metadata
   ```

8. Template matching with custom CPU allocation:
   ```bash
   python -m introtrimmer -i audio_files -t templates -o processed_files --reserved-cpus 4 --match-cpu-ratio 50
   ```

## Troubleshooting

### Virtual Environment Issues

1. If you get "command not found: python":

   - Make sure you're using the correct Python command for your system:
     ```bash
     # Try these alternatives:
     python3 -m venv venv
     python3.12 -m venv venv
     ```

2. If your virtual environment isn't activating:

   - Make sure you're using the correct activation command for your shell:

     ```bash
     # bash/zsh (macOS/Linux):
     source venv/bin/activate

     # fish (macOS/Linux):
     source venv/bin/activate.fish

     # csh/tcsh (macOS/Linux):
     source venv/bin/activate.csh

     # Windows PowerShell:
     .\venv\Scripts\Activate.ps1

     # Windows cmd.exe:
     .\venv\Scripts\activate.bat
     ```

3. If packages aren't installing:
   - Make sure your virtual environment is activated (check with `which python`)
   - Try upgrading pip: `pip install --upgrade pip`
   - If you get permission errors, DO NOT use sudo with pip in a virtual environment

### GUI Issues

1. If you get "GUI modules not found" error:

   - This means tkinter is not installed properly
   - Do NOT use `pip install tk` as it won't work
   - Install tkinter using your system package manager as shown in the installation section

2. For macOS users:

   - If using Homebrew Python, you might need to reinstall Python with tcl/tk support:
     ```bash
     brew uninstall python@3.12
     brew install python@3.12 --with-tcl-tk
     ```

3. For Linux users:
   - Make sure you have the correct tkinter package for your Python version
   - The package name might vary depending on your distribution

### FFmpeg Issues

If you get FFmpeg-related errors:

1. Make sure FFmpeg is installed and accessible from command line:
   ```bash
   ffmpeg -version
   ```
2. On Windows, make sure FFmpeg is added to your system PATH
3. Try reinstalling FFmpeg if you get codec-related errors

### Template Matching Issues

1. If matching is too strict (missing intros):

   - Lower the threshold value (e.g., `--threshold 0.80`)
   - Use multiple template files for variations

2. If matching is too lenient (false positives):

   - Increase the threshold value (e.g., `--threshold 0.90`)
   - Use more precise template files

3. If processing is too slow:
   - Adjust CPU allocation with `--reserved-cpus` and `--match-cpu-ratio`
   - Process fewer files at once
   - Use smaller template files
