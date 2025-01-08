# Audio Intro Remover

A powerful tool for batch processing audio files to remove intros, outros, or specific segments.

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

## Basic Usage

Make sure your virtual environment is activated before running any commands:

```bash
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

Remove 8.65 seconds from the beginning of all audio files in a folder:

```bash
python main.py path/to/audio/folder
```

Specify custom duration:

```bash
python main.py path/to/audio/folder --duration 5.5
```

## Advanced Features

1. Create backups before processing:

   ```bash
   python main.py folder --backup
   ```

2. Save to different directory:

   ```bash
   python main.py folder --output-dir processed
   ```

3. Smart trim using silence detection:

   ```bash
   python main.py folder --smart-trim
   ```

4. Add fade effects:

   ```bash
   python main.py folder --fade-duration 2.0
   ```

5. Convert format:

   ```bash
   python main.py folder --output-format .mp3 --audio-quality high
   ```

6. Process multiple segments:

   ```bash
   # Create segments.json:
   # [[0, 10], [30, 40]]  # Removes 0-10s and 30-40s
   python main.py folder --segments segments.json
   ```

7. Batch processing with different durations:
   ```bash
   # Create batch_config.json:
   # {"file1.mp3": 5.5, "file2.mp3": 10.2}
   python main.py folder --batch-config batch_config.json
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

1. Make sure FFmpeg is installed and accessible from command line
2. Try running `ffmpeg -version` to verify the installation
3. On Windows, make sure FFmpeg is added to your system PATH

## Supported Formats

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- AAC (.aac)
- OGG (.ogg)
- FLAC (.flac)

## Options

```bash
python main.py --help
```

Key options:

- `--duration`, `-d`: Duration to remove (seconds)
- `--backup`, `-b`: Create backups
- `--output-dir`, `-o`: Output directory
- `--dry-run`: Preview changes without modifying files
- `--from-end`: Remove from end instead of beginning
- `--smart-trim`: Use silence detection
- `--audio-quality`: Choose quality (high/medium/low)
- `--gui`: Launch graphical interface
- `--preview`: Preview audio before processing

### Template-based Intro Removal

1. Create a folder for your intro templates:

   ```bash
   mkdir intro_templates
   ```

2. Copy your intro template files into the folder:

   ```bash
   cp intro1.mp3 intro2.mp3 intro3.mp3 intro_templates/
   ```

3. Run the script with template matching:

   ```bash
   # Basic usage
   python main.py your_audio_folder --template-folder intro_templates

   # With backup (recommended for first run)
   python main.py your_audio_folder --template-folder intro_templates --backup

   # Test run first (no changes made)
   python main.py your_audio_folder --template-folder intro_templates --dry-run
   ```

4. Adjust matching sensitivity if needed:

   ```bash
   # More strict matching (fewer false positives)
   python main.py your_audio_folder --template-folder intro_templates --match-threshold 0.90

   # More lenient matching (fewer false negatives)
   python main.py your_audio_folder --template-folder intro_templates --match-threshold 0.80
   ```
