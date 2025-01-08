"""Module loading utilities."""

from typing import Tuple, Optional, Any


def load_gui_modules() -> Tuple[Optional[Any], Optional[Any]]:
    """Load GUI modules (tkinter)."""
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox

        return tk, ttk
    except ImportError:
        error_msg = """
GUI modules not found. Tkinter must be installed through your system package manager:

macOS:
    brew install python-tk

Ubuntu/Debian:
    sudo apt-get install python3-tk

Fedora:
    sudo dnf install python3-tkinter

Windows:
    Reinstall Python and make sure to check "tcl/tk and IDLE" during installation

Note: Do NOT use 'pip install tk' as it won't work!
"""
        return error_msg, None


def load_audio_modules() -> (
    Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]
):
    """Load audio processing modules."""
    try:
        import sounddevice as sd
        import soundfile as sf
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent

        return sd, sf, AudioSegment, detect_nonsilent
    except ImportError:
        return None, None, None, None


def load_audio_processing_modules() -> (
    Tuple[Optional[Any], Optional[Any], Optional[Any]]
):
    """Load advanced audio processing modules."""
    try:
        import librosa
        import soundfile as sf
        from scipy import signal

        return librosa, sf, signal
    except ImportError:
        return None, None, None
