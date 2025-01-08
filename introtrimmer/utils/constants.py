"""Constants used throughout the application."""

SUPPORTED_FORMATS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")

SUPPORTED_CODECS = {
    ".mp3": "libmp3lame",
    ".wav": "pcm_s16le",
    ".m4a": "aac",
    ".aac": "aac",
    ".ogg": "libvorbis",
    ".flac": "flac",
}

QUALITY_PRESETS = {
    "high": {"audio_bitrate": "320k", "acodec": "libmp3lame"},
    "medium": {"audio_bitrate": "192k", "acodec": "libmp3lame"},
    "low": {"audio_bitrate": "128k", "acodec": "libmp3lame"},
}
