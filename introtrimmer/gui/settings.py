"""Settings management for persistent GUI preferences."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class SettingsManager:
    """Manages loading and saving of GUI settings."""
    
    def __init__(self, app_name: str = "audio_intro_remover"):
        """Initialize settings manager with app-specific config directory."""
        self.app_name = app_name
        self._setup_config_directory()
        
    def _setup_config_directory(self):
        """Set up the configuration directory."""
        # Use platform appropriate config directory
        if os.name == 'nt':  # Windows
            config_base = os.environ.get('APPDATA', os.path.expanduser('~'))
        else:  # Unix/Linux/macOS
            config_base = os.path.expanduser('~/.config')
        
        self.config_dir = Path(config_base) / self.app_name
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.settings_file = self.config_dir / 'settings.json'
        
    def get_default_settings(self) -> Dict[str, Any]:
        """Get default settings structure."""
        # Get script directory for default tracking file
        script_dir = Path(__file__).parent.parent.parent
        default_tracking_file = str(script_dir / "processed_files.json")
        
        return {
            # Window settings
            "window": {
                "geometry": "800x600",
                "selected_tab": 0  # 0=Basic, 1=Template, 2=Advanced
            },
            
            # Basic mode settings
            "basic_mode": {
                "input_folder": "",
                "duration": "8.65",
                "from_end": False
            },
            
            # Template mode settings  
            "template_mode": {
                "input_folder": "",
                "template_folder": "",
                "match_threshold": "0.85"
            },
            
            # Common paths
            "paths": {
                "output_folder": "",
                "backup_folder": "",
                "tracking_file": default_tracking_file
            },
            
            # Processing options
            "processing": {
                "quality": "high",
                "fade_duration": "",
                "make_backup": True,
                "recursive": True,
                "smart_trim": False,
                "preserve_original_quality": False,
                "force_process": False
            },
            
            # Recent folders for convenience
            "recent": {
                "input_folders": [],
                "template_folders": [],
                "output_folders": [],
                "backup_folders": []
            }
        }
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file, returning defaults if file doesn't exist."""
        if not self.settings_file.exists():
            return self.get_default_settings()
        
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                saved_settings = json.load(f)
            
            # Merge with defaults to ensure all keys exist
            default_settings = self.get_default_settings()
            merged_settings = self._merge_settings(default_settings, saved_settings)
            
            return merged_settings
            
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Error loading settings: {e}. Using defaults.")
            return self.get_default_settings()
    
    def save_settings(self, settings: Dict[str, Any]) -> None:
        """Save settings to file."""
        try:
            # Ensure directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first for atomic operation
            temp_file = self.settings_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.replace(self.settings_file)
            
        except (IOError, OSError) as e:
            logging.error(f"Error saving settings: {e}")
    
    def _merge_settings(self, defaults: Dict[str, Any], saved: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge saved settings with defaults."""
        result = defaults.copy()
        
        for key, value in saved.items():
            if key in result:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    result[key] = self._merge_settings(result[key], value)
                else:
                    result[key] = value
            else:
                result[key] = value
                
        return result
    
    def add_recent_folder(self, folder_type: str, folder_path: str, max_recent: int = 10) -> None:
        """Add a folder to recent folders list."""
        settings = self.load_settings()
        
        if folder_type not in settings["recent"]:
            settings["recent"][folder_type] = []
        
        recent_list = settings["recent"][folder_type]
        
        # Remove if already exists
        if folder_path in recent_list:
            recent_list.remove(folder_path)
        
        # Add to beginning
        recent_list.insert(0, folder_path)
        
        # Limit to max_recent items
        settings["recent"][folder_type] = recent_list[:max_recent]
        
        self.save_settings(settings)
    
    def get_recent_folders(self, folder_type: str) -> list:
        """Get recent folders of specified type."""
        settings = self.load_settings()
        return settings.get("recent", {}).get(folder_type, [])
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to defaults."""
        default_settings = self.get_default_settings()
        self.save_settings(default_settings) 