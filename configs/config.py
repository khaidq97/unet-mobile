import yaml
from types import SimpleNamespace
from pathlib import Path
from typing import Union, Dict, Any

root_dir = Path(__file__).parent

class Config(SimpleNamespace):
    def __init__(self, config_path: Union[str, Path, None] = None):
        """
        Initialize Config object from a YAML file.
        
        Args:
            config_path: Path to YAML config file. If None, defaults to 'config.yml' in root_dir.
        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the YAML file is invalid.
            ValueError: If the YAML does not map to a dictionary.
        """
        # Resolve config filepath
        if config_path is None:
            config_path = root_dir / "config.yml"
        else:
            config_path = Path(config_path)

        # Load YAML file
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file)
        except (FileNotFoundError, yaml.YAMLError) as e:
            raise type(e)(f"Failed to load config file {config_path} - {str(e)}")

        # Ensure YAML is a dictionary
        if not isinstance(config_data, dict):
            raise ValueError(f"Config file {config_path} must map to a dictionary")

        # Convert nested dicts/lists to SimpleNamespace recursively
        processed_data = {k: Config._to_namespace(v) for k, v in config_data.items()}

        # Initialize SimpleNamespace with config_data
        super().__init__(**processed_data)

    @staticmethod
    def _to_namespace(data: Any) -> Any:
        """
        Recursively convert dictionaries to SimpleNamespace objects.
        
        Args:
            data: Input data (dict, list, or other type).
        Returns:
            SimpleNamespace object or original data if not a dict.
        """
        if isinstance(data, dict):
            return SimpleNamespace(**{k: Config._to_namespace(v) for k, v in data.items()})
        elif isinstance(data, list):
            return [Config._to_namespace(item) for item in data]
        else:
            return data

    def save_to_yaml(self, output_path: Union[str, Path]) -> None:
        """
        Save the current config to a YAML file.
        
        Args:
            output_path: Path to save the YAML file.
        """
        output_path = Path(output_path)
        config_dict = Config._to_dict(self)
        try:
            with open(output_path, 'w', encoding="utf-8") as file:
                yaml.safe_dump(config_dict, file, sort_keys=False)
        except Exception as e:
            raise Exception(f"Failed to save config to {output_path} - {str(e)}")

    @staticmethod
    def _to_dict(obj: Any) -> Any:
        """
        Recursively convert SimpleNamespace to dictionary.
        
        Args:
            obj: Input object (SimpleNamespace, list, or other type).
        Returns:
            Dictionary or original data.
        """
        if isinstance(obj, SimpleNamespace):
            return {k: Config._to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, list):
            return [Config._to_dict(item) for item in obj]
        else:
            return obj

if __name__ == "__main__":
    try:
        # Example usage
        config = Config()
        print(config.data.extensions)  # Output: ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        print(config.training.epochs)  # Output: 100
        # Save config to a new file
        config.save_to_yaml("config_backup.yml")
    except Exception as e:
        print(f"Error: {e}")