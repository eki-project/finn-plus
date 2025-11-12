from pathlib import Path


def get_templates_folder() -> Path:
    """Returns the Path to the finn/templates/ folder."""
    return Path(__file__).parent
