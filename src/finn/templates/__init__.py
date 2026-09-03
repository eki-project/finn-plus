"""Templates module initialization."""
import jinja2
from pathlib import Path


def get_templates_folder() -> Path:
    """Return the Path to the finn/templates/ folder."""
    return Path(__file__).parent


def get_jinja_environment(*args, **kwargs) -> jinja2.Environment:  # noqa
    """Return a jinja2 templating environment with a loader prepared for this template
    directory.
    """
    return jinja2.Environment(
        *args, **kwargs, loader=jinja2.FileSystemLoader(Path(__file__).parent)
    )
