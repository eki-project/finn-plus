"""Templates module initialization."""
from pathlib import Path


def get_templates_folder() -> Path:
    """Return the Path to the finn/templates/ folder."""
    return Path(__file__).parent


def load_codegen_template(name: str) -> str:
    """Return the contents of a code-generation template from ``templates/codegen/``.

    These files hold the HLS/RTL scaffolding that the fpgadataflow backends fill
    in by substituting ``$PLACEHOLDER$`` markers, so they are returned verbatim.

    Args:
        name: File name inside ``templates/codegen/`` (e.g. ``hls_ipgen.tcl``).

    Returns:
        The template text, unmodified.

    """
    return (get_templates_folder() / "codegen" / name).read_text(encoding="utf-8")
