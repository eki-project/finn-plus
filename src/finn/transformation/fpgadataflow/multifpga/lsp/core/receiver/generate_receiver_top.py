"""Generate the LSP receiver top function from a JSON config of its connections.

Config: {"connections": [{"mac": "0xA1", "streams": [{"width": 64, "depth": 512}, ...]}, ...]}
Connection I gets macStreams[I], its own CTRL status stream and the ports c<I>_s<J>.

Usage: python generate_receiver_top.py <config.json> <output.cpp>
"""

import jinja2
import json
import sys
from pathlib import Path


def render_receiver_top(config: dict, config_name: str) -> str:
    """Return the C++ source of the receiver top function for the given config."""
    connections = config["connections"]
    if not connections:
        raise ValueError(f"{config_name}: at least one connection is required")
    for index, connection in enumerate(connections):
        if not connection["streams"]:
            raise ValueError(f"{config_name}: connection {index} has no streams")

    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(__file__).parent),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )
    return environment.get_template("receiver_top.cpp.j2").render(
        connections=connections, config_name=config_name
    )


if __name__ == "__main__":
    config_path, output_path = Path(sys.argv[1]), Path(sys.argv[2])
    source = render_receiver_top(json.loads(config_path.read_text()), config_path.name)
    output_path.write_text(source)
