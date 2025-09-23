#!/usr/bin/env python3
"""Split the generated API documentation into separate files per module."""

import re
import sys
from pathlib import Path


def split_api_documentation(api_file_path="docs/api.md", output_dir="wiki-content"):
    """Split docs/api.md into separate files for each module.

    Args:
        api_file_path: Path to the generated API documentation file
        output_dir: Directory to write the split files to
    """
    api_file = Path(api_file_path)
    wiki_dir = Path(output_dir)
    wiki_dir.mkdir(exist_ok=True)

    if not api_file.exists():
        print(f"❌ No API documentation found at {api_file_path}")
        return False

    with open(api_file, "r") as f:
        content = f.read()

    # Split content by module headers (lines starting with # finn.)
    # Use regex to find module boundaries
    module_pattern = r"^# (finn\.[^\n]+)$"
    modules = re.split(module_pattern, content, flags=re.MULTILINE)

    # Create main API index
    index_content = []
    index_content.append("# FINN API Documentation")
    index_content.append("")
    index_content.append(
        "This is the comprehensive API reference for the FINN framework, generated\
        automatically from source code."
    )
    index_content.append("")
    index_content.append("## Modules")
    index_content.append("")

    module_files = []

    # Process each module (skip first element which is content before first module)
    for i in range(1, len(modules), 2):
        if i + 1 < len(modules):
            module_name = modules[i].strip()
            module_content = modules[i + 1].strip()

            if module_content:  # Only create file if there's content
                # Create safe filename
                safe_filename = module_name.replace(".", "-").replace("_", "-").replace(" ", "-")
                filename = f"{safe_filename}.md"
                module_files.append((module_name, filename))

                # Create module file content
                file_content = []
                file_content.append(f"# {module_name}")
                file_content.append("")
                file_content.append(module_content)

                # Write module file
                module_file = wiki_dir / filename
                with open(module_file, "w") as f:
                    f.write("\n".join(file_content))

                print(f"✅ Created {filename} ({len(module_content)} chars)")

    # Add module links to index
    for module_name, filename in sorted(module_files):
        # Create a cleaner display name
        index_content.append(f"- [{module_name}]({filename.replace('.md', '')})")

    index_content.append("")
    index_content.append("---")
    index_content.append("")
    index_content.append("*Documentation generated automatically from source code.*")

    # Write main index file
    with open(wiki_dir / "API-Documentation.md", "w") as f:
        f.write("\n".join(index_content))

    print(f"✅ Created main API index with {len(module_files)} modules")
    return True


def main():
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(description="Split API documentation into separate files")
    parser.add_argument(
        "--input",
        default="docs/api.md",
        help="Path to the input API documentation file (default: docs/api.md)",
    )
    parser.add_argument(
        "--output",
        default="wiki-content",
        help="Output directory for split files (default: wiki-content)",
    )

    args = parser.parse_args()

    success = split_api_documentation(args.input, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
