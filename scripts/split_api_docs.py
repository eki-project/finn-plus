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

            # Skip base/parent modules with minimal content (typically just package docstrings)
            # These are usually modules like "finn.builder" that only contain brief descriptions
            if module_content and len(module_content) > 100:  # Only process modules with substantial content
                # Create safe filename
                safe_filename = (
                    module_name.replace(".", "-")
                    .replace("_", "-")
                    .replace(" ", "-")
                    .replace("\\", "")
                )
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
            else:
                print(f"⏭️  Skipped {module_name} (base module, {len(module_content)} chars)")

    # Create hierarchical structure for the index
    hierarchy = {}
    for module_name, filename in module_files:
        parts = module_name.split('.')
        current = hierarchy
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
        current['_filename'] = filename
        current['_module_name'] = module_name

    def add_hierarchy_to_index(node, prefix="", level=0):
        """Recursively add hierarchical structure to index."""
        indent = "  " * level
        for key in sorted(node.keys()):
            if key.startswith('_'):
                continue
            
            subnode = node[key]
            if '_filename' in subnode:
                # This is a leaf node with actual content
                index_content.append(f"{indent}- [{subnode['_module_name']}]({subnode['_filename'].replace('.md', '')})")
            else:
                # This is a parent node, show as header
                if level == 0:
                    index_content.append(f"\n### {key.title()}")
                elif level == 1:
                    index_content.append(f"\n{indent}**{key.title()}**")
                else:
                    index_content.append(f"\n{indent}*{key.title()}*")
                
                # Recursively add children
                add_hierarchy_to_index(subnode, prefix + key + ".", level + 1)

    # Add the hierarchical structure
    add_hierarchy_to_index(hierarchy)

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
