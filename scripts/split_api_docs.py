#!/usr/bin/env python3
"""Split the generated API documentation into separate files per module."""

import re
import sys
from pathlib import Path


def split_api_documentation(api_file_path="docs/api.md", output_dir="wiki-content"):
    """Split docs/api.md into separate files for each category.

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
    index_content.append("Welcome to the comprehensive API reference for the FINN+ framework.")
    index_content.append("This documentation is generated automatically from source code.")
    index_content.append("")
    index_content.append("## 📋 Module Overview")
    index_content.append("")
    index_content.append("The FINN+ framework is organized into the following main components:")
    index_content.append("")

    # Group modules by category
    categories = {}

    # Process each module (skip first element which is content before first module)
    for i in range(1, len(modules), 2):
        if i + 1 < len(modules):
            module_name = modules[i].strip()
            module_content = modules[i + 1].strip()

            # Skip base/parent modules with minimal content (typically just package docstrings)
            # These are usually modules like "finn.builder" that only contain brief descriptions
            if (
                module_content and len(module_content) > 100
            ):  # Only process modules with substantial content
                # Extract category from module name (second part after 'finn.')
                parts = module_name.split(".")
                if len(parts) >= 2:
                    category = parts[1]  # e.g., 'analysis', 'builder', 'custom_op', etc.

                    if category not in categories:
                        categories[category] = []

                    # Special handling for finn.builder.build_dataflow_config
                    if (
                        module_name == "finn.builder.build_dataflow_config"
                        or module_name == "finn.builder.build\\_dataflow\\_config"
                    ):
                        replacement_content = """This module contains the DataflowBuildConfig class and related functionality for configuring FINN+ dataflow builds.

📖 **For detailed documentation, please visit:**
**[DataflowBuildConfig Documentation](https://github.com/eki-project/finn-plus/wiki/DataflowBuildConfig-Documentation)**

"""
                        categories[category].append((module_name, replacement_content))
                        print(
                            f"✅ Added {module_name} to category '{category}' (replaced with wiki link)"
                        )
                    else:
                        categories[category].append((module_name, module_content))
                        print(
                            f"✅ Added {module_name} to category '{category}' ({len(module_content)} chars)"
                        )
                else:
                    print(f"⏭️  Skipped {module_name} (unable to determine category)")
            else:
                print(f"⏭️  Skipped {module_name} (base module, {len(module_content)} chars)")

    # Create one file per category
    category_files = []
    for category, module_list in categories.items():
        # Create safe filename for category
        safe_filename = category.replace(" ", "-").replace("\\", "")
        filename = f"finn.{safe_filename}.md"
        category_files.append((category, filename))

        # Create category file content
        file_content = []

        # Add title and description
        safe_category_title = category.replace("\\", "")
        file_content.append(f"This page contains the complete API reference for all modules in the `finn.{safe_category_title}` package.")
        file_content.append("")

        # Generate table of contents
        file_content.append("## Table of Contents")
        file_content.append("")
        module_list.sort()  # Sort modules alphabetically
        for module_name, _ in module_list:
            # Create anchor link from module name
            anchor = module_name.replace(".", "").replace("\\", "").lower()
            file_content.append(f"- [{module_name}](#{anchor})")
        file_content.append("")
        file_content.append("---")
        file_content.append("")

        # Add all modules from this category
        for module_name, module_content in module_list:
            file_content.append(f"## {module_name}")
            file_content.append("")
            file_content.append(module_content)
            file_content.append("")
            file_content.append("---")
            file_content.append("")

        file_content.append("📚 **Navigation**: [← Back to API Documentation](API-Documentation)")
        file_content.append("")
        file_content.append(
            "*This page was generated automatically from source code documentation.*"
        )

        # Write category file
        category_file = wiki_dir / filename
        with open(category_file, "w") as f:
            f.write("\n".join(file_content))

        print(f"✅ Created {filename} with {len(module_list)} modules")

    # Create simple structure for the index (no hierarchy, just categories)
    category_emojis = {
        "analysis": "🔍",
        "benchmarking": "⚡",
        "builder": "🏗️",
        "core": "⚙️",
        "custom_op": "🔧",
        "interface": "🔌",
        "qnn-data": "📊",
        "transformation": "🔄",
        "util": "🛠️",
    }

    # Add category links to the index
    for category in sorted(categories.keys()):
        emoji = category_emojis.get(category, "🔧")
        safe_filename = category.replace(" ", "-").replace("\\", "")
        filename = f"finn.{safe_filename}"
        index_content.append(f"- {emoji} **[{category.title()}]({filename})**")

    # Write main index file
    with open(wiki_dir / "FINN-API-Documentation.md", "w") as f:
        f.write("\n".join(index_content))

    print(f"✅ Created main API index with {len(categories)} categories")
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
