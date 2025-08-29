#!/usr/bin/env python
import argparse
import ast
from typing import Any, Dict, List, Optional


def get_file_content(path: str) -> str:
    """Read file content from given path."""
    with open(path, "r") as f:
        content = f.read()
        return content


def extract_field_comment(source_lines: List[str], lineno: int) -> Optional[str]:
    """Extract field documentation comments that appear before the field definition."""
    # Look for comment lines before the current line that start with #:
    comment_lines: List[str] = []
    current_line: int = lineno - 2  # Start one line before the field definition

    while current_line >= 0:
        line: str = source_lines[current_line].strip()
        if line.startswith("#:"):
            # Extract the comment text
            comment_text: str = line[2:].strip()
            comment_lines.insert(0, comment_text)
            current_line -= 1
        elif line == "":
            # Skip empty lines
            current_line -= 1
        else:
            # Stop when we hit non-comment, non-empty line
            break

    # Join multiple comment lines with space
    return " ".join(comment_lines) if comment_lines else None


def get_enum_values(node: ast.ClassDef) -> List[Dict[str, str]]:
    """Extract enum values from an enum class node."""
    enum_values: List[Dict[str, str]] = []
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    value: str = ""
                    if isinstance(item.value, ast.Constant):
                        value = repr(item.value.value)
                    elif isinstance(item.value, ast.Str):
                        value = repr(item.value.s)
                    enum_values.append({"name": target.id, "value": value})
    return enum_values


def parse_dataflow_build_config(content: str) -> Dict[str, Any]:
    """Parse the DataflowBuildConfig class and extract documentation."""
    parsed: ast.Module = ast.parse(content)
    source_lines: List[str] = content.split("\n")

    # Find enums and classes
    enums: Dict[str, Dict[str, Any]] = {}
    dataflow_config: Optional[ast.ClassDef] = None

    for node in ast.walk(parsed):
        # Extract enum classes
        if isinstance(node, ast.ClassDef):
            # Check if it's an enum by looking at base classes
            is_enum: bool = any(
                (isinstance(base, ast.Name) and base.id == "Enum")
                or (isinstance(base, ast.Attribute) and base.attr == "Enum")
                for base in node.bases
            )

            if is_enum:
                docstring: Optional[str] = ast.get_docstring(node)
                enum_values: List[Dict[str, str]] = get_enum_values(node)
                enums[node.name] = {"docstring": docstring, "values": enum_values}
            elif node.name == "DataflowBuildConfig":
                dataflow_config = node

    # Parse DataflowBuildConfig class
    if dataflow_config is None:
        raise ValueError("DataflowBuildConfig class not found")

    class_docstring: Optional[str] = ast.get_docstring(dataflow_config)

    # Extract field documentation
    fields: List[Dict[str, Any]] = []
    for node in dataflow_config.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            field_name: str = node.target.id

            # Get type annotation
            type_annotation: str = (
                ast.unparse(node.annotation) if hasattr(ast, "unparse") else str(node.annotation)
            )

            # Get default value
            default_value: Optional[str] = None
            if node.value:
                if isinstance(node.value, ast.Constant):
                    default_value = repr(node.value.value)
                elif isinstance(node.value, ast.Name):
                    default_value = node.value.id
                elif isinstance(node.value, ast.Attribute):
                    default_value = (
                        ast.unparse(node.value) if hasattr(ast, "unparse") else str(node.value)
                    )
                else:
                    default_value = (
                        ast.unparse(node.value) if hasattr(ast, "unparse") else str(node.value)
                    )

            # Get field comment (documentation)
            comment: Optional[str] = extract_field_comment(source_lines, node.lineno)

            fields.append(
                {
                    "name": field_name,
                    "type": type_annotation,
                    "default": default_value,
                    "description": comment,
                }
            )

    return {"class_docstring": class_docstring, "fields": fields, "enums": enums}


def create_enum_links(text: str, enum_names: List[str]) -> str:
    """Replace enum names in text with markdown links to their documentation sections."""
    linked_text = text
    for enum_name in enum_names:
        # Replace enum name with link using backticks for code styling
        import re

        pattern = r"\b" + re.escape(enum_name) + r"\b"
        replacement = f"[`{enum_name}`](#🏷️-{enum_name.lower()})"
        linked_text = re.sub(pattern, replacement, linked_text)
    return linked_text


def format_code_with_links(text: str, enum_names: List[str]) -> str:
    """Format text as code while preserving enum links."""
    import re

    # Find all enum names and their positions
    enum_positions = []
    for enum_name in enum_names:
        pattern = r"\b" + re.escape(enum_name) + r"\b"
        for match in re.finditer(pattern, text):
            enum_positions.append((match.start(), match.end(), enum_name))

    if not enum_positions:
        # No enums found, wrap everything in code
        return f"`{text}`"

    # Sort by position
    enum_positions.sort()

    # Build the result with code formatting for non-enum parts
    result = ""
    last_end = 0

    for start, end, enum_name in enum_positions:
        # Add code-formatted text before this enum
        if start > last_end:
            before_text = text[last_end:start]
            if before_text:
                result += f"`{before_text}`"

        # Add the enum link
        result += f"[`{enum_name}`](#%EF%B8%8F-{enum_name.lower()})"
        last_end = end

    # Add any remaining text after the last enum
    if last_end < len(text):
        after_text = text[last_end:]
        if after_text:
            result += f"`{after_text}`"

    return result


def generate_markdown_documentation(config_data: Dict[str, Any], output_file: str) -> None:
    """Generate markdown documentation from parsed config data."""

    with open(output_file, "w") as f:
        f.write("# DataflowBuildConfig Documentation\n\n")

        # Write class description
        if config_data["class_docstring"]:
            f.write(f"{config_data['class_docstring']}\n\n")

        # Sort enums and fields by name for consistent ordering
        sorted_enum_names = sorted(config_data["enums"].keys())
        sorted_fields = sorted(config_data["fields"], key=lambda x: x["name"])

        # Create list of enum names for linking
        enum_names = list(config_data["enums"].keys())

        # Write enum documentation
        if config_data["enums"]:
            f.write("## Enumerations\n\n")
            for enum_name in sorted_enum_names:
                enum_data = config_data["enums"][enum_name]
                f.write(f"### 🏷️ {enum_name}\n\n")

                if enum_data["docstring"]:
                    f.write(f"> {enum_data['docstring']}\n\n")

                if enum_data["values"]:
                    f.write("| Option | Value |\n")
                    f.write("|--------|-------|\n")
                    for value_data in enum_data["values"]:
                        name = value_data["name"]
                        value = value_data["value"].replace("|", "\\|")
                        # Description column for future use
                        f.write(f"| `{name}` | {value} |\n")
                    f.write("\n")

        # Write configuration fields - Enhanced formatting
        f.write("## Configuration Fields\n\n")

        for field in sorted_fields:
            name = field["name"]
            field_type = field["type"].replace("|", "\\|")
            default = field["default"] if field["default"] is not None else "None"
            default = str(default).replace("|", "\\|")
            description = (
                field["description"] if field["description"] else "*No description available*"
            )
            description = description.replace("|", "\\|")

            # Create enum links in type and default value with proper code formatting
            field_type_linked = format_code_with_links(field_type, enum_names)
            default_linked = format_code_with_links(default, enum_names)
            description_linked = create_enum_links(description, enum_names)

            # Add status icon based on documentation
            status_icon = "📝" if field["description"] else "❓"

            f.write(f"### {status_icon} `{name}`\n\n")

            # Enhanced table with better formatting
            f.write("| Property | Value |\n")
            f.write("|----------|-------|\n")
            f.write(f"| **Type** | {field_type_linked} |\n")
            f.write(f"| **Default** | {default_linked} |\n")
            f.write(f"| **Description** | {description_linked} |\n")
            f.write("\n")

            # Add a subtle separator between fields
            f.write("---\n\n")

        f.write("\n")


def main() -> int:
    """Main function to parse config and generate documentation."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate documentation for DataflowBuildConfig"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="../src/finn/builder/build_dataflow_config.py",
        help="Path to build_dataflow_config.py file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="DataflowBuildConfig_Documentation.md",
        help="Output markdown file name",
    )
    parser.add_argument(
        "--strict",
        "-s",
        action="store_true",
        help="Fail if any configuration fields are missing documentation",
    )

    args: argparse.Namespace = parser.parse_args()

    try:
        # Read the source file
        print(f"Reading {args.input}...")
        content: str = get_file_content(args.input)

        # Parse the configuration
        print("Parsing DataflowBuildConfig class...")
        config_data: Dict[str, Any] = parse_dataflow_build_config(content)

        # Generate markdown documentation
        print(f"Generating documentation: {args.output}...")
        generate_markdown_documentation(config_data, args.output)

        # Report statistics
        fields_with_desc: int = sum(1 for field in config_data["fields"] if field["description"])

        # Check for undocumented fields and handle strict mode
        undocumented_fields: List[str] = [
            field["name"] for field in config_data["fields"] if not field["description"]
        ]

        print("\n" + "=" * 60)
        print("DOCUMENTATION GENERATION COMPLETE")
        print("=" * 60)
        print(f"📄 Output file: {args.output}")
        print(f"📊 Configuration fields: {len(config_data['fields'])}")
        print(f"📝 Fields with descriptions: {fields_with_desc}/{len(config_data['fields'])}")
        print(f"🏷️  Enumerations: {len(config_data['enums'])}")

        if undocumented_fields:
            missing_desc: int = len(undocumented_fields)
            print(f"⚠️  Fields missing descriptions: {missing_desc}")

            if args.strict:
                print(f"\n❌ STRICT MODE: Found {missing_desc} undocumented field(s):")
                for field_name in undocumented_fields:
                    print(f"  - {field_name}")
                print("\nGeneration failed due to undocumented fields.")
                return 1

        print("\nEnumerations found:")
        for enum_name in config_data["enums"].keys():
            print(f"  - {enum_name}")

        print(f"\n✅ Documentation successfully generated in {args.output}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
