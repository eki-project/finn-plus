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


def generate_markdown_documentation(config_data: Dict[str, Any], output_file: str) -> None:
    """Generate markdown documentation from parsed config data."""

    with open(output_file, "w") as f:
        f.write("# DataflowBuildConfig Documentation\n\n")

        # Write class description
        if config_data["class_docstring"]:
            f.write(f"{config_data['class_docstring']}\n\n")

        # Write table of contents
        f.write("## Table of Contents\n\n")
        f.write("1. [Enumerations](#enumerations)\n")
        for enum_name in config_data["enums"].keys():
            f.write(f"   - [{enum_name}](#{enum_name.lower()})\n")
        f.write("2. [Configuration Fields](#configuration-fields)\n\n")

        # Write enum documentation
        if config_data["enums"]:
            f.write("## Enumerations\n\n")
            for enum_name, enum_data in config_data["enums"].items():
                f.write(f"### {enum_name}\n\n")
                if enum_data["docstring"]:
                    f.write(f"{enum_data['docstring']}\n\n")

                if enum_data["values"]:
                    f.write("| Name | Value |\n")
                    f.write("|------|-------|\n")
                    for value_data in enum_data["values"]:
                        name: str = value_data["name"]
                        value: str = value_data["value"].replace("|", "\\|")  # Escape pipes
                        f.write(f"| `{name}` | {value} |\n")
                    f.write("\n")

        # Write configuration fields
        # f.write("## Configuration Fields\n\n")
        # f.write("| Field Name | Type | Default Value | Description |\n")
        # f.write("|------------|------|---------------|-------------|\n")

        # for field in config_data["fields"]:
        #     name: str = field["name"]
        #     field_type: str = field["type"].replace("|", "\\|")  # Escape pipes for markdown
        #     default: str = field["default"] if field["default"] is not None else "None"
        #     default = str(default).replace("|", "\\|")  # Escape pipes for markdown
        #     description: str = field["description"] if field["description"] else ""
        #     description = description.replace("|", "\\|")  # Escape pipes for markdown

        #     f.write(f"| `{name}` | `{field_type}` | `{default}` | {description} |\n")

        # f.write("\n")
        
        f.write("## Configuration Fields\n\n")
        
        for field in config_data["fields"]:
            name: str = field["name"]
            field_type: str = field["type"].replace("|", "\\|")  # Escape pipes for markdown
            default: str = field["default"] if field["default"] is not None else "None"
            default = str(default).replace("|", "\\|")  # Escape pipes for markdown
            description: str = field["description"] if field["description"] else "*No description available*"
            description = description.replace("|", "\\|")  # Escape pipes for markdown

            # Create a section for each field
            f.write(f"### `{name}`\n\n")
            
            # Create a transposed table for this field
            f.write("| Property | Value |\n")
            f.write("|----------|-------|\n")
            f.write(f"| Type | `{field_type}` |\n")
            f.write(f"| Default Value | `{default}` |\n")
            f.write(f"| Description | {description} |\n")
            f.write("\n")

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
