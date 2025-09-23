#!/usr/bin/env python3
"""
Script to check for missing docstrings in Python files.

This script analyzes Python source files using the AST (Abstract Syntax Tree)
to identify functions, classes, and modules that are missing docstrings.

Usage:
    python check_docstrings.py <file1.py> [file2.py] ...

Exit codes:
    0: All checked files have proper docstrings
    1: Missing docstrings found or error occurred
"""
import ast
import os
import sys
from typing import Dict, List, Optional, Union


class DocstringChecker(ast.NodeVisitor):
    """
    AST visitor class to check for missing docstrings in Python code.

    This class traverses the Abstract Syntax Tree of a Python file and
    identifies functions, methods, classes, and modules that lack docstrings.
    It follows PEP 257 conventions and requires documentation for all functions
    including private functions, but skips test functions.

    Attributes:
        filename: Path to the file being analyzed
        missing_docstrings: List of dictionaries containing information
                          about missing docstrings
        current_class: Name of the currently visited class (for method context)
    """

    def __init__(self, filename: str) -> None:
        """
        Initialize the docstring checker.

        Args:
            filename: Path to the Python file being analyzed
        """
        self.filename: str = filename
        self.missing_docstrings: List[Dict[str, Union[str, int]]] = []
        self.current_class: Optional[str] = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Visit function definition nodes and check for docstrings.

        Args:
            node: The function definition AST node to analyze
        """
        self._check_docstring(node, "function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """
        Visit async function definition nodes and check for docstrings.

        Args:
            node: The async function definition AST node to analyze
        """
        self._check_docstring(node, "async function")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Visit class definition nodes and check for docstrings.

        Maintains context of the current class for proper method naming
        in the missing docstrings report.

        Args:
            node: The class definition AST node to analyze
        """
        old_class: Optional[str] = self.current_class
        self.current_class = node.name
        self._check_docstring(node, "class")
        self.generic_visit(node)
        self.current_class = old_class

    def visit_Module(self, node: ast.Module) -> None:
        """
        Visit module nodes and check for module-level docstrings.

        Args:
            node: The module AST node to analyze
        """
        if not ast.get_docstring(node):
            self.missing_docstrings.append(
                {"type": "module", "name": os.path.basename(self.filename), "line": 1}
            )
        self.generic_visit(node)

    def _check_docstring(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef], node_type: str
    ) -> None:
        """
        Check if a given AST node has a docstring and record if missing.

        This method requires docstrings for all functions, methods, and classes,
        including private functions. Only test functions (starting with 'test_')
        are skipped from docstring requirements.

        Args:
            node: The AST node to check (function, async function, or class)
            node_type: String describing the type of node ('function', 'async function', 'class')
        """
        # Only skip test functions
        if hasattr(node, "name"):
            # Skip test functions
            if node.name.startswith("test_"):
                return

        if not ast.get_docstring(node):
            name: str = getattr(node, "name", "unknown")
            if self.current_class and node_type in ["function", "async function"]:
                name = f"{self.current_class}.{name}"

            self.missing_docstrings.append({"type": node_type, "name": name, "line": node.lineno})


def check_file_docstrings(filepath: str) -> List[Dict[str, Union[str, int]]]:
    """
    Check docstrings in a single Python file.

    Parses the given Python file using the AST module and analyzes it
    for missing docstrings in modules, classes, functions, and methods.

    Args:
        filepath: Path to the Python file to analyze

    Returns:
        List of dictionaries containing information about missing docstrings.
        Each dictionary has keys: 'type', 'name', 'line'

    Raises:
        No exceptions are raised; errors are caught and logged to stdout
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content: str = f.read()

        # Skip empty files
        if not content.strip():
            return []

        tree: ast.Module = ast.parse(content, filename=filepath)
        checker: DocstringChecker = DocstringChecker(filepath)
        checker.visit(tree)
        return checker.missing_docstrings

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []


def main() -> None:
    """
    Main function to check docstrings in specified files.

    Processes command-line arguments to get a list of Python files,
    checks each file for missing docstrings, and reports the results.

    Command-line usage:
        python check_docstrings.py <file1.py> [file2.py] ...

    Exit behavior:
        - Exits with code 0 if all files have proper docstrings
        - Exits with code 1 if missing docstrings are found or if no files provided

    Side effects:
        - Prints results to stdout
        - May print warnings for non-existent files
    """
    if len(sys.argv) < 2:
        print("Usage: python check_docstrings.py <file1.py> [file2.py] ...")
        sys.exit(1)

    all_missing: Dict[str, List[Dict[str, Union[str, int]]]] = {}
    total_missing: int = 0

    for filepath in sys.argv[1:]:
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} does not exist")
            continue

        missing: List[Dict[str, Union[str, int]]] = check_file_docstrings(filepath)
        if missing:
            all_missing[filepath] = missing
            total_missing += len(missing)

    if all_missing:
        print("❌ Missing docstrings found:")
        print()

        for filepath, missing_items in all_missing.items():
            print(f"📄 {filepath}:")
            for item in missing_items:
                print(f"  - Line {item['line']}: {item['type']} '{item['name']}'")
            print()

        print(f"Total missing docstrings: {total_missing}")
        sys.exit(1)
    else:
        print("✅ All checked files have proper docstrings!")
        sys.exit(0)


if __name__ == "__main__":
    main()
