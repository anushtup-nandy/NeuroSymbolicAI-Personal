#!/usr/bin/env python3
"""
Syntax Check for All Modules
=============================
Checks that all Python files have valid syntax.
"""

import py_compile
import sys
from pathlib import Path

def check_syntax(file_path):
    """Check syntax of a single Python file."""
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

def main():
    """Check all Python files in modules directory."""
    modules_dir = Path("modules")
    python_files = list(modules_dir.rglob("*.py"))
    
    # Also check main.py
    python_files.append(Path("main.py"))
    
    print(f"Checking {len(python_files)} Python files...\n")
    
    errors = []
    for file_path in python_files:
        success, error = check_syntax(file_path)
        if success:
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            errors.append((file_path, error))
    
    print(f"\n{'='*60}")
    if errors:
        print(f"❌ {len(errors)} file(s) with syntax errors:\n")
        for file_path, error in errors:
            print(f"{file_path}:")
            print(f"  {error}\n")
        sys.exit(1)
    else:
        print(f"✅ All {len(python_files)} files have valid syntax!")
        sys.exit(0)

if __name__ == "__main__":
    main()
