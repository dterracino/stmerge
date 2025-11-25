#!/usr/bin/env python3
"""
Model Merger - Entry Point

Simple runner script that imports and executes the CLI.
This keeps the package clean and allows it to be imported as a library
without executing anything.
"""

import sys
from model_merger.cli import main

if __name__ == '__main__':
    sys.exit(main())