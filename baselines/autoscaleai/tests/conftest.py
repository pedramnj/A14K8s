# tests/conftest.py
# Ensure project root is on sys.path so pytest can import packages like `agent`, `env`, etc.

import os
import sys

# Compute project root (assumes tests/ is directly under the project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
