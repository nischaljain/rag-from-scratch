"""Allows `python -m rag ...`."""

import sys

from .cli import main

sys.exit(main())
