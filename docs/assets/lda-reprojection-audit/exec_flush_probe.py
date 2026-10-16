"""Does Python's pre-exec stdout buffer survive os.execv?

Prints a marker WITHOUT flushing to a <block-buffered file>, then execv's
/bin/echo.  If the marker appears in the output file, execv flushed the
Python-level buffer first; if it does not, the pre-exec output was lost.

Usage:  .venv/bin/python exec_flush_probe.py > out.txt 2>&1
"""
import os
import sys

print("MARKER_PRE_EXEC_NO_FLUSH")
sys.stdout.flush()
print("MARKER_PRE_EXEC_BUFFERED")  # left in the userspace buffer
os.execv("/bin/echo", ["echo", "AFTER_EXEC"])
