#!/usr/bin/env python3
"""Research → product pointer.

Multi-F accuracy-vs-F (F∈{0,4,6,8..12}) was measured on warm-hybrid and is
frozen in docs/notebook/RESULTS-HYBRID.md. The product path is pinned to F=4; this script
delegates to tools/eval_warmf4.py.

    python3 tools/eval_warmschema.py [weights/needle_tools_g512.npk]
"""
import os
import runpy
import sys

if __name__ == '__main__':
    if any(a.lstrip('-').isdigit() for a in sys.argv[1:]):
        print("NOTE: multi-F sweep is research-only (see docs/notebook/RESULTS-HYBRID.md).",
              file=sys.stderr)
        print("Product path is F=4; running eval_warmf4.py.", file=sys.stderr)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'eval_warmf4.py')
    args = [a for a in sys.argv[1:]
            if not a.lstrip('-').isdigit()]
    sys.argv = [path] + args
    runpy.run_path(path, run_name='__main__')
