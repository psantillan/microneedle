"""Flatten the JAX oracle dump into the vectors file the host harness reads.

engine/main_host.c parses a flat text format with fscanf rather than JSON, so
the parity gate stays dependency-free and compiles anywhere. This is the
converter that produces it. It existed only in a docstring before, which meant
the load-bearing host gate was fed by a file nothing in the repo could
regenerate.

    python3 tools/ref_to_vectors.py weights/reference_int4p_int8e.json \\
                                    weights/vectors_int4p_int8e.txt
"""

import json
import sys


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    cases = json.load(open(src))["cases"]
    with open(dst, "w") as f:
        for c in cases:
            enc, gen = c["enc_ids"], c["gen_ids"]
            f.write(f"{c['id']} {len(enc)} {' '.join(map(str, enc))} "
                    f"{len(gen)} {' '.join(map(str, gen))}\n")
    print(f"{dst}: {len(cases)} cases")


if __name__ == "__main__":
    main()
