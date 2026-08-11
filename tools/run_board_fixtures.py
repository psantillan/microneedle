"""Drive the board through all 8 fixture cases over serial and diff against the
JAX oracle.

This is the acceptance test for the PIE path, which adds int8 activation
quantization the oracle does not have. Token-level divergence here is a
measurement of that quantization; the binding requirement is that the JSON
stays valid and the tool call stays correct.

RITUAL GUARD: the references are outputs of a SPECIFIC model. Flash the
matching npk (v3 for the stock references, or the model named by NE_REF's
provenance) before running, or every thin-margin case reports DIFF against
a model that was never asked. This mistake has now been made twice.
"""

import json
import sys
import time

import serial

import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF  = os.environ.get("NE_REF", os.path.join(REPO, "weights", "reference_int4p_int8e.json"))
PORT = os.environ.get("NE_PORT", "/dev/ttyACM1")


def main():
    ref = json.load(open(REF))
    s = serial.Serial(PORT, 115200, timeout=3)
    # reset and wait for the smoke test + ready
    s.setDTR(False); s.setRTS(True); time.sleep(0.15); s.setRTS(False)
    boot = b""
    t0 = time.time()
    while b"ready" not in boot and time.time() - t0 < 180:
        boot += s.read(4096)
    boot_txt = boot.decode("utf-8", "replace")
    boot_fail = False
    for ln in boot_txt.splitlines():
        # UT/AB are the boot self-tests: a miswired kernel is reported here,
        # before any fixture runs
        if ln.startswith(("R ", "copied", "engine", "npk:", "smoke", "SMOKE", "UT ", "AB ")):
            print("boot|", ln[:120])
            boot_fail |= "FAIL" in ln
    if boot_fail:
        print("kernel unit test FAILED at boot -- fixture results would be meaningless")
        sys.exit(1)
    if b"ready" not in boot:
        print("board never became ready"); sys.exit(1)

    passes = seq_same = 0
    rows = []
    for case in ref["cases"]:
        ids = case["enc_ids"]
        cmd = (f"G {len(ids)} " + " ".join(map(str, ids)) + "\n").encode()
        # pace the write: the board's UART RX ring is finite and fgets drains
        # it only as fast as the idle loop runs
        for off in range(0, len(cmd), 128):
            s.write(cmd[off:off + 128])
            time.sleep(0.02)
        line = b""
        t0 = time.time()
        while time.time() - t0 < 300:
            chunk = s.read(4096)
            if chunk:
                line += chunk
                if b"R " in line and b"\n" in line[line.index(b"R "):]:
                    break
        txt = line.decode("utf-8", "replace")
        rline = next((l for l in txt.splitlines() if l.startswith("R ")), None)
        if not rline:
            print(f"{case['id']:<22} NO RESPONSE"); continue
        # Serial log lines (wifi/httpd) can interleave mid-line, welding
        # fragments like "503W" onto a token. Take the leading digits of
        # each field: the count then bounds the token slice, so a welded
        # tail can only corrupt the free-text stats, never a token.
        import re
        parts = rline.split()
        digits = [re.match(r"\d+", p) for p in parts[1:]]
        nums = [int(m.group()) for m in digits if m]
        n = nums[0]
        if len(nums) < 1 + n:
            print(f"{case['id']:<22} GARBLED ({rline[:60]!r})"); continue
        got = nums[1:1 + n]
        stats = " ".join(p for p in parts[2 + n:] if "=" in p)
        same = got == case["gen_ids"]
        seq_same += same
        passes += 1
        print(f"{case['id']:<22} {'SAME' if same else 'DIFF':<5} "
              f"got={n:<3} ref={len(case['gen_ids']):<3} {stats}")
        if not same:
            print(f"    got: {' '.join(map(str, got))}")
        rows.append({"id": case["id"], "same": same, "got": got,
                     "ref": case["gen_ids"], "stats": stats})
        if not same:
            i = 0
            while i < min(n, len(case["gen_ids"])) and got[i] == case["gen_ids"][i]:
                i += 1
            print(f"    first divergence at {i}")

    print(f"\n{seq_same}/{passes} sequences identical to the selected reference")
    if passes != len(ref["cases"]) or seq_same != len(ref["cases"]):
        raise SystemExit(1)   # a gate that cannot fail is not a gate
    if len(sys.argv) > 1:
        json.dump(rows, open(sys.argv[1], "w"), indent=2)
        print(f"wrote {sys.argv[1]}")


if __name__ == "__main__":
    main()
