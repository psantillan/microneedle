"""Score a packed .npk on the six demo tools, through the engine that ships.

The upstream training harness prints its own eval, which measures a different
generation path. This measures the artifact you are about to flash: same .npk,
same int4 kernels, same C engine.

    python3 tools/eval_tools.py weights/needle_tools_g512.npk
"""
import json, os, re, subprocess, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tokenizer'))
import bpe
tok=bpe.NeedleBPE(); sys.path.insert(0, os.path.join(REPO, 'train'))
from make_dataset import TOOLS as _T
TOOLS = list(_T.values())
if len(sys.argv) < 2:
    raise SystemExit(__doc__)
CASES=[("What's the weather in Oslo?","get_weather"),("When does the sun set in Reykjavik?","get_sun_times"),
       ("Tell me about tardigrades","look_up"),("Set a timer for 5 minutes","set_timer"),
       ("Play a chime","play_tone"),("How long have you been running?","board_status"),
       ("Write me a poem about the sea",None)]
vecfile = '/tmp/ne_tools_vec.txt'
enc=[bpe.build_encoder_input(tok,q,TOOLS) for q,_ in CASES]
tj=enc[0][1]
lines=[f"c{i} {len(ids)} {' '.join(map(str,ids))} 1 1" for i,(ids,_) in enumerate(enc)]
open(vecfile,'w').write("\n".join(lines)+"\n")
host=os.path.join(REPO,'engine','needle_host')
try:
    r=subprocess.run([host, sys.argv[1], vecfile],capture_output=True,text=True)
except OSError as e:
    raise SystemExit(f"cannot run {host}: {e}\n  build it first: make -C {os.path.join(REPO,'engine')}")
out=r.stdout
got={}; cur=None
for l in out.splitlines():
    m=re.match(r'^(c\d+)\s',l)
    if m: cur=m.group(1)
    if l.strip().startswith('got:') and cur: got[cur]=[int(x) for x in l.split(':')[1].split()]
# needle_host exits 1 on a parity miss too, and it always misses here (the
# vector file carries a dummy expected sequence), so the run must be judged on
# whether every case came back rather than on the exit status. Without that
# check a dead engine scores 1/7: the no-call case reads an empty answer as
# correct.
missing=[i for i in range(len(CASES)) if f"c{i}" not in got]
if missing:
    raise SystemExit(f"{host} returned no output for cases {missing} (exit {r.returncode})\n"
                     f"{(r.stderr.strip() or out.strip())[:800]}")
ok=0
for i,(q,want) in enumerate(CASES):
    t=tok.decode(got.get(f"c{i}",[]))
    try:
        c=json.loads(t); n=c[0]['name'] if c else None; a=c[0].get('arguments') if c else None
    except Exception: n,a='(unparseable)',None
    hit=(n==want) if want else (n is None); ok+=hit
    print(f"  {'OK ' if hit else 'XX '} {q[:36]:<36} -> {str(n):<14} {json.dumps(a) if a else ''}")
print(f"\n  {ok}/{len(CASES)} correct with ALL SIX tools offered ({len(tok.encode(tj))+8} token prompt)")
