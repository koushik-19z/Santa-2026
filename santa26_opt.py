# Santa 2026 single-notebook optimizer: cycles public C++ optimizers (goodpjw2008 elfopt, Wang Jakie strategies,
# Atakan Aldemir optimizers; all Apache-2.0) over the frontier attached as a private dataset.
# Every returned program is replayed by an exact referee; only strictly shorter ones are kept.
# Output: submission.csv (all 1000 puzzles, verified), improved.csv, run_log.txt
import csv, json, os, subprocess, sys, time, random, glob, zipfile, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
csv.field_size_limit(10**9)
CFG = json.loads('{"budget": 10800.0, "sec_per_puzzle": 4.0, "lo": 25}')
T0 = time.time(); BUDGET = CFG["budget"]
W = Path("/kaggle/working"); W.mkdir(exist_ok=True)
LOGF = open(W / "run_log.txt", "a")
def log(*a):
    s = f"[{time.time()-T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
J = os.cpu_count() or 4
INP = Path("/kaggle/input")
SRC = W / "src"; SRC.mkdir(exist_ok=True)
for z in INP.rglob("*.zip"):
    try: zipfile.ZipFile(z).extractall(SRC / z.stem)
    except Exception as e: log("unzip fail", z, e)
def find(name):
    for root in (INP, SRC):
        h = sorted(root.rglob(name))
        if h: return h[0]
    return None
DS = find("frontier.csv").parent
WJ = find("anneal.cpp"); WJ = WJ.parent if WJ else None
AK = find("elf_interface_closure.cpp"); AK = AK.parent if AK else None
log("cpus", J, "ds", DS, "wj", WJ, "ak", AK, "cfg", CFG)
BIN = W / "bin"; BIN.mkdir(exist_ok=True)
def cc(src, out, inc):
    if src is None or not Path(src).exists(): return (out, False, "missing")
    r = subprocess.run(["g++", "-std=c++17", "-O3", "-march=native", "-I", str(inc), "-o", str(BIN / out), str(src)], capture_output=True, text=True)
    return (out, r.returncode == 0, r.stderr[-400:])
jobs = [(find("elfopt.cpp"), "elfopt", DS)]
if WJ:
    for n in ["anneal", "incremental", "compact_rotate", "quartet", "anneal_seeded", "incremental_seeded"]: jobs.append((WJ / f"{n}.cpp", n, WJ))
if AK:
    jobs += [(AK / "elf_interface_closure.cpp", "closure", AK), (AK / "elf_multiplier_frontier.cpp", "frontier", AK)]
OK = {}
with ThreadPoolExecutor(J) as ex:
    for out, ok, err in ex.map(lambda j: cc(*j), jobs):
        OK[out] = ok
        if not ok: log("COMPILE FAIL", out, err)
log("compiled", OK)

P = {}
with open(find("puzzles.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        P[row["id"]] = (json.loads(row["starting_values"]), json.loads(row["targets"]), int(row["value_limit"]))
def read_sub(fn):
    with open(fn, encoding="utf-8") as f: return {r["id"]: json.loads(r["program"]) for r in csv.DictReader(f)}
def replay(pid, prog):
    s, t, L = P[pid]
    if not isinstance(prog, list) or len(prog) > 4096: return None
    reg = list(s) + [None] * (48 - len(s)); miss = set(t) - set(s)
    for ins in prog:
        if not isinstance(ins, list) or len(ins) != 4: return None
        op, i, j, d = ins
        if not all(type(x) is int and 0 <= x < 48 for x in (i, j, d)): return None
        a, b = reg[i], reg[j]
        if a is None or b is None: return None
        if op == "+": v = a + b
        elif op == "-": v = a - b
        elif op == "*": v = a * b
        elif op == "/" and a % b == 0: v = a // b
        else: return None
        if not 1 <= v <= L: return None
        reg[d] = v; miss.discard(v)
    if miss: return None
    if len(json.dumps(prog, separators=(",", ":")).encode()) > 131072: return None
    return len(prog)

S = read_sub(DS / "frontier.csv")
assert len(S) == 1000 and all(replay(k, S[k]) == len(S[k]) for k in P), "frontier invalid"
BASE = {k: len(S[k]) for k in P}
log("frontier total", sum(BASE.values()))
ALTS = {}
af = find("alts.jsonl")
if af:
    for ln in open(af):
        try: r = json.loads(ln)
        except Exception: continue
        ALTS.setdefault(r["k"], []).append(r["p"])
log("alts", sum(len(v) for v in ALTS.values()))

def save():
    tmp = W / "submission.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "program"])
        for k in sorted(P): w.writerow([k, json.dumps(S[k], separators=(",", ":"))])
    os.replace(tmp, W / "submission.csv")
    with open(W / "improved.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "program"])
        for k in sorted(P):
            if len(S[k]) < BASE[k]: w.writerow([k, json.dumps(S[k], separators=(",", ":"))])
save()

# ------------------------------------------------------------------ engines
def seeds_for(k, m=4):
    c = [p for p in ALTS.get(k, []) if len(p) <= len(S[k]) + 2 and p != S[k]]
    random.shuffle(c); return c[:m]
def run_wj(name, batch, rounds, tmo):
    rows = [str(len(batch))]
    for k in batch:
        s, t, L = P[k]; tt = sorted(set(t))
        tapes = [S[k]] + (seeds_for(k) if name.endswith("_seeded") else [])
        rows += [f"{L} 48 {rounds} 0 {random.randrange(1, 2**31)}", " ".join(map(str, (len(s), *s))), " ".join(map(str, (len(tt), *tt))), str(len(tapes))]
        for tp in tapes:
            rows.append(str(len(tp))); rows += [f"{op} {x} {y} {d}" for op, x, y, d in tp]
    r = subprocess.run([str(BIN / name)], input="\n".join(rows) + "\n", capture_output=True, text=True, timeout=tmo)
    out = {}
    for k, line in zip(batch, r.stdout.splitlines()):
        tok = line.split()
        if tok: out[k] = [[tok[i], int(tok[i + 1]), int(tok[i + 2]), int(tok[i + 3])] for i in range(1, len(tok), 4)]
    return out
CL = ["90 26 7 {salt} 4150", "92 32 4 {salt} 5250", "100 28 9 {salt} 4950"]
FR = ["8 4200 9 10 9 {salt} 1500", "14 8000 12 13 9 {salt} 3000", "8 4200 9 10 13 {salt} 1500"]
def run_ak(name, batch, args, tmo):
    cases = []
    for k in batch:
        s, t, L = P[k]; tt = sorted(set(t)); prog = S[k]
        cases.append("\n".join([f"{k} {L} {len(s)} {len(tt)} {len(prog)}", " ".join(map(str, s)), " ".join(map(str, tt))]
                               + [f"{op} {x} {y} {d}" for op, x, y, d in prog]))
    argv = [str(BIN / name)] + args.format(salt=random.randrange(1, 2**31)).split()
    r = subprocess.run(argv, input="\n".join(cases) + "\n", capture_output=True, text=True, timeout=tmo)
    lines = r.stdout.splitlines(); out, pos = {}, 0
    while pos < len(lines):
        head = lines[pos].split(); pos += 1
        if len(head) < 2: continue
        try: k, n = head[0], int(head[1])
        except ValueError: continue
        out[k] = [[x[0], int(x[1]), int(x[2]), int(x[3])] for x in (l.split() for l in lines[pos:pos + n])]; pos += n
    return out
def run_elf(batch, per, tmo):
    txt = []
    for k in batch:
        s, t, L = P[k]; prog = S[k]
        h = ["P", k, str(len(s))] + [str(x) for x in s] + [str(len(t))] + [str(x) for x in t] + [str(L), "48", str(len(prog))]
        txt.append("\n".join([" ".join(h)] + ["%s %d %d %d" % tuple(x) for x in prog]) + "\n")
    r = subprocess.run([str(BIN / "elfopt"), "--time", str(per), "--seed", str(random.randrange(1, 10**9)), "--kicks", "1000", "--kslack", str(random.choice([1, 2, 2, 3]))],
                       input="".join(txt), capture_output=True, text=True, timeout=tmo)
    res, lines, i = {}, r.stdout.split("\n"), 0
    while i < len(lines):
        ln = lines[i].split()
        if ln and ln[0] == "R":
            pid, n = ln[1], int(ln[3])
            res[pid] = [[o, int(a), int(b), int(d)] for o, a, b, d in (lines[i + 1 + t].split() for t in range(n))]
            i += n + 1
        else: i += 1
    return res
def call(name, batch, par, tmo):
    if name == "elf": return run_elf(batch, par, tmo)
    if name == "closure": return run_ak("closure", batch, CL[random.randrange(3)], tmo)
    if name == "frontier": return run_ak("frontier", batch, FR[random.randrange(3)], tmo)
    return run_wj(name, batch, par, tmo)
def absorb(out):
    g = 0
    for k, prog in out.items():
        if k in P:
            n = replay(k, prog)
            if n is not None and n < len(S[k]): g += len(S[k]) - n; S[k] = prog
    return g

# ------------------------------------------------------------------ calibration
engines = [e for e in ["closure", "frontier", "anneal_seeded", "incremental_seeded", "compact_rotate", "elf", "quartet", "anneal"]
           if OK.get(e if e != "elf" else "elfopt")]
cal_ids = sorted([k for k in P if 90 <= len(S[k]) <= 150])[:J]
SPP = {}   # seconds per puzzle at the calibration parameter
for e in engines:
    par = 5 if e == "elf" else 200
    t1 = time.time()
    try:
        with ThreadPoolExecutor(J) as ex: outs = list(ex.map(lambda k: call(e, [k], par, 900), cal_ids))
        g = sum(absorb(o) for o in outs)
        SPP[e] = max(0.05, time.time() - t1)   # J puzzles ran in parallel, one per core -> wall time = sec per puzzle per core
        log("calib", e, "par", par, "sec/puzzle", round(SPP[e], 2), "gain", g)
    except Exception as ex_:
        log("calib FAIL", e, repr(ex_)[:200])
engines = [e for e in engines if e in SPP]
save()

# ------------------------------------------------------------------ main loop
ids_all = [k for k in P if len(S[k]) >= CFG["lo"]]
focus = [k for k in ids_all if 40 <= len(S[k]) <= 200]
TGT = CFG["sec_per_puzzle"]
cycle = 0
while time.time() - T0 < BUDGET - 600:
    cycle += 1
    for e in engines:
        left = BUDGET - 600 - (time.time() - T0)
        if left < 120: break
        pool_ids = focus if cycle % 2 == 1 else ids_all
        ids = pool_ids[:]; random.shuffle(ids)
        # scale the engine parameter so one pass costs ~TGT seconds per puzzle, never more than the time left
        tgt = min(TGT, left * J / max(1, len(ids)))
        if e == "elf": par = max(2.0, round(tgt, 1)); est = par
        elif e in ("closure", "frontier"): par = 0; est = SPP[e]
        else: par = int(max(60, min(30000, 200 * tgt / max(0.05, SPP[e])))); est = SPP[e] * par / 200
        if est * len(ids) / J > left:
            ids = ids[:max(J, int(left * J / max(0.1, est)))]
        bs = 2
        batches = [ids[i:i + bs] for i in range(0, len(ids), bs)]
        tmo = max(120, est * bs * 6 + 60)
        t1 = time.time(); g = 0; errs = 0
        def job(b):
            if time.time() - T0 > BUDGET - 420: return {}
            try: return call(e, b, par, tmo)
            except Exception: return None
        with ThreadPoolExecutor(J) as ex:
            for out in ex.map(job, batches):
                if out is None: errs += 1; continue
                g += absorb(out)
        save()
        log(f"cycle {cycle} {e} par={par} puzzles={len(ids)} took={time.time()-t1:.0f}s gain={g} errs={errs} total={sum(len(S[k]) for k in P)}")

save()
tot = sum(replay(k, S[k]) for k in P)
log("FINAL verified total", tot, "gain", sum(BASE.values()) - tot)
