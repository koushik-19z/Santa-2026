# Santa 2026 LNS kernel: runs elfopt (goodpjw2008, Apache-2.0) on a shard of the frontier, keeps verified improvements.
import csv, json, os, subprocess, sys, time, random, glob
from pathlib import Path
csv.field_size_limit(10**9)
CFG = json.loads('{"budget": 600.0, "tper": 10.0, "lo": 60, "hi": 200, "shard": 0, "nshards": 100, "extra": "--kicks 1000 --kslack 2"}')
T0 = time.time(); BUDGET = CFG["budget"]
DS = sorted(Path("/kaggle/input").rglob("elfopt.cpp"))[0].parent
W = Path("/kaggle/working"); W.mkdir(exist_ok=True)
print("ds", DS, "cpus", os.cpu_count(), flush=True)
r = subprocess.run(["g++", "-std=c++17", "-O3", "-march=native", "-o", str(W / "elfopt"), str(DS / "elfopt.cpp")], capture_output=True, text=True)
if r.returncode: print(r.stderr[:5000]); raise SystemExit(1)

P = {}
with open(DS / "puzzles.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        P[row["id"]] = (json.loads(row["starting_values"]), json.loads(row["targets"]), int(row["value_limit"]))
def read_sub(fn):
    with open(fn, encoding="utf-8") as f: return {r["id"]: json.loads(r["program"]) for r in csv.DictReader(f)}
def replay(pid, prog):
    s, t, L = P[pid]
    if len(prog) > 4096: return None
    reg = list(s) + [None] * (48 - len(s)); miss = set(t) - set(s)
    for op, i, j, d in prog:
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
    return None if miss else len(prog)
def encode(pid, prog):
    s, t, L = P[pid]
    h = ["P", pid, str(len(s))] + [str(x) for x in s] + [str(len(t))] + [str(x) for x in t] + [str(L), "48", str(len(prog))]
    return "\n".join([" ".join(h)] + ["%s %d %d %d" % tuple(x) for x in prog]) + "\n"
def parse(text):
    res, lines, k = {}, text.split("\n"), 0
    while k < len(lines):
        ln = lines[k].split()
        if ln and ln[0] == "R":
            pid, n = ln[1], int(ln[3])
            res[pid] = [[o, int(i), int(j), int(d)] for o, i, j, d in (lines[k + 1 + t].split() for t in range(n))]
            k += n + 1
        else: k += 1
    return res

S = read_sub(DS / "frontier.csv")
ids = sorted(P, key=lambda k: -len(S[k]))
ids = [k for k in ids if CFG["lo"] <= len(S[k]) <= CFG["hi"]]
ids = ids[CFG["shard"]::CFG["nshards"]]
print("puzzles", len(ids), "total", sum(len(S[k]) for k in ids), flush=True)
J = os.cpu_count() or 4
base = {k: len(S[k]) for k in ids}
ALTS = {}
if (DS / "alts.jsonl").exists():
    idset = set(ids)
    for ln in open(DS / "alts.jsonl"):
        try: r = json.loads(ln)
        except Exception: continue
        if r["k"] in idset: ALTS.setdefault(r["k"], []).append(r["p"])
print("alts for", len(ALTS), "puzzles", flush=True)
rnd = 0
def save():
    with open(W / "improved.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "program"])
        for k in ids:
            if len(S[k]) < base[k]: w.writerow([k, json.dumps(S[k], separators=(",", ":"))])
while time.time() - T0 < BUDGET:
    per = CFG["tper"]
    left = BUDGET - (time.time() - T0)
    est = per * len(ids) / J + 60
    if est > left:
        per = max(3.0, (left - 90) * J / max(1, len(ids)))
        if per < 3.5: break
    random.shuffle(ids)
    shards = [ids[i::J] for i in range(J)]
    use_alt = CFG.get("alt", 1) and rnd % 2 == 1
    def start_of(k):
        if use_alt:
            c = [p for p in ALTS.get(k, []) if len(S[k]) <= len(p) <= len(S[k]) + 1 and p != S[k]]
            if c: return random.choice(c)
        return S[k]
    procs = []
    for i, sh in enumerate(shards):
        (W / f"in_{i}.txt").write_text("".join(encode(k, start_of(k)) for k in sh))
        seed = random.randrange(1, 10**9)
        cmd = f"{W}/elfopt --time {per} --seed {seed} {CFG['extra']} < {W}/in_{i}.txt > {W}/out_{i}.txt 2> {W}/log_{i}.txt"
        procs.append(subprocess.Popen(cmd, shell=True))
    for p in procs: p.wait()
    g = 0
    for i in range(J):
        for pid, prog in parse((W / f"out_{i}.txt").read_text()).items():
            n = replay(pid, prog)
            if n is not None and n < len(S[pid]): g += len(S[pid]) - n; S[pid] = prog
            elif n is not None and n == len(S[pid]) and prog != S[pid]: ALTS.setdefault(pid, []).append(prog)
    save()
    rnd += 1
    print(f"[{time.time()-T0:7.0f}s] round {rnd} alt={use_alt} per={per:.1f}s gain={g} cum={sum(base[k]-len(S[k]) for k in ids)}", flush=True)
for f in W.glob("*.txt"): f.unlink()
save()
print("done cum gain", sum(base[k] - len(S[k]) for k in ids), flush=True)
