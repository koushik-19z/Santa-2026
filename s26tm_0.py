# Santa 2026 multi-engine kernel: cycles public optimizers (Atakan Aldemir, Wang Jakie, goodpjw2008; Apache-2.0)
# over a shard of the frontier; every returned program is replayed by an exact referee; only strictly shorter kept.
import csv, json, os, subprocess, sys, time, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
csv.field_size_limit(10**9)
CFG = json.loads('{"budget": 900.0, "lo": 60, "hi": 200, "shard": 0, "nshards": 100, "tmo": 200.0, "plan": [["anneal", 300, 1], ["incremental", 300, 1], ["compact_rotate", 300, 1], ["quartet", 300, 1], ["closure", 0, 1], ["frontier", 0, 1], ["elf", 5, 2]]}')
T0 = time.time(); BUDGET = CFG["budget"]
DS = sorted(Path("/kaggle/input").rglob("elfopt.cpp"))[0].parent
W = Path("/kaggle/working"); W.mkdir(exist_ok=True)
J = os.cpu_count() or 4
print("ds", DS, "cpus", J, flush=True)
def cc(src, out):
    r = subprocess.run(["g++", "-std=c++17", "-O3", "-march=native", "-o", str(W / out), str(src)], capture_output=True, text=True)
    if r.returncode: print("COMPILE FAIL", out, r.stderr[:3000], flush=True); return False
    return True
ok = {}
ok["elfopt"] = cc(DS / "elfopt.cpp", "elfopt")
for n in ["anneal", "incremental", "compact_rotate", "quartet"]: ok[n] = cc(DS / "wj" / f"{n}.cpp", n)
ok["closure"] = cc(DS / "atakan" / "elf_interface_closure.cpp", "closure")
ok["frontier"] = cc(DS / "atakan" / "elf_multiplier_frontier.cpp", "frontier")
print("compiled", ok, round(time.time() - T0), "s", flush=True)

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
    for ins in prog:
        if len(ins) != 4: return None
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
    return None if miss else len(prog)

S = read_sub(DS / "frontier.csv")
ids = sorted(P, key=lambda k: -len(S[k]))
ids = [k for k in ids if CFG["lo"] <= len(S[k]) <= CFG["hi"]]
ids = ids[CFG["shard"]::CFG["nshards"]]
base = {k: len(S[k]) for k in ids}
print("puzzles", len(ids), "total", sum(base.values()), flush=True)
def save():
    with open(W / "improved.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "program"])
        for k in ids:
            if len(S[k]) < base[k]: w.writerow([k, json.dumps(S[k], separators=(",", ":"))])
def accept(k, prog):
    n = replay(k, prog)
    if n is not None and n < len(S[k]): S[k] = prog; return len(S[k]) and (base[k] - n)
    return 0

# ---------------- engines
def run_wj(name, batch, rounds):
    rows = [str(len(batch))]
    for k in batch:
        s, t, L = P[k]; tt = sorted(set(t))
        rows += [f"{L} 48 {rounds} 0 {random.randrange(1, 2**31)}", " ".join(map(str, (len(s), *s))),
                 " ".join(map(str, (len(tt), *tt))), "1", str(len(S[k]))] + [f"{op} {x} {y} {d}" for op, x, y, d in S[k]]
    r = subprocess.run([str(W / name)], input="\n".join(rows) + "\n", capture_output=True, text=True, timeout=CFG["tmo"])
    out = {}
    for k, line in zip(batch, r.stdout.splitlines()):
        tok = line.split()
        if not tok: continue
        out[k] = [[tok[i], int(tok[i + 1]), int(tok[i + 2]), int(tok[i + 3])] for i in range(1, len(tok), 4)]
    return out
def run_ak(name, batch, args):
    cases = []
    for k in batch:
        s, t, L = P[k]; tt = sorted(set(t)); prog = S[k]
        cases.append("\n".join([f"{k} {L} {len(s)} {len(tt)} {len(prog)}", " ".join(map(str, s)), " ".join(map(str, tt))]
                               + [f"{op} {x} {y} {d}" for op, x, y, d in prog]))
    argv = [str(W / name)] + args.format(salt=random.randrange(1, 2**31)).split()
    r = subprocess.run(argv, input="\n".join(cases) + "\n", capture_output=True, text=True, timeout=CFG["tmo"])
    lines = r.stdout.splitlines(); out, pos = {}, 0
    while pos < len(lines):
        head = lines[pos].split(); pos += 1
        if len(head) < 2: continue
        k, n = head[0], int(head[1])
        out[k] = [[x[0], int(x[1]), int(x[2]), int(x[3])] for x in (l.split() for l in lines[pos:pos + n])]; pos += n
    return out
def run_elf(batch, per, extra):
    fin = W / f"e_{random.randrange(10**9)}.txt"
    txt = []
    for k in batch:
        s, t, L = P[k]; prog = S[k]
        h = ["P", k, str(len(s))] + [str(x) for x in s] + [str(len(t))] + [str(x) for x in t] + [str(L), "48", str(len(prog))]
        txt.append("\n".join([" ".join(h)] + ["%s %d %d %d" % tuple(x) for x in prog]) + "\n")
    fin.write_text("".join(txt))
    r = subprocess.run(f"{W}/elfopt --time {per} --seed {random.randrange(1, 10**9)} {extra} < {fin}", shell=True, capture_output=True, text=True, timeout=CFG["tmo"])
    fin.unlink()
    res, lines, k = {}, r.stdout.split("\n"), 0
    while k < len(lines):
        ln = lines[k].split()
        if ln and ln[0] == "R":
            pid, n = ln[1], int(ln[3])
            res[pid] = [[o, int(i), int(j), int(d)] for o, i, j, d in (lines[k + 1 + t].split() for t in range(n))]
            k += n + 1
        else: k += 1
    return res

CL = ["90 26 7 {salt} 4150", "92 32 4 {salt} 5250", "100 28 9 {salt} 4950"]
FR = ["8 4200 9 10 9 {salt} 1500", "14 8000 12 13 9 {salt} 3000", "8 4200 9 10 13 {salt} 1500"]
plan = CFG["plan"]
it = 0
while time.time() - T0 < BUDGET - 300:
    step = plan[it % len(plan)]; it += 1
    name = step[0]
    if not ok.get(name if name != "elf" else "elfopt", False): continue
    order = ids[:]; random.shuffle(order)
    bs = step[2] if len(step) > 2 else 4
    batches = [order[i:i + bs] for i in range(0, len(order), bs)]
    t1 = time.time(); g = 0
    def job(b):
        if time.time() - T0 > BUDGET - 240: return {}
        try:
            if name in ("anneal", "incremental", "compact_rotate", "quartet"): return run_wj(name, b, step[1])
            if name == "closure": return run_ak("closure", b, CL[it % 3])
            if name == "frontier": return run_ak("frontier", b, FR[it % 3])
            if name == "elf": return run_elf(b, step[1], "--kicks 1000 --kslack 2")
        except Exception as e:
            print("ERR", name, repr(e)[:200], flush=True)
        return {}
    with ThreadPoolExecutor(J) as ex:
        for out in ex.map(job, batches):
            for k, prog in out.items():
                if k in base:
                    n = replay(k, prog)
                    if n is not None and n < len(S[k]): g += len(S[k]) - n; S[k] = prog
    save()
    print(f"[{time.time()-T0:7.0f}s] {name} {step[1]} took {time.time()-t1:.0f}s gain={g} cum={sum(base[k]-len(S[k]) for k in ids)}", flush=True)
save()
print("done cum gain", sum(base[k] - len(S[k]) for k in ids), flush=True)
