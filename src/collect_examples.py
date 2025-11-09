# collect_examples.py
import os, argparse, random

def read_lines(p): 
    return [x.rstrip("\n") for x in open(p, encoding="utf-8").read().splitlines()]

def find_pred(run):
    d = os.path.join("runs", run)
    if not os.path.isdir(d): return None
    for fn in sorted(os.listdir(d)):
        if fn.startswith("pred.beam") and fn.endswith(".fr"):
            return os.path.join(d, fn)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_en", default="dev200.en")
    ap.add_argument("--dev_fr", default="dev200.fr")
    ap.add_argument("--runs", default="base_eval200_b3,no_xattn_eval200_b3,no_posenc_eval200_b3,h1_eval200_b3")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    en = read_lines(args.dev_en); fr = read_lines(args.dev_fr)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    preds = {}
    for r in runs:
        pf = find_pred(r)
        if pf:
            preds[r] = read_lines(pf)
        else:
            print("skip (no pred):", r)

    n = min(len(en), *(len(v) for v in preds.values())) if preds else 0
    idx = sorted(random.sample(range(n), min(args.k, n)))
    os.makedirs("runs", exist_ok=True)
    out = os.path.join("runs", "examples_subset.txt")
    with open(out, "w", encoding="utf-8") as f:
        for i in idx:
            f.write(f"=== Example {i} ===\n")
            f.write(f"EN : {en[i]}\nREF: {fr[i]}\n")
            for r in runs:
                if r in preds:
                    f.write(f"PRED[{r}]: {preds[r][i]}\n")
            f.write("\n")
    print("saved:", out)

if __name__ == "__main__":
    main()
