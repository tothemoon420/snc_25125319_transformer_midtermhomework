import os, re, argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs", help="runs 根目录")
    p.add_argument("--beam_values", nargs="*", type=int, default=None,
                   help="仅显示这些 beam（如: 1 3）")
    p.add_argument("--out", default=None, help="保存柱状图路径（默认 runs/bleu_bars.png）")
    p.add_argument("--no_plot", action="store_true", help="只打印表格行，不画图")
    return p.parse_args()

beam_pat_file = re.compile(r"pred\.beam(\d+)\.fr$")
beam_pat_name = re.compile(r"(?:[_-]b|beam)(\d+)$", re.IGNORECASE)
bleu_pat = re.compile(r"SacreBLEU\s*=\s*([0-9.]+)")

def infer_beam(run_dir, run_name):
    # 1) 从 pred.beamX.fr 文件推断
    beams = []
    for fn in os.listdir(run_dir):
        m = beam_pat_file.search(fn)
        if m:
            beams.append(int(m.group(1)))
    if len(beams) == 1:
        return beams[0]
    # 2) 从 run_name 尾部 *_b5 或 *beam5 推断
    m = beam_pat_name.search(run_name)
    if m:
        return int(m.group(1))
    return None

def latex_escape(s: str) -> str:
    return s.replace("_", r"\_")

def main():
    args = parse_args()
    root = args.root
    if not os.path.isdir(root):
        print("未找到目录：", root)
        return

    items = []  # (run_name, beam, bleu)
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        bleu_file = os.path.join(d, "bleu.txt")
        if not os.path.isfile(bleu_file):
            continue
        txt = open(bleu_file, encoding="utf-8").read()
        m = bleu_pat.search(txt)
        if not m:
            continue
        score = float(m.group(1))
        beam = infer_beam(d, name)
        items.append((name, beam, score))

    if args.beam_values:
        items = [x for x in items if (x[1] in args.beam_values)]

    if not items:
        print("未找到可汇总的 BLEU；请确认 runs/*/bleu.txt 是否存在。")
        return

    # 输出 LaTeX 表格行
    print("\n% ---- Paste into LaTeX table ----")
    print("% Columns: Run & Beam & BLEU")
    for name, beam, score in sorted(items, key=lambda z: (z[0], z[1] if z[1] is not None else 0)):
        bstr = str(beam) if beam is not None else "?"
        print(f"{latex_escape(name)} & {bstr} & {score:.2f} \\\\")
    print("% --------------------------------\n")

    if args.no_plot:
        return

    # 尝试画图（如果没装 matplotlib 就跳过）
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib 未安装，跳过画图（可用 pip install matplotlib 安装）")
        return

    labels = [f"{name} (b={beam if beam is not None else '?'})" for name, beam, _ in items]
    vals = [s for _,_,s in items]
    plt.figure(figsize=(max(6, 0.6*len(items)+3), 4.5))
    plt.bar(labels, vals)
    plt.ylabel("SacreBLEU")
    plt.title("BLEU by run")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out = args.out or os.path.join(root, "bleu_bars.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150)
    print("saved:", out)

if __name__ == "__main__":
    main()