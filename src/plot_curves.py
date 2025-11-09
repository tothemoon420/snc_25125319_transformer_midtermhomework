# -*- coding: utf-8 -*-
"""
plot_curves.py
- 自动扫描 runs/*/metrics.csv 并绘制曲线
- 仅依赖 matplotlib（避免 pandas/NumPy ABI 问题）
- 支持：选择 x/y 列、移动平均平滑、筛选 run、导出最优指标汇总

示例：
  python plot_curves.py
  python plot_curves.py --y train_loss --smooth 3
  python plot_curves.py --only base,no_xattn
  python plot_curves.py --out runs/curves_loss.png --summary runs/summary.csv
"""

import os
import csv
import argparse
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs", help="runs 根目录")
    p.add_argument("--x", default="epoch", help="横轴列名（默认 epoch）")
    p.add_argument("--y", default="train_loss", help="纵轴列名（默认 train_loss）")
    p.add_argument("--only", default="", help="仅绘制的 run 名，逗号分隔（可留空）")
    p.add_argument("--exclude", default="", help="排除的 run 名，逗号分隔（可留空）")
    p.add_argument("--smooth", type=int, default=1, help="移动平均窗口（>=1，无平滑=1）")
    p.add_argument("--title", default="Training Curves", help="图标题")
    p.add_argument("--width", type=float, default=8.0, help="图宽(inch)")
    p.add_argument("--height", type=float, default=5.0, help="图高(inch)")
    p.add_argument("--dpi", type=int, default=150, help="保存分辨率")
    p.add_argument("--out", default=None, help="输出图片路径（默认 runs/curves_<y>.png）")
    p.add_argument("--summary", default=None, help="导出每个 run 的最优点汇总 CSV（可选）")
    p.add_argument("--ylog", action="store_true", help="y 轴对数坐标")
    return p.parse_args()

def list_metrics(root: str) -> List[Tuple[str, str]]:
    """返回 (run_name, metrics.csv 路径) 列表"""
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        f = os.path.join(root, name, "metrics.csv")
        if os.path.isfile(f):
            out.append((name, f))
    return out

def read_xy(path: str, xcol: str, ycol: str) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if xcol not in row or ycol not in row:
                # 跳过缺列的文件
                continue
            try:
                xs.append(float(row[xcol]))
                ys.append(float(row[ycol]))
            except ValueError:
                # 非数值行跳过
                continue
    return xs, ys

def moving_average(vals: List[float], k: int) -> List[float]:
    if k <= 1 or len(vals) == 0:
        return vals
    out = []
    s = 0.0
    buf = []
    for v in vals:
        buf.append(v)
        s += v
        if len(buf) > k:
            s -= buf.pop(0)
        out.append(s / len(buf))
    return out

def filter_runs(items: List[Tuple[str, str]], only: str, exclude: str) -> List[Tuple[str, str]]:
    only_set = set([x.strip() for x in only.split(",") if x.strip()]) if only else None
    excl_set = set([x.strip() for x in exclude.split(",") if x.strip()]) if exclude else set()
    out = []
    for name, f in items:
        if only_set is not None and name not in only_set:
            continue
        if name in excl_set:
            continue
        out.append((name, f))
    return out

def best_point(xs: List[float], ys: List[float], minimize: bool=True) -> Tuple[float, float, int]:
    """返回 (best_x, best_y, idx)。minimize=True 表示 y 越小越好（如 loss）。"""
    if not xs or not ys:
        return float("nan"), float("nan"), -1
    best_idx = 0
    for i in range(1, len(ys)):
        better = (ys[i] < ys[best_idx]) if minimize else (ys[i] > ys[best_idx])
        if better:
            best_idx = i
    return xs[best_idx], ys[best_idx], best_idx

def main():
    args = parse_args()
    items = list_metrics(args.root)
    if not items:
        print(f"未在 {args.root} 下发现任何 metrics.csv，先训练再来。")
        return

    items = filter_runs(items, args.only, args.exclude)
    if not items:
        print("筛选后没有可绘制的 run。")
        return

    plt.figure(figsize=(args.width, args.height))
    summary_rows: List[Dict[str, str]] = []

    for run_name, f in items:
        xs, ys = read_xy(f, args.x, args.y)
        if not xs or not ys:
            print("skip (空数据或缺列):", f)
            continue
        # 排序（防止 epoch 无序）
        pairs = sorted(zip(xs, ys), key=lambda z: z[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ys_s = moving_average(ys, args.smooth)

        plt.plot(xs, ys_s, label=run_name)

        # 记录最优点（对 loss 类指标取最小）
        bx, by, bidx = best_point(xs, ys, minimize=True)
        summary_rows.append({"run": run_name, f"best_{args.y}": f"{by:.6f}", args.x: int(bx) if bx==int(bx) else bx})

    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.title(args.title if args.title else f"Curves of {args.y}")
    if args.ylog:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    out_img = args.out or os.path.join(args.root, f"curves_{args.y}.png")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=args.dpi)
    print("saved figure:", out_img)

    # 导出汇总（可选）
    if args.summary:
        os.makedirs(os.path.dirname(args.summary), exist_ok=True)
        # 按 run 名排序
        summary_rows.sort(key=lambda r: r["run"])
        with open(args.summary, "w", encoding="utf-8", newline="") as fw:
            w = csv.DictWriter(fw, fieldnames=["run", args.x, f"best_{args.y}"])
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        print("saved summary:", args.summary)

if __name__ == "__main__":
    main()
