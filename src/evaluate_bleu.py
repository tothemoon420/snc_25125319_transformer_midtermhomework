# -*- coding: utf-8 -*-
"""
evaluate_bleu.py
- 加载已训练权重，对 dev 集 (splits/dev.en, splits/dev.fr) 生成预测并计算 SacreBLEU
- 支持 beam=1(贪心) / beam>1，并可设置长度惩罚 alpha 和 no-repeat-ngrams
- 需已存在 files/dict.p（训练时写出的词表：src_itos/tgt_itos）

用法示例（与训练形状一致非常关键！）：
  python evaluate_bleu.py ^
    --model_path runs\base\model.pt ^
    --layers 2 --d_model 128 --d_ff 512 --heads 8 ^
    --beam 5 --alpha 0.6 --no_repeat_ngram 3 ^
    --run_name base_eval
"""

import os
import sys
import json
import csv
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
import sacrebleu

# 与训练脚本约定一致的 special token id
PAD, UNK, BOS, EOS = 0, 1, 2, 3

# ------------------------
# 基础工具
# ------------------------
def tokenize(s: str) -> List[str]:
    return s.strip().split()

def make_pad_mask(ids: torch.Tensor, pad_id: int = PAD):
    # ids: [B, T] -> True 表示可见
    return (ids != pad_id)

def make_causal_mask(T: int, device):
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

def expand_mask(q_len, k_len, base: torch.Tensor):
    # base: [B, k_len] -> [B, q_len, k_len]
    return base.unsqueeze(1).expand(-1, q_len, -1)

def ids_from_tokens(tokens: List[str], w2i: dict, add_bos_eos=True):
    ids = [w2i.get(t, UNK) for t in tokens]
    if add_bos_eos:
        return [BOS] + ids + [EOS]
    return ids

def tokens_from_ids(ids: List[int], i2w: List[str]):
    toks = []
    for i in ids:
        if i in (PAD, BOS, EOS):
            continue
        toks.append(i2w[i] if i < len(i2w) else "<unk>")
    return toks

def violates_no_repeat(seq: List[int], next_id: int, n: int) -> bool:
    if n <= 0:
        return False
    test = seq + [next_id]
    if len(test) < n:
        return False
    last = tuple(test[-n:])
    for i in range(len(test) - n):
        if tuple(test[i:i+n]) == last:
            return True
    return False

@torch.no_grad()
def greedy_decode(model, src_ids, src_mask, max_len=64):
    device = src_ids.device
    mem = model.encode(src_ids, src_mask)
    B = src_ids.size(0)
    ys = torch.full((B, 1), BOS, dtype=torch.long, device=device)
    for _ in range(max_len):
        Tt = ys.size(1)
        tgt_pad = make_pad_mask(ys, PAD)
        tgt_pm = expand_mask(Tt, Tt, tgt_pad).to(device)
        causal = make_causal_mask(Tt, device).unsqueeze(0)
        tgt_mask = tgt_pm & causal              # [B,Tt,Tt]
        mem_mask = expand_mask(Tt, src_ids.size(1), make_pad_mask(src_ids, PAD)).to(device)

        out = model.decode(ys, mem, tgt_mask, mem_mask)   # [B,Tt,D]
        logits = model.generator(out[:, -1:, :])           # [B,1,V]
        next_id = logits.squeeze(1).argmax(-1)             # [B]
        ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)
        if (next_id == EOS).all():
            break
    return ys

@torch.no_grad()
def beam_search_decode(model, src_ids, src_mask, beam=5, alpha=0.6, max_len=64, no_repeat_ngram=3):
    device = src_ids.device
    mem = model.encode(src_ids, src_mask)
    B = src_ids.size(0)
    results = []
    for b in range(B):
        memory = mem[b:b+1]
        beams = [([BOS], 0.0, False)]  # (seq, logprob, ended)
        for _ in range(max_len):
            new_beams = []
            for seq, lp, ended in beams:
                if ended:
                    new_beams.append((seq, lp, True))
                    continue
                ys = torch.tensor([seq], dtype=torch.long, device=device)  # [1,T]
                Tt = ys.size(1)
                tgt_pad = make_pad_mask(ys, PAD)
                tgt_pm = expand_mask(Tt, Tt, tgt_pad).to(device)
                causal = make_causal_mask(Tt, device).unsqueeze(0)
                tgt_mask = tgt_pm & causal
                mem_mask = expand_mask(Tt, src_ids.size(1), make_pad_mask(src_ids[b:b+1], PAD)).to(device)

                out = model.decode(ys, memory, tgt_mask, mem_mask)   # [1,T,D]
                logits = model.generator(out[:, -1, :])               # [1,V]
                logp = F.log_softmax(logits, dim=-1).squeeze(0)       # [V]
                topk_lp, topk_id = torch.topk(logp, k=beam, dim=-1)
                for k in range(beam):
                    nid = int(topk_id[k])
                    if no_repeat_ngram and violates_no_repeat(seq, nid, no_repeat_ngram):
                        continue
                    new_seq = seq + [nid]
                    new_lp = lp + float(topk_lp[k])
                    new_beams.append((new_seq, new_lp, nid == EOS))
            # 长度惩罚归一化 + 选择 top beam
            def score(item):
                seq, lp, ended = item
                L = max(1, len(seq) - 1)  # 不含 BOS
                return lp / (((5 + L) / 6) ** alpha)
            beams = sorted(new_beams, key=score, reverse=True)[:beam]
            if all(e for _, _, e in beams):
                break
        best = max(beams, key=score)[0]
        results.append(best)
    return results

# ------------------------
# CLI
# ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # 必填：模型路径 + 结构参数（与训练完全一致）
    p.add_argument("--model_path", required=True, help="训练得到的权重 .pt")
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--d_ff", type=int, required=True)
    p.add_argument("--heads", type=int, required=True)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_crossattn", action="store_true")
    p.add_argument("--no_posenc", action="store_true")

    # 数据
    p.add_argument("--dev_en", default="splits/dev.en")
    p.add_argument("--dev_fr", default="splits/dev.fr")
    p.add_argument("--batch_size", type=int, default=32)

    # 解码
    p.add_argument("--beam", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--no_repeat_ngram", type=int, default=3)
    p.add_argument("--max_len", type=int, default=64)

    # 输出
    p.add_argument("--run_name", default="eval")
    return p.parse_args()

# ------------------------
# 主流程
# ------------------------
@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 词表
    dict_path = os.path.join("files", "dict.p")
    if not os.path.isfile(dict_path):
        print("缺少 files/dict.p（请先在训练阶段写出词表后再评测）")
        sys.exit(1)
    with open(dict_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    en_i2w = d["src_itos"]
    fr_i2w = d["tgt_itos"]
    en_w2i = {w: i for i, w in enumerate(en_i2w)}

    # 模型：优先尝试从 train_quick 导入，保证与训练时权重命名一致
    Transformer = None
    try:
        import train_quick as tq
        Transformer = tq.Transformer
    except Exception:
        try:
            import train_quick_ablate as tqa
            Transformer = tqa.Transformer
        except Exception:
            Transformer = None

    if Transformer is None:
        print("未找到 Transformer 定义（train_quick.py / train_quick_ablate.py）。")
        sys.exit(1)

    # 实例化（兼容不同构造签名）
    kwargs = dict(
        src_vocab=len(en_i2w), tgt_vocab=len(fr_i2w),
        d_model=args.d_model, d_ff=args.d_ff, layers=args.layers,
        heads=args.heads, dropout=args.dropout,
    )
    try:
        model = Transformer(use_posenc=(not args.no_posenc),
                            enable_cross=(not args.no_crossattn),
                            **kwargs).to(device)
    except TypeError:
        # 你的 Transformer 构造函数如果不支持 use_posenc/enable_cross，则退化为默认
        model = Transformer(**kwargs).to(device)

    # 载入权重
    if not os.path.isfile(args.model_path):
        print("未找到权重：", args.model_path)
        sys.exit(1)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 读 dev
    if not (os.path.isfile(args.dev_en) and os.path.isfile(args.dev_fr)):
        print("未找到 dev 文件：", args.dev_en, args.dev_fr)
        sys.exit(1)
    with open(args.dev_en, "r", encoding="utf-8") as f:
        src_lines = [x.strip() for x in f if x.strip()]
    with open(args.dev_fr, "r", encoding="utf-8") as f:
        ref_lines = [x.strip() for x in f if x.strip()]

    # 输出目录与文件
    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f"pred.beam{args.beam}.fr")

    # 批量解码
    fw = open(out_path, "w", encoding="utf-8")
    bs = max(1, int(args.batch_size))
    for i in range(0, len(src_lines), bs):
        part = src_lines[i:i+bs]
        src_tok = [tokenize(x) for x in part]
        src_ids = [[BOS] + [en_w2i.get(t, UNK) for t in toks] + [EOS] for toks in src_tok]
        T = max(len(x) for x in src_ids)
        src_batch = torch.tensor(
            [x + [PAD]*(T-len(x)) for x in src_ids],
            dtype=torch.long, device=device
        )
        src_mask = expand_mask(T, T, make_pad_mask(src_batch, PAD)).to(device)

        if args.beam == 1:
            ys = greedy_decode(model, src_batch, src_mask, max_len=args.max_len)
            for j in range(ys.size(0)):
                fw.write(" ".join(tokens_from_ids(ys[j].tolist(), fr_i2w)) + "\n")
        else:
            seqs = beam_search_decode(
                model, src_batch, src_mask,
                beam=args.beam, alpha=args.alpha,
                max_len=args.max_len, no_repeat_ngram=args.no_repeat_ngram
            )
            for seq in seqs:
                fw.write(" ".join(tokens_from_ids(seq, fr_i2w)) + "\n")
    fw.close()
    print("预测写入：", out_path)

    # SacreBLEU
    pred = open(out_path, "r", encoding="utf-8").read().strip().splitlines()
    bleu = sacrebleu.corpus_bleu(pred, [ref_lines])
    print(f"SacreBLEU = {bleu.score:.2f}")

    with open(os.path.join(run_dir, "bleu.txt"), "w", encoding="utf-8") as f:
        f.write(f"SacreBLEU = {bleu.score:.4f}\n")

if __name__ == "__main__":
    main()
