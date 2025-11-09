# -*- coding: utf-8 -*-
"""
train_quick_ablate.py
- 训练 Encoder-Decoder Transformer（小模型）
- 支持：--no_crossattn / --no_posenc / --heads / --dropout / --smoothing / --run_name
- 保存：runs/<run_name>/{params.json, metrics.csv, model.pt}
- 推理：--translate "text" 或 --predict_dev 生成 runs/<run_name>/pred.beam{B}.fr
注意：
- 数据：files/en2fr.csv（两列：en,fr），dev：splits/dev.en + splits/dev.fr
- 仅使用标准库 csv 读取，避免 pandas 依赖问题
"""
import os, sys, math, json, csv, time, argparse, random
from collections import Counter, defaultdict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------
# 常量
# -------------------
PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# 简单分词/词表
# -------------------
def tokenize(s: str) -> List[str]:
    # 简单空格分词；可按需改进
    return s.strip().split()

def read_csv_pairs(path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        assert "en" in reader.fieldnames and "fr" in reader.fieldnames, \
            f"{path} 需要包含 en, fr 两列表头"
        for row in reader:
            en = row["en"].strip()
            fr = row["fr"].strip()
            if en and fr:
                pairs.append((en, fr))
    return pairs

def build_vocab_from_pairs(pairs: List[Tuple[str, str]], min_freq=1):
    cnt_en, cnt_fr = Counter(), Counter()
    for en, fr in pairs:
        cnt_en.update(tokenize(en))
        cnt_fr.update(tokenize(fr))
    def build(cnt: Counter):
        itos = list(SPECIAL_TOKENS)
        for tok, c in cnt.items():
            if c >= min_freq:
                itos.append(tok)
        stoi = {w:i for i, w in enumerate(itos)}
        return stoi, itos
    en_w2i, en_i2w = build(cnt_en)
    fr_w2i, fr_i2w = build(cnt_fr)
    return en_w2i, en_i2w, fr_w2i, fr_i2w

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

# -------------------
# 批处理/掩码
# -------------------
def pad_batch(seqs: List[List[int]], pad=PAD):
    T = max(len(s) for s in seqs)
    out = []
    for s in seqs:
        out.append(s + [pad]*(T-len(s)))
    return torch.tensor(out, dtype=torch.long)

def make_pad_mask(ids: torch.Tensor, pad_id: int):
    # ids: [B, T] -> True 表示可见
    return (ids != pad_id)

def make_causal_mask(T: int, device):
    # [T,T] 下三角 True
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

def expand_mask(q_len, k_len, base: torch.Tensor):
    # base: [B, k_len] -> [B, q_len, k_len]
    return base.unsqueeze(1).expand(-1, q_len, -1)

def build_masks(src_ids, tgt_in_ids, pad_id=PAD):
    # src self-attn
    src_pad = make_pad_mask(src_ids, pad_id)                     # [B, Ts]
    src_mask = expand_mask(src_ids.size(1), src_ids.size(1), src_pad)  # [B,Ts,Ts]
    # tgt self-attn (causal + pad)
    tgt_pad = make_pad_mask(tgt_in_ids, pad_id)                  # [B, Tt]
    tgt_pm = expand_mask(tgt_in_ids.size(1), tgt_in_ids.size(1), tgt_pad).to(tgt_in_ids.device)  # [B,Tt,Tt]
    causal = make_causal_mask(tgt_in_ids.size(1), tgt_in_ids.device).unsqueeze(0)               # [1,Tt,Tt]
    tgt_mask = tgt_pm & causal                                                                        # [B,Tt,Tt]
    # mem mask (decoder -> encoder)
    mem_mask = expand_mask(tgt_in_ids.size(1), src_ids.size(1), src_pad)                             # [B,Tt,Ts]
    return src_mask, tgt_mask, mem_mask

# -------------------
# 模型组件
# -------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T,D]
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [T,1]
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        # x: [B,T,D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

def scaled_dot_attn(Q, K, V, mask=None, dropout=None):
    # Q,K,V: [B,h,T,d_k]
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # [B,h,Tq,Tk]
    if mask is not None:
        # mask: [B,1 or h, Tq, Tk] -> True=可见
        scores = scores.masked_fill(~mask, -1e9)
    attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    out = attn @ V
    return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads
        self.d_k = d_model // heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        B, T, D = x.size()
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)  # [B,h,T,d_k]

    def _merge(self, x):
        B, h, T, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, h*d_k)

    def forward(self, q, k, v, mask=None):
        Q = self._split(self.w_q(q))
        K = self._split(self.w_k(k))
        V = self._split(self.w_v(v))
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B,1,Tq,Tk]
        out, attn = scaled_dot_attn(Q, K, V, mask=mask, dropout=self.drop)
        out = self.w_o(self._merge(out))
        return out, attn

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float=0.1, act="gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU() if act.lower() == "gelu" else nn.ReLU()

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))

class PreNormResidual(nn.Module):
    def __init__(self, d_model: int, sublayer: nn.Module, dropout: float=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer
        self.drop = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        return x + self.drop(self.sublayer(self.norm(x), *args, **kwargs))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.s1 = PreNormResidual(d_model,
                                  lambda x, mask=None: self.self_attn(x, x, x, mask)[0],
                                  dropout)
        self.s2 = PreNormResidual(d_model, self.ffn, dropout)

    def forward(self, x, src_mask=None):
        x = self.s1(x, mask=src_mask)
        x = self.s2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1, enable_cross=True):
        super().__init__()
        self.enable_cross = enable_cross
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        if enable_cross:
            self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)

        self.s1 = PreNormResidual(d_model,
                                  lambda y, mask=None: self.self_attn(y, y, y, mask)[0],
                                  dropout)
        if enable_cross:
            self.s2 = PreNormResidual(d_model,
                                      lambda y, mem, mask=None: self.cross_attn(y, mem, mem, mask)[0],
                                      dropout)
        self.s3 = PreNormResidual(d_model, self.ffn, dropout)

    def forward(self, y, memory, tgt_mask=None, mem_mask=None):
        y = self.s1(y, mask=tgt_mask)
        if self.enable_cross:
            y = self.s2(y, memory, mask=mem_mask)
        y = self.s3(y)
        return y

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, d_ff=512, layers=2,
                 heads=8, dropout=0.1, use_posenc=True, enable_cross=True):
        super().__init__()
        self.d_model = d_model
        self.use_posenc = use_posenc
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        self.posenc = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, heads, d_ff, dropout, enable_cross=enable_cross)
            for _ in range(layers)
        ])
        self.generator = nn.Linear(d_model, tgt_vocab)

    def encode(self, src_ids, src_mask):
        x = self.src_embed(src_ids) * math.sqrt(self.d_model)
        if self.use_posenc:
            x = self.posenc(x)
        for layer in self.encoder:
            x = layer(x, src_mask=src_mask)
        return x

    def decode(self, tgt_in_ids, memory, tgt_mask, mem_mask):
        y = self.tgt_embed(tgt_in_ids) * math.sqrt(self.d_model)
        if self.use_posenc:
            y = self.posenc(y)
        for layer in self.decoder:
            y = layer(y, memory, tgt_mask=tgt_mask, mem_mask=mem_mask)
        return y

    def forward(self, src_ids, tgt_in_ids, src_mask, tgt_mask, mem_mask):
        mem = self.encode(src_ids, src_mask)
        y = self.decode(tgt_in_ids, mem, tgt_mask, mem_mask)
        logits = self.generator(y)
        return logits

# -------------------
# 优化/调度/损失
# -------------------
class NoamLR:
    def __init__(self, optimizer, d_model, warmup=2000):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = d_model ** -0.5

    def step(self):
        self._step += 1
        lr = self.factor * min(self._step ** -0.5, self._step * (self.warmup ** -1.5))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.optimizer.step()

def make_criterion(vocab_size: int, smoothing: float):
    # 优先使用原生 label_smoothing；旧版 PyTorch 回退到 KLDiv 实现
    try:
        return nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=float(smoothing))
    except TypeError:
        # 旧版本 fallback
        class SmoothedCE(nn.Module):
            def __init__(self, eps):
                super().__init__()
                self.eps = float(eps)
                self.v = vocab_size

            def forward(self, logits, target):
                # logits: [B*T, V], target: [B*T]
                logp = F.log_softmax(logits, dim=-1)
                nll = F.nll_loss(logp, target, reduction='none', ignore_index=PAD)
                with torch.no_grad():
                    smooth = torch.full_like(logp, 1.0 / (self.v))
                    # 把 PAD 的行置零，使其不影响 loss
                    mask = (target == PAD).unsqueeze(-1)
                    smooth = smooth.masked_fill(mask, 0.0)
                # 交叉熵 = (1-ε)*NLL + ε*均匀分布KL
                lsm = -(logp * smooth).sum(dim=-1)
                # 对 PAD 位置置零
                lsm = lsm.masked_fill(target == PAD, 0.0)
                nll = nll.masked_fill(target == PAD, 0.0)
                return ((1 - self.eps) * nll + self.eps * lsm).mean()
        return SmoothedCE(smoothing)

# -------------------
# 数据加载为批
# -------------------
def batch_iter(pairs_tok: List[Tuple[List[str], List[str]]], w2i_en, w2i_fr,
               batch_size=64, shuffle=True):
    idxs = list(range(len(pairs_tok)))
    if shuffle:
        random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        sub = idxs[i:i+batch_size]
        src_seqs, tgt_seqs = [], []
        for j in sub:
            en_tok, fr_tok = pairs_tok[j]
            src_seqs.append(ids_from_tokens(en_tok, w2i_en, add_bos_eos=True))
            tgt_seqs.append(ids_from_tokens(fr_tok, w2i_fr, add_bos_eos=True))
        yield src_seqs, tgt_seqs

# -------------------
# 解码（贪心 / beam）
# -------------------
def violates_no_repeat(seq: List[int], next_id: int, n: int) -> bool:
    if n <= 0: return False
    test = seq + [next_id]
    if len(test) < n: return False
    last = tuple(test[-n:])
    # 在 test[:-n] 中查重
    for i in range(len(test) - n):
        if tuple(test[i:i+n]) == last:
            return True
    return False

@torch.no_grad()
def greedy_decode(model, src_ids, src_mask, max_len=64):
    device = src_ids.device
    mem = model.encode(src_ids, src_mask)
    B = src_ids.size(0)
    ys = torch.full((B,1), BOS, dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_mask = build_masks(src_ids, ys)[1].to(device)
        mem_mask = build_masks(src_ids, ys)[2].to(device)
        out = model.decode(ys, mem, tgt_mask, mem_mask)
        logits = model.generator(out[:, -1:, :])  # [B,1,V]
        next_id = logits.squeeze(1).argmax(-1)    # [B]
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
                tgt_mask = build_masks(src_ids[b:b+1], ys)[1]
                mem_mask = build_masks(src_ids[b:b+1], ys)[2]
                out = model.decode(ys, memory, tgt_mask, mem_mask)
                logits = model.generator(out[:, -1, :])  # [1,V]
                logp = F.log_softmax(logits, dim=-1).squeeze(0)  # [V]
                topk_lp, topk_id = torch.topk(logp, k=beam, dim=-1)
                for k in range(beam):
                    nid = int(topk_id[k])
                    if no_repeat_ngram and violates_no_repeat(seq, nid, no_repeat_ngram):
                        continue
                    new_seq = seq + [nid]
                    new_lp = lp + float(topk_lp[k])
                    new_beams.append((new_seq, new_lp, nid == EOS))
            # 归一化 + 选择 top beam
            def score(item):
                seq, lp, ended = item
                L = max(1, len(seq)-1)  # 不含BOS
                lp_norm = lp / (((5+L)/6) ** alpha)
                return lp_norm
            beams = sorted(new_beams, key=score, reverse=True)[:beam]
            if all(e for _,_,e in beams):
                break
        best = max(beams, key=score)[0]
        results.append(best)
    return results  # List[List[int]]

# -------------------
# 训练/验证 主过程
# -------------------
def train_loop(args):
    device = get_device()
    print("Device:", device)

    # 读训练对（csv）
    train_pairs = read_csv_pairs(args.train_csv)
    if len(train_pairs) == 0:
        print(f"空训练集：{args.train_csv}")
        sys.exit(1)
    # 构建词表（如存在缓存则可自行扩展加载）
    en_w2i, en_i2w, fr_w2i, fr_i2w = build_vocab_from_pairs(train_pairs, min_freq=1)

    # 持久化词表（可供评测脚本使用）
    os.makedirs("files", exist_ok=True)
    with open(os.path.join("files","dict.p"), "w", encoding="utf-8") as f:
        json.dump({
            "src_itos": en_i2w, "tgt_itos": fr_i2w
        }, f, ensure_ascii=False, indent=2)

    # 预分词的训练集（减少重复分词）
    train_tok = [(tokenize(en), tokenize(fr)) for en, fr in train_pairs]

    # 模型
    enable_cross = (not args.no_crossattn)
    use_posenc = (not args.no_posenc)
    model = Transformer(
        src_vocab=len(en_i2w), tgt_vocab=len(fr_i2w),
        d_model=args.d_model, d_ff=args.d_ff, layers=args.layers,
        heads=args.heads, dropout=args.dropout,
        use_posenc=use_posenc, enable_cross=enable_cross
    ).to(device)

    # 目标：teacher forcing
    V = len(fr_i2w)
    criterion = make_criterion(V, args.smoothing)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, d_model=args.d_model, warmup=2000)

    # 运行目录
    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,tokens\n")

    # 训练
    model.train()
    global_step = 0
    for ep in range(args.epochs):
        tloss, tokens = 0.0, 0
        for src_seqs, tgt_seqs in batch_iter(train_tok, en_w2i, fr_w2i,
                                             batch_size=args.batch_size, shuffle=True):
            src_ids = pad_batch(src_seqs, PAD).to(device)              # [B,Ts]
            tgt_ids = pad_batch(tgt_seqs, PAD).to(device)              # [B,Tt]
            # teacher forcing：输入=去掉最后一项；标签=去掉第一项
            tgt_in = tgt_ids[:, :-1]
            tgt_out = tgt_ids[:, 1:]

            src_mask, tgt_mask, mem_mask = build_masks(src_ids, tgt_in, PAD)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            mem_mask = mem_mask.to(device)

            logits = model(src_ids, tgt_in, src_mask, tgt_mask, mem_mask)   # [B,Tt-1,V]
            B, Tm, Vv = logits.size()
            loss = criterion(logits.reshape(B*Tm, Vv), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()
            global_step += 1

            tloss += float(loss.item()) * tgt_out.numel()
            tokens += int(tgt_out.numel())

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(f"{ep+1},{tloss/max(tokens,1):.8f},{tokens}\n")
        print(f"[epoch {ep+1}] train_loss={tloss/max(tokens,1):.6f} tokens={tokens}")

    # 保存权重
    out_model = args.model_path or os.path.join(run_dir, "model.pt")
    torch.save(model.state_dict(), out_model)
    print("saved:", out_model)

    # 训练后可选：dev 预测
    if args.predict_dev:
        predict_dev(args, model, en_w2i, fr_i2w)

@torch.no_grad()
def predict_dev(args, model, en_w2i, fr_i2w):
    device = next(model.parameters()).device
    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    src_lines = []
    dev_en = args.dev_src
    if not os.path.isfile(dev_en):
        print(f"未找到 dev 源文件：{dev_en}（跳过预测）")
        return
    with open(dev_en, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                src_lines.append(s)
    batch = 32
    out_path = os.path.join(run_dir, f"pred.beam{args.beam}.fr")
    fw = open(out_path, "w", encoding="utf-8")
    for i in range(0, len(src_lines), batch):
        part = src_lines[i:i+batch]
        src_tok = [tokenize(x) for x in part]
        src_ids = pad_batch([ids_from_tokens(t, en_w2i, add_bos_eos=True) for t in src_tok]).to(device)
        src_mask = expand_mask(src_ids.size(1), src_ids.size(1), make_pad_mask(src_ids, PAD)).to(device)
        if args.beam == 1:
            ys = greedy_decode(model, src_ids, src_mask, max_len=args.max_len)
            for j in range(ys.size(0)):
                seq = ys[j].tolist()
                toks = tokens_from_ids(seq, fr_i2w)
                fw.write(" ".join(toks) + "\n")
        else:
            seqs = beam_search_decode(model, src_ids, src_mask,
                                      beam=args.beam, alpha=args.alpha,
                                      max_len=args.max_len,
                                      no_repeat_ngram=args.no_repeat_ngram)
            for seq in seqs:
                toks = tokens_from_ids(seq, fr_i2w)
                fw.write(" ".join(toks) + "\n")
    fw.close()
    print("dev 预测写入：", out_path)

@torch.no_grad()
def translate_single(args):
    # 仅推理：载入模型并翻译 args.translate
    device = get_device()
    # 词表
    dict_path = os.path.join("files","dict.p")
    if not os.path.isfile(dict_path):
        print("缺少 files/dict.p（请先训练一轮生成词表或手动提供）。")
        sys.exit(1)
    with open(dict_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    en_i2w = d["src_itos"]
    fr_i2w = d["tgt_itos"]
    en_w2i = {w:i for i,w in enumerate(en_i2w)}

    enable_cross = (not args.no_crossattn)
    use_posenc = (not args.no_posenc)
    model = Transformer(
        src_vocab=len(en_i2w), tgt_vocab=len(fr_i2w),
        d_model=args.d_model, d_ff=args.d_ff, layers=args.layers,
        heads=args.heads, dropout=args.dropout,
        use_posenc=use_posenc, enable_cross=enable_cross
    ).to(device)

    if not os.path.isfile(args.model_path):
        print("未找到权重：", args.model_path)
        sys.exit(1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    s = args.translate.strip()
    toks = tokenize(s)
    ids = ids_from_tokens(toks, en_w2i, add_bos_eos=True)
    src_ids = torch.tensor([ids], dtype=torch.long, device=device)
    src_mask = expand_mask(src_ids.size(1), src_ids.size(1), make_pad_mask(src_ids, PAD)).to(device)

    if args.beam == 1:
        ys = greedy_decode(model, src_ids, src_mask, max_len=args.max_len)
        seq = ys[0].tolist()
    else:
        seq = beam_search_decode(model, src_ids, src_mask,
                                 beam=args.beam, alpha=args.alpha,
                                 max_len=args.max_len,
                                 no_repeat_ngram=args.no_repeat_ngram)[0]
    out = " ".join(tokens_from_ids(seq, fr_i2w))
    print("EN:", s)
    print("FR:", out)

# -------------------
# CLI
# -------------------
def parse_args():
    p = argparse.ArgumentParser()
    # 训练/数据
    p.add_argument("--train_csv", default="files/en2fr.csv")
    p.add_argument("--dev_src", default="splits/dev.en")
    p.add_argument("--dev_ref", default="splits/dev.fr")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    # 模型
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--no_crossattn", action="store_true")
    p.add_argument("--no_posenc", action="store_true")
    # 运行/保存
    p.add_argument("--run_name", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--model_path", default=None)
    # 推理
    p.add_argument("--translate", default=None)
    p.add_argument("--predict_dev", action="store_true")
    p.add_argument("--beam", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--no_repeat_ngram", type=int, default=3)
    p.add_argument("--max_len", type=int, default=64)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    if args.translate is not None:
        translate_single(args)
        return
    train_loop(args)

if __name__ == "__main__":
    main()
