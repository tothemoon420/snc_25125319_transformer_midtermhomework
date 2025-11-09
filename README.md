# Transformer EN→FR: From-Scratch + Ablations

本仓库复现并扩展你已完成的英→法 Transformer 小规模实验，提供**基线训练、消融、BLEU 评测、曲线绘图、样例导出**的一键命令，确保可复现。

## 1. 环境与依赖

### 1.1 建议使用 Conda + CUDA（Windows / Linux）
```bash
conda create -n tfmr python=3.10 -y
conda activate tfmr

# 用 conda 装 PyTorch（示例为 CUDA 12.1；若无独显可改成 cpuonly）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 其余依赖用 pip
pip install -r requirements.txt
```
> 如使用 CPU，可把 `pytorch-cuda=12.1` 去掉；或在官网选择合适的安装指令。

### 1.2 目录结构
```
project_root/
├─ src/
│  └─ README.md
│  └─train_quick.py          # 训练 + 快速翻译（你现有的脚本）
│  └─evaluate_bleu.py        # BLEU 评测（beam/alpha/no-repeat）
│  └─plot_curves.py          # 训练曲线绘图（读 runs/*/metrics.csv）
│  └─summarize_bleu.py       # 汇总 BLEU 并画柱状图
│  └─collect_examples.py     # 导出 EN/REF/PRED 样例
├─ splits/                   # train.csv / dev.en / dev.fr
├─ files/                    # 数据集/中间文件（如 en2fr.csv 等）        
├─ results/
│  └─runs/                   # 每次运行的日志/曲线/预测/分数
├─ scripts/
│  └─ run.sh                 # 运行脚本
├─ README.md
├─ _pycache_
└─ requirements.txt
```

## 2. 数据准备
确保已有：
- `splits/train.csv`（两列：`en,fr`），
- `splits/dev.en` 与 `splits/dev.fr`（行对齐）。

如需从 `files/en2fr.csv` 生成，可用你先前的 pandas 脚本（见报告“环境与数据切分”小节）。

## 3. 训练（基线与消融）
> 固定随机种子以保证可复现：`--seed 42`  
> 将权重存放到 `runs/<run_name>/model.pt`，避免覆盖。

**基线：**
```bash
python train_quick.py   --epochs 20 --batch_size 64   --layers 2 --d_model 128 --d_ff 512 --heads 8   --dropout 0.1 --smoothing 0.1   --seed 42   --run_name base   --model_path runs/base/model.pt
```

**典型消融（逐个跑）：**
```bash
# 1) 无交叉注意力（解码器只做因果自注意力）
python train_quick.py --epochs 20 --batch_size 64   --layers 2 --d_model 128 --d_ff 512 --heads 8   --dropout 0.1 --smoothing 0.1 --seed 42   --no_xattn   --run_name no_xattn   --model_path runs/no_xattn/model.pt

# 2) 去位置编码
python train_quick.py --epochs 20 --batch_size 64   --layers 2 --d_model 128 --d_ff 512 --heads 8   --dropout 0.1 --smoothing 0.1 --seed 42   --no_posenc   --run_name no_posenc   --model_path runs/no_posenc/model.pt

# 3) 头数=1
python train_quick.py --epochs 20 --batch_size 64   --layers 2 --d_model 128 --d_ff 512 --heads 1   --dropout 0.1 --smoothing 0.1 --seed 42   --run_name h1   --model_path runs/h1/model.pt

# 4) Dropout 关闭 / 提高
python train_quick.py --epochs 20 --batch_size 64   --layers 2 --d_model 128 --d_ff 512 --heads 8   --dropout 0.0 --smoothing 0.1 --seed 42   --run_name do0   --model_path runs/do0/model.pt

python train_quick.py --epochs 20 --batch_size 64   --layers 2 --d_model 128 --d_ff 512 --heads 8   --dropout 0.3 --smoothing 0.1 --seed 42   --run_name do30   --model_path runs/do30/model.pt

# 5) Label Smoothing 关闭
python train_quick.py --epochs 20 --batch_size 64   --layers 2 --d_model 128 --d_ff 512 --heads 8   --dropout 0.1 --smoothing 0.0 --seed 42   --run_name no_smooth   --model_path runs/no_smooth/model.pt
```

## 4. 评测与图表

### 4.1 全量 dev 的 BLEU（对比 beam=1 / beam=3）
```bash
# 基线 beam=1
python evaluate_bleu.py   --model_path runs/base/model.pt   --layers 2 --d_model 128 --d_ff 512 --heads 8   --dev_en splits/dev.en --dev_fr splits/dev.fr   --beam 1   --run_name base_full_b1

# 基线 beam=3 + 长度惩罚 + ngram 去重
python evaluate_bleu.py   --model_path runs/base/model.pt   --layers 2 --d_model 128 --d_ff 512 --heads 8   --dev_en splits/dev.en --dev_fr splits/dev.fr   --beam 3 --alpha 0.6 --no_repeat_ngram 3   --run_name base_full_b3
```

**消融（按需替换 `--model_path` 与 `--run_name`）：**
```bash
python evaluate_bleu.py   --model_path runs/no_xattn/model.pt   --layers 2 --d_model 128 --d_ff 512 --heads 8   --dev_en splits/dev.en --dev_fr splits/dev.fr   --beam 3 --alpha 0.6 --no_repeat_ngram 3   --run_name no_xattn_full_b3
# 其余：no_posenc_full_b3, h1_full_b3, do0_full_b3, do30_full_b3, no_smooth_full_b3
```

### 4.2 汇总图表
```bash
# 训练曲线（读 runs/*/metrics.csv）
python plot_curves.py

# BLEU 柱状图（仅 beam=3 结果）
python summarize_bleu.py --beam_values 3 --out runs/bleu_bars_full_b3.png

# 样例导出（可选）
python collect_examples.py --run base_full_b3 --k 50 --out results/examples_base.tsv
```

## 5. 日志与可复现性产物
每个 `runs/<run_name>/` 至少包含：
- `params.json`：完整参数（含随机种子）；
- `metrics.csv`：`epoch,train_loss,tokens`；
- `model.pt`：权重；
- `bleu.json`（评测后产生）：BLEU 分数、设置（beam/alpha 等）；
- 可选：`translations.txt`、曲线图。

## 6. 硬件说明与预期耗时
- **硬件**：建议 NVIDIA RTX 3060（12GB）或同级；CPU 亦可但速度显著慢。
- **时长**（经验值，因数据量与实现差异会波动）：
  - 训练（20 epoch，小模型）：几十分钟量级；
  - 全量 dev 评测：beam=1 通常几分钟；beam=3 需更久（约数分钟到十几分钟）。
