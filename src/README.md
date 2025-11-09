# src/
占位目录。可将 `train_quick.py / evaluate_bleu.py` 中的模块化代码（如模型、Tokenizer、数据管道、解码器）逐步搬入：
- `src/model.py`：Transformer 结构（Encoder/Decoder/MHA/FFN/PosEnc）。
- `src/data.py`：数据加载/分词/批处理与 mask 生成。
- `src/decoding.py`：贪心/束搜索、长度惩罚与 ngram 去重。
- `src/metrics.py`：BLEU 评测封装（可调用 sacrebleu）。
- `src/utils.py`：seed 固定、日志与文件 IO、计时。
