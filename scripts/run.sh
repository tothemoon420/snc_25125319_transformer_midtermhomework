#!/usr/bin/env bash
set -e

# -------- config --------
SEED=42
LAYERS=2
DM=128
DFF=512
HEADS=8
BS=64
DROP=0.1
SMOOTH=0.1

# -------- train: base --------
python train_quick.py \
  --epochs 20 --batch_size ${BS} \
  --layers ${LAYERS} --d_model ${DM} --d_ff ${DFF} --heads ${HEADS} \
  --dropout ${DROP} --smoothing ${SMOOTH} --seed ${SEED} \
  --run_name base \
  --model_path runs/base/model.pt

# -------- eval: base (beam=1,3) --------
python evaluate_bleu.py \
  --model_path runs/base/model.pt \
  --layers ${LAYERS} --d_model ${DM} --d_ff ${DFF} --heads ${HEADS} \
  --dev_en splits/dev.en --dev_fr splits/dev.fr \
  --beam 1 \
  --run_name base_full_b1

python evaluate_bleu.py \
  --model_path runs/base/model.pt \
  --layers ${LAYERS} --d_model ${DM} --d_ff ${DFF} --heads ${HEADS} \
  --dev_en splits/dev.en --dev_fr splits/dev.fr \
  --beam 3 --alpha 0.6 --no_repeat_ngram 3 \
  --run_name base_full_b3

# -------- ablations: train --------
python train_quick.py --epochs 20 --batch_size ${BS} \
  --layers ${LAYERS} --d_model ${DM} --d_ff ${DFF} --heads ${HEADS} \
  --dropout ${DROP} --smoothing ${SMOOTH} --seed ${SEED} \
  --no_xattn --run_name no_xattn --model_path runs/no_xattn/model.pt

python train_quick.py --epochs 20 --batch_size ${BS} \
  --layers ${LAYERS} --d_model ${DM} --d_ff ${DFF} --heads ${HEADS} \
  --dropout ${DROP} --smoothing ${SMOOTH} --seed ${SEED} \
  --no_posenc --run_name no_posenc --model_path runs/no_posenc/model.pt

python train_quick.py --epochs 20 --batch_size ${BS} \
  --layers ${LAYERS} --d_model ${DM} --d_ff ${DFF} --heads 1 \
  --dropout ${DROP} --smoothing ${SMOOTH} --seed ${SEED} \
  --run_name h1 --model_path runs/h1/model.pt

python train_quick.py --epochs 20 --batch_size ${BS} \
  --layers ${LAYERS} --d_model ${DM} --d_ff ${DFF} --heads ${HEADS} \
  --dropout 0.0 --smoothing ${SMOOTH} --seed ${SEED} \
  --run_name do0 --model_path runs/do0/model.pt

python train_quick.py --epochs 20 --batch_size ${BS} \
  --layers ${LAYERS} --d_model ${DM} --d_ff ${DFF} --heads ${HEADS} \
  --dropout 0.3 --smoothing ${SMOOTH} --seed ${SEED} \
  --run_name do30 --model_path runs/do30/model.pt

python train_quick.py --epochs 20 --batch_size ${BS} \
  --layers ${LAYERS} --d_model ${DM} --d_ff ${DFF} --heads ${HEADS} \
  --dropout ${DROP} --smoothing 0.0 --seed ${SEED} \
  --run_name no_smooth --model_path runs/no_smooth/model.pt

# -------- ablations: eval (beam=3) --------
for name in no_xattn no_posenc h1 do0 do30 no_smooth; do
  python evaluate_bleu.py \
    --model_path runs/${name}/model.pt \
    --layers ${LAYERS} --d_model ${DM} --d_ff ${DFF} --heads ${HEADS} \
    --dev_en splits/dev.en --dev_fr splits/dev.fr \
    --beam 3 --alpha 0.6 --no_repeat_ngram 3 \
    --run_name ${name}_full_b3 || true
done

# -------- plots --------
python plot_curves.py
python summarize_bleu.py --beam_values 3 --out runs/bleu_bars_full_b3.png

echo "Done."
