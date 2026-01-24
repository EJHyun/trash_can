#!/bin/bash
set -euo pipefail

: '
Usage examples:
$ ./run_fixed_timediff.sh wikipedia fusion 5 100 1.0
$ ./run_fixed_timediff.sh douban_movie fusion 10 200 0.5
$ ./run_fixed_timediff.sh amazon_video fusion 3 50 2.0
'

dataset="${1:?dataset required}"
window_type="${2:?window_type required (e.g., fusion)}"
top_k="${3:?top_k required (integer)}"
bottom_k="${4:?bottom_k required (integer)}"
cl_weight="${5:?cl_weight required (float)}"

# 고정된 timediff 파라미터
timediff_scale_method="log"
timediff_embed_layers=1
timediff_activation="leaky_relu"
timediff_separate_encoder="false"

# 고정 하이퍼파라미터
model="JODIE_ORCA"
window_size=0.05
threshold=0.8
min_deg=2
tau=0.2
timediff_embed_dim=16

# 모델 이름에 실험 설정 포함
encoder_type="shared_enc"
model_name="${model}_${window_type}_${timediff_scale_method}_l${timediff_embed_layers}_${timediff_activation}_${encoder_type}"

# GPU 설정
GPU=0

# 인자 구성
COMMON_ARGS=(--model "$model_name" --gpu="$GPU" --timediff_embed_dim "$timediff_embed_dim" --timediff_embed_layers "$timediff_embed_layers" --timediff_activation "$timediff_activation" --timediff_scale_method "$timediff_scale_method")

PRECOMPUTE_ARGS=(
  --window_type "$window_type" \
  --window_size "$window_size" \
  --top_k "$top_k" \
  --threshold "$threshold" \
  --bottom_k "$bottom_k" \
  --min_deg "$min_deg" \
  --timediff_scale_method "$timediff_scale_method"
)

CL_ARGS=(--cl_weight "$cl_weight" --tau "$tau")

TAG="wt${window_type}_ws${window_size}_tk${top_k}_th${threshold}_bk${bottom_k}_md${min_deg}_clw${cl_weight}_clt${tau}"
LOG_DIR="./log/${dataset}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${model_name}__${TAG}.log"

echo "=========================================="
echo "Dataset: $dataset"
echo "Window Type: $window_type"
echo "Top K: $top_k"
echo "Bottom K: $bottom_k"
echo "CL Weight: $cl_weight"
echo "Model: $model_name"
echo "=========================================="

python precompute_sim_neighbors.py \
  --network "$dataset" \
  "${PRECOMPUTE_ARGS[@]}"
  
python -u train.py \
  --epochs 1000 \
  --network "$dataset" \
  "${COMMON_ARGS[@]}" \
  "${PRECOMPUTE_ARGS[@]}" \
  "${CL_ARGS[@]}" \
  >> "$LOG_FILE"

python test.py \
  --network "$dataset" \
  "${COMMON_ARGS[@]}" \
  "${PRECOMPUTE_ARGS[@]}" \
  "${CL_ARGS[@]}" \
  >> "$LOG_FILE"

python alert.py --script "${dataset}_${model_name}__${TAG}"

echo "Experiment completed: $model_name"
