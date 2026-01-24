#!/bin/bash
set -euo pipefail

: '
Usage examples:
$ ./run.sh wikipedia standard 1 relu false
$ ./run.sh wikipedia log 2 leaky_relu true
$ ./run.sh amazon_video log_minmax 1 relu false
'

dataset="${1:?dataset required}"
timediff_scale_method="${2:?timediff_scale_method required (standard/log/log_minmax)}"
timediff_embed_layers="${3:?timediff_embed_layers required (0/1/2)}"
timediff_activation="${4:?timediff_activation required (relu/leaky_relu/none)}"
timediff_separate_encoder="${5:?timediff_separate_encoder required (true/false)}"

# layer=0이면 activation은 무시하고 기본값 사용
if [ "$timediff_embed_layers" -eq 0 ]; then
    timediff_activation="relu"  # layer=0일 때는 activation이 사용되지 않으므로 기본값 설정
    model_name_activation="none"  # 모델 이름에는 'none'으로 표시
else
    model_name_activation="$timediff_activation"  # layer>0일 때는 그대로 사용
fi

# 고정 하이퍼파라미터
model="JODIE_ORCA"
window_type="fusion"
window_size=0.05
top_k=5
threshold=0.8
bottom_k=100
min_deg=2
cl_weight=1.0
tau=0.2
timediff_embed_dim=16

# 모델 이름에 실험 설정 포함
if [ "$timediff_separate_encoder" = "true" ]; then
    encoder_type="sep_enc"
else
    encoder_type="shared_enc"
fi

model_name="${model}_${window_type}_${timediff_scale_method}_l${timediff_embed_layers}_${model_name_activation}_${encoder_type}"

# GPU 설정
GPU=0

# 인자 구성
COMMON_ARGS=(--model "$model_name" --gpu="$GPU" --timediff_embed_dim "$timediff_embed_dim" --timediff_embed_layers "$timediff_embed_layers" --timediff_activation "$timediff_activation" --timediff_scale_method "$timediff_scale_method")

if [ "$timediff_separate_encoder" = "true" ]; then
    COMMON_ARGS+=(--timediff_separate_encoder)
fi

PRECOMPUTE_ARGS=(
  --window_type "$window_type" \
  --window_size "$window_size" \
  --top_k "$top_k" \
  --threshold "$threshold" \
  --bottom_k "$bottom_k" \
  --min_deg "$min_deg" \
  --timediff_scale_method "$timediff_scale_method"
)
if [ "$timediff_separate_encoder" = "true" ]; then
  PRECOMPUTE_ARGS+=(--timediff_separate_encoder)
fi
CL_ARGS=(--cl_weight "$cl_weight" --tau "$tau")

TAG="wt${window_type}_ws${window_size}_tk${top_k}_th${threshold}_bk${bottom_k}_md${min_deg}_clw${cl_weight}_clt${tau}"
LOG_DIR="./log/${dataset}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${model_name}__${TAG}.log"

echo "=========================================="
echo "Dataset: $dataset"
echo "Scale Method: $timediff_scale_method"
echo "Layers: $timediff_embed_layers"
echo "Activation: $timediff_activation"
echo "Encoder: $encoder_type"
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
