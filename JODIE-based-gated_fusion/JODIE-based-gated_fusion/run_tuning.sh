#!/bin/bash
set -euo pipefail

: '
이 스크립트는 모든 실험 조합을 자동으로 실행합니다.
- Scale Method: standard, log, log_minmax (3가지)
- Layers: 0 (activation 없음), 1, 2 (각각 2가지 activation) 
- Activation: layer=0에서는 제외, layer=1,2에서는 relu, leaky_relu (2가지)
- Encoder: false(공유), true(분리) (2가지)
총: 3 * [1(layer0) + 2(layer1) + 2(layer2)] * 2 = 30가지 실험 per dataset
'

datasets=("amazon_video")
scale_methods=("standard" "log" "log_minmax")
layers=(0 1 2)
activations=("relu" "leaky_relu")
encoders=("false" "true")

# Calculate total experiments: layer=0은 activation 1개, layer=1,2는 activation 2개씩
total_experiments=$((${#datasets[@]} * ${#scale_methods[@]} * (1 + ${#activations[@]} + ${#activations[@]}) * ${#encoders[@]}))
current_experiment=0

echo "=========================================="
echo "Total Experiments: $total_experiments"
echo "=========================================="

start_time=$(date +%s)

for dataset in "${datasets[@]}"; do
    for scale_method in "${scale_methods[@]}"; do
        for layer in "${layers[@]}"; do
            
            # layer=0일 때는 activation을 "none"으로 고정
            if [ "$layer" -eq 0 ]; then
                activation_list=("none")
            else
                activation_list=("${activations[@]}")
            fi
            
            for activation in "${activation_list[@]}"; do
                for encoder in "${encoders[@]}"; do
                    current_experiment=$((current_experiment + 1))
                    
                    encoder_type="shared"
                    if [ "$encoder" = "true" ]; then
                        encoder_type="separate"
                    fi
                    
                    echo ""
                    echo "╔════════════════════════════════════════════════════════════╗"
                    echo "║ Experiment [$current_experiment/$total_experiments]"
                    echo "║ Dataset: $dataset | Scale: $scale_method | Layers: $layer"
                    if [ "$layer" -eq 0 ]; then
                        echo "║ Activation: (none, scalar) | Encoder: $encoder_type"
                    else
                        echo "║ Activation: $activation | Encoder: $encoder_type"
                    fi
                    echo "╚════════════════════════════════════════════════════════════╝"
                    echo ""
                    
                    if bash run.sh "$dataset" "$scale_method" "$layer" "$activation" "$encoder"; then
                        echo "✓ Experiment [$current_experiment/$total_experiments] COMPLETED"
                    else
                        echo "✗ Experiment [$current_experiment/$total_experiments] FAILED"
                        exit 1
                    fi
                done
            done
        done
    done
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))

echo ""
echo "=========================================="
echo "All Experiments Completed!"
echo "Total Time: ${hours}h ${minutes}m ${seconds}s"
echo "=========================================="
