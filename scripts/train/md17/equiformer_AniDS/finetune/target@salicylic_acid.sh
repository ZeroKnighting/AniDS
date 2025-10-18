TARGET="salicylic_acid"
LR=5e-4
ENERGY_COEFFICIENT=1
FORCE_COEFFICIENT=80

DENOISING_POS_PROB=${3:-0.25}
DENOISING_POS_STD=0.05
DENOISING_POS_WEIGHT=5.0
DENOISING_CORRUPT_RATIO=${2:-0.25}

OUTPUT_DIR="models/md17/equiformer_AniDS/salicylic_acid"

CONFIG_PATH="md17/configs/equiformer_dens/equiformer_AniDS_N@6_L@2_C@128-64-32.yml"
KL_WEIGHT=${1:-1}

echo "KL_WEIGHT: $KL_WEIGHT"
echo "DENOISING_POS_PROB: $DENOISING_POS_PROB"
echo "DENOISING_CORRUPT_RATIO: $DENOISING_CORRUPT_RATIO"
echo "KL_WEIGHT: $KL_WEIGHT"
python main_PCQM4Mv2_dens.py \
    --output-dir $OUTPUT_DIR \
    --config-yml $CONFIG_PATH \
    --target  $TARGET \
    --data-path 'datasets/equiformer/md17' \
    --epochs 1500 \
    --lr $LR \
    --batch-size 8 \
    --eval-batch-size 32 \
    --weight-decay 1e-6 \
    --energy-weight $ENERGY_COEFFICIENT \
    --force-weight $FORCE_COEFFICIENT \
    --kl_weight $KL_WEIGHT \
    --denoising-pos-prob $DENOISING_POS_PROB \
    --denoising-pos-weight $DENOISING_POS_WEIGHT \
    --denoising-pos-std $DENOISING_POS_STD \
    --denoising-corrupt-ratio $DENOISING_CORRUPT_RATIO \
    --use-denoising-pos-weight-linear-decay \
    --test-interval 20 \
    --test-max-iter 500 \
    --model-ema \
    --model-ema-decay 0.999 \
    --use_vae \
    --freeze_vae \
    --test_type "AniDS" \
    --checkpoint-path "./best.ckpt"
    
#    --checkpoint-path $CHECKPOINT_PATH
#    --evaluate