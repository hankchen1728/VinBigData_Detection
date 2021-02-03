python3 train.py \
    --project vinbigdata \
    -c 6 \
    --num-workers 32 \
    --cuda-devices 4,5,6,7 \
    --batch-size 8 \
    --lr 0.0001 \
    --optim adamw \
    --num-epochs 500
