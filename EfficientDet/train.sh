python3 train.py \
    --project vinbigdata \
    -c 2 \
    --num-workers 40 \
    --cuda-devices 4,5,6,7 \
    --batch-size 80 \
    --lr 0.0001 \
    --optim adamw \
    --num-epochs 150 \
    --load_weights /work/VinBigData/exp-records/efficientdet-d2/vinbigdata/efficientdet-d2_30_4650.pth \
    --es-patience 30 \
    --saved-path /work/VinBigData/exp-records/efficientdet-d2-resume
