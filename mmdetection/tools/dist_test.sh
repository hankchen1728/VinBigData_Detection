#!/usr/bin/env bash

CONFIG="../cfg.py"
CHECKPOINT="./x101/epoch_12.pth"
GPUS=2
PORT=${PORT:-29500}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
