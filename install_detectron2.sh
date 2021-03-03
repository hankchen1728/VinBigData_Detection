# See: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

# Load cuda
# module load cuda/10.2
# module load cudnn-7.6/10.2

# Install pre-bulti Detectron2
# See: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md#install-pre-built-detectron2-linux-only
# to choose compatible version
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.7/index.html
