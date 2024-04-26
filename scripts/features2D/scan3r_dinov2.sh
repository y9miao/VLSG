export VLSG_SPACE=/home/yang/toolbox/ECCV2024/CodePlace/OfficialCode/VLSG
export Data_ROOT_DIR=/home/yang/big_ssd/Scan3R/3RScan
export CONDA_BIN=/home/yang/anaconda3/bin

# activate conda env
# conda activate VLSG
source $CONDA_BIN/activate VLSG

cd $VLSG_SPACE

# generate patch-level features with Dinov2
python ./preprocessing/img_features/DinoV2/scan3r_dinov2_generator.py \
    --config ./preprocessing/img_features/DinoV2/scan3r_dinov2_generator.yaml

