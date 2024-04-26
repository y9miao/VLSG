# export VLSG space 
export VLSG_SPACE=/home/yang/toolbox/ECCV2024/CodePlace/OfficialCode/VLSG
# export SCAN3R DATA ROOR
export Scan3R_ROOT_DIR=/home/yang/big_ssd/Scan3R/3RScan
# export retrieval out dir
export VLSG_TRAINING_OUT_DIR=home/yang/toolbox/ECCV2024/CodePlace/Results/train
# export resume folder 
export RESUME_DIR=$VLSG_TRAINING_OUT_DIR

# activate conda env
export CONDA_BIN=/home/yang/anaconda3/bin
source $CONDA_BIN/activate VLSG

# go into VLSG space
cd $VLSG_SPACE
python ./src/trainval/trainval_patchobj_TXAE_SGIAE.py \
--config ./scripts/train_val/train.yaml
