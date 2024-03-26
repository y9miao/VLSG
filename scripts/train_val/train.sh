# export VLSG space 
export VLSG_SPACE=${dir of the VLSG repo}
# export SCAN3R DATA ROOR
export Scan3R_ROOT_DIR=${dir of the dataset (path to "3RScan")}
# export retrieval out dir
export VLSG_TRAINING_OUT_DIR=${dir of output training results}
# export resume folder 
export RESUME_DIR=$VLSG_TRAINING_OUT_DIR

# go into VLSG space
cd $VLSG_SPACE
python ./src/trainval/trainval_patchobj_TXAE_SGIAE.py \
--config ./scripts/train_val/train.yaml
