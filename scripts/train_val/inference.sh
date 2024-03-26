# export VLSG space 
export VLSG_SPACE=${dir of the VLSG repo}
# export SCAN3R DATA ROOR
export Scan3R_ROOT_DIR=${dir of the dataset (path to "3RScan")}
# export training out dir
export ROOM_RETRIEVAL_OUT_DIR=${dir of the inference result}
# export resume folder 
export RESUME_DIR=${dir of output training results}

# go into VLSG space
cd $VLSG_SPACE
python ./src/room_retrieval/patch_SGIE_aligner/inference_room.py \
--config ./scripts/train_val/val.yaml