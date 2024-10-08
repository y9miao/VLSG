# export VLSG space 
export VLSG_SPACE=/home/yang/big_ssd/Scan3R/VLSG
# export SCAN3R DATA ROOR
export Data_ROOT_DIR=/home/yang/big_ssd/Scan3R/3RScan
# export training out dir
export ROOM_RETRIEVAL_OUT_DIR=/home/yang/big_ssd/Scan3R/3RScan/out_room_retrieval/PaperMetric/OursStatistics
# export resume folder 
export RESUME_DIR=/home/yang/big_ssd/Scan3R/VLSG/checkpoint
# export conda env
export CONDA_BIN=/home/yang/anaconda3/bin

# activate conda env
source $CONDA_BIN/activate GCVit

# go into VLSG space
cd $VLSG_SPACE
python ./src/room_retrieval/patch_SGMatch/inference_room_SGMatch.py \
--config ./src/room_retrieval/patch_SGMatch/inference_room_PARGI_statistics.yaml