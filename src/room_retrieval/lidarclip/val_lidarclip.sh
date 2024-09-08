# export VLSG space 
export VLSG_SPACE=/home/yang/big_ssd/Scan3R/VLSG
# export SCAN3R DATA ROOR
export Data_ROOT_DIR=/home/yang/big_ssd/Scan3R/3RScan
# export training out dir
export ROOM_RETRIEVAL_OUT_DIR=/home/yang/big_ssd/Scan3R/3RScan/out_room_retrieval/PaperMetric/lidarclip
# export resume folder 
export RESUME_DIR=/home/yang/big_ssd/Scan3R/3RScan/out_patch_obj_match/week8/lidar_clip_trainval_scan_Npair/snapshots
# export conda env
export CONDA_BIN=/home/yang/anaconda3/bin

# activate conda env
# conda activate VLSG
source $CONDA_BIN/activate lidarclip

# go into VLSG space
cd $VLSG_SPACE
python ./src/room_retrieval/lidarclip/val_lidarclip.py \
--config ./src/room_retrieval/lidarclip/scan3r_lidarclip.yaml