# export VLSG space 
export VLSG_SPACE=/home/yang/big_ssd/Scan3R/VLSG
# export SCAN3R DATA ROOR
export Data_ROOT_DIR=/home/yang/big_ssd/Scan3R/3RScan
# export training out dir
export ROOM_RETRIEVAL_OUT_DIR=/home/yang/big_ssd/Scan3R/3RScan/out_room_retrieval/PaperMetric/liploc
# export resume folder 
export RESUME_DIR=
# export conda env
export CONDA_BIN=/home/yang/anaconda3/bin

# activate conda env
# conda activate VLSG
source $CONDA_BIN/activate lidarclip

# go into VLSG space
cd $VLSG_SPACE
python ./src/room_retrieval/liploc/val_liploc.py \
--config ./src/room_retrieval/liploc/scan3r_liploc.yaml