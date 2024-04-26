export VLSG_SPACE=/home/yang/toolbox/ECCV2024/CodePlace/OfficialCode/VLSG
export Data_ROOT_DIR=/home/yang/big_ssd/Scan3R/3RScan
export CONDA_BIN=/home/yang/anaconda3/bin

# activate conda env
# conda activate VLSG
source $CONDA_BIN/activate VLSG

cd $VLSG_SPACE
python ./preprocessing/scan3r/preprocess_scan3r.py \
    --config ./preprocessing/scan3r/preprocess_scan3r.yaml \
    --split train
python ./preprocessing/scan3r/preprocess_scan3r.py \
    --config ./preprocessing/scan3r/preprocess_scan3r.yaml \
    --split val
python ./preprocessing/scan3r/preprocess_scan3r.py \
    --config ./preprocessing/scan3r/preprocess_scan3r.yaml \
    --split test