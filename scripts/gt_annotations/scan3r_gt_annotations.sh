export VLSG_SPACE=/home/yang/toolbox/ECCV2024/CodePlace/OfficialCode/VLSG
export Data_ROOT_DIR=/home/yang/big_ssd/Scan3R/3RScan
export CONDA_BIN=/home/yang/anaconda3/bin

# activate conda env
# conda activate VLSG
source $CONDA_BIN/activate VLSG

cd $VLSG_SPACE
# project 3D object annotations to 2D query images
python ./preprocessing/gt_anno_2D/scan3r_obj_projector.py
# aggretate pixel-wise annotations to patch-wise annotations
python ./preprocessing/gt_anno_2D/scan3r_obj_img_associate.py \
    --config ./preprocessing/gt_anno_2D/gt_anno.yaml

