export VLSG_SPACE=/home/yang/toolbox/ECCV2024/CodePlace/OfficialCode/VLSG
export Data_ROOT_DIR=/home/yang/big_ssd/Scan3R/3RScan
export CONDA_BIN=/home/yang/anaconda3/bin

# activate conda env
# conda activate VLSG
source $CONDA_BIN/activate VLSG

cd $VLSG_SPACE

# generate patch-level features with Dinov2
python ./preprocessing/sg_features/obj_visual_embeddings/Dinov2/gen_obj_visual_embeddings.py \
    --config ./preprocessing/sg_features/obj_visual_embeddings/Dinov2/obj_visual_embeddings.yaml \
    --split train

python ./preprocessing/sg_features/obj_visual_embeddings/Dinov2/gen_obj_visual_embeddings.py \
    --config ./preprocessing/sg_features/obj_visual_embeddings/Dinov2/obj_visual_embeddings.yaml \
    --split val

python ./preprocessing/sg_features/obj_visual_embeddings/Dinov2/gen_obj_visual_embeddings.py \
    --config ./preprocessing/sg_features/obj_visual_embeddings/Dinov2/obj_visual_embeddings.yaml \
    --split test

