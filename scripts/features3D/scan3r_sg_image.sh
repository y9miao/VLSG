export VLSG_SPACE=${dir of the VLSG repo}
export Data_ROOT_DIR=${dir of the dataset (path to "3RScan")}

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

