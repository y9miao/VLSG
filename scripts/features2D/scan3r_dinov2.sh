export VLSG_SPACE=${dir of the VLSG repo}
export Data_ROOT_DIR=${dir of the dataset (path to "3RScan")}

cd $VLSG_SPACE

# generate patch-level features with Dinov2
python ./preprocessing/img_features/DinoV2/scan3r_dinov2_generator.py \
    --config ./preprocessing/img_features/DinoV2/scan3r_dinov2_generator.yaml

