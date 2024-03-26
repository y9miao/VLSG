export VLSG_SPACE=${dir of the VLSG repo}
export Data_ROOT_DIR=${dir of the dataset (path to "3RScan")}

cd $VLSG_SPACE

# project 3D object annotations to 2D query images
python ./preprocessing/gt_anno_2D/scan3r_obj_projector.py
# aggretate pixel-wise annotations to patch-wise annotations
python ./preprocessing/gt_anno_2D/scan3r_obj_img_associate.py \
    --config ./preprocessing/gt_anno_2D/gt_anno.yaml

