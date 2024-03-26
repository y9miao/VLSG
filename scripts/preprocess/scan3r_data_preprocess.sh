export VLSG_SPACE=${dir of the VLSG repo}
export Data_ROOT_DIR=${dir of the dataset (path to "3RScan")}

cd $VLSG_SPACE

python ./preprocessing/scan3r/preprocess_scan3r.py \
    --config ./preprocessing/scan3r/preprocess_scan3r.yaml