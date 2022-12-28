#!/bin/bash
echo ========================================
echo 1/10: Extract frames
echo ========================================
conda activate ROMP
python save_video_frames.py --video /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427.MOV --save_to /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/raw_720p  --width 1280 --height 720 --every 10 --skip=0
conda deactivate
echo ========================================
echo 2/10: Masks
echo ========================================
cd /neuman/preprocess/detectron2/demo
conda activate ROMP
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/raw_720p/*.png --output /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/raw_masks  --opts MODEL.WEIGHTS ./model_final_2d9806.pkl
conda deactivate
cd /neuman/preprocess
echo ========================================
echo 3/10: Sparse scene reconstrution
echo ========================================
cd /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427
mkdir recon
colmap feature_extractor --database_path ./recon/db.db --image_path ./raw_720p --ImageReader.mask_path ./raw_masks --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pool=true --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1
colmap exhaustive_matcher --database_path ./recon/db.db --SiftMatching.guided_matching=true
mkdir -p ./recon/sparse
colmap mapper --database_path ./recon/db.db --image_path ./raw_720p --output_path ./recon/sparse
if [ -d "./recon/sparse/1" ]; then echo "Bad reconstruction"; exit 1; else echo "Ok"; fi
mkdir -p ./recon/dense
colmap image_undistorter --image_path raw_720p --input_path ./recon/sparse/0/ --output_path ./recon/dense
colmap patch_match_stereo --workspace_path ./recon/dense
colmap model_converter --input_path ./recon/dense/sparse/ --output_path ./recon/dense/sparse --output_type=TXT
mkdir ./output
cp -r ./recon/dense/images ./output/images
cp -r ./recon/dense/stereo/depth_maps ./output/depth_maps
cp -r ./recon/dense/sparse ./output/sparse
cd /neuman/preprocess
echo ========================================
echo 4/10: Masks for rectified images
echo ========================================
cd /neuman/preprocess/detectron2/demo
conda activate ROMP
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/images/*.png --output /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/segmentations  --opts MODEL.WEIGHTS ./model_final_2d9806.pkl
conda deactivate
cd /neuman/preprocess
echo ========================================
echo 5/10: DensePose
echo ========================================
cd /neuman/preprocess/detectron2/projects/DensePose
conda activate ROMP
#wget https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl
python apply_net.py dump configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml model_final_844d15.pkl /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/images /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/densepose --output /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/densepose/output.pkl -v
#python apply_net.py dump configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/images /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/densepose --output /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/densepose/output.pkl -v
conda deactivate
cd /neuman/preprocess
echo ========================================
echo 6/10: 2D keypoints
echo ========================================
cd /neuman/preprocess/mmpose
conda activate open-mmlab
#wget https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512_udp-7cad61ef_20210222.pth
python demo/bottom_up_img_demo.py configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512_udp.py higher_hrnet48_coco_512x512_udp-7cad61ef_20210222.pth --img-path /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/images --out-img-root /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/keypoints --kpt-thr=0.3 --pose-nms-thr=0.9
conda deactivate
cd /neuman/preprocess
echo ========================================
echo 7/10: Monocular depth
echo ========================================
cd /neuman/preprocess/BoostingMonocularDepth
conda activate ROMP
python run.py --Final --data_dir /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/images --output_dir /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/mono_depth --depthNet 2
conda deactivate
cd /neuman/preprocess
echo ========================================
echo 8/10: SMPL parameters
echo ========================================
cd /neuman/preprocess/ROMP
wget https://github.com/jiangwei221/ROMP/releases/download/v1.1/model_data.zip
unzip model_data.zip
wget https://github.com/Arthur151/ROMP/releases/download/v1.1/trained_models_try.zip
unzip trained_models_try.zip
conda activate ROMP
python -m romp.predict.image --inputs /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/images --output_dir /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/smpl_pred
conda deactivate
cd /neuman/preprocess
echo ========================================
echo 9/10: Solve scale ambiguity
echo ========================================
cd /neuman/preprocess
conda activate neuman_env
python export_alignment.py --scene_dir /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/sparse --images_dir /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/images --raw_smpl /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output/smpl_pred --smpl_estimator="romp"
conda deactivate
cd /neuman/preprocess
echo ========================================
echo 10/10: Optimize SMPL using silhouette
echo ========================================
cd /neuman/preprocess
conda activate neuman_env
python optimize_smpl.py --scene_dir /home/disk1/inhee/videos/iphone_inhee_statue/IMG_3427/output
conda deactivate
cd /neuman/preprocess
