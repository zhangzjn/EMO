#!/bin/zsh
echo 'Install packages ...'
pip3 install terminaltables mmpycocotools prettytable xtcocotools timm==0.6.5
pip3 install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip3 install --upgrade numpy

echo 'Create VM ...'
mkdir /dev/shm/tmp
mount -t tmpfs -o size=1200G -o nr_inodes=10000000 tmpfs /dev/shm/tmp

echo 'copy [coco_2017] ...'
mkdir /dev/shm/tmp/coco_2017
cd /youtu_fuxi_team1_ceph/vtzhang/codes/pts_cls/down-stream-tasks/mmdetection
python3 copy_coco.py -s /youtu_fuxi_team1_ceph/vtzhang/codes/data -m train -y 2017 -t 20
echo '[coco_2017] train finished ...'
python3 copy_coco.py -s /youtu_fuxi_team1_ceph/vtzhang/codes/data -m val -y 2017 -t 20
echo '[coco_2017] val finished ...'
python3 copy_coco.py -s /youtu_fuxi_team1_ceph/vtzhang/codes/data -m test -y 2017 -t 20
echo '[coco_2017] test finished ..'
rsync -av /youtu_fuxi_team1_ceph/vtzhang/codes/data/coco_2017/annotations /dev/shm/tmp/coco_2017
rsync -av /youtu_fuxi_team1_ceph/vtzhang/codes/data/coco_2017/person_detection_results /dev/shm/tmp/coco_2017
#ln -sf /dev/shm/tmp/coco_2017 data/coco

#echo 'copy [ADEChallengeData2016] ...'
#mkdir /dev/shm/tmp/ade
#cd /youtu_fuxi_team1_ceph/vtzhang/codes/pts_cls/EAFormer_det_seg
#rsync -av /youtu_fuxi_team1_ceph/vtzhang/codes/data/ade/ADEChallengeData2016 /dev/shm/tmp/ade
#ln -sf /dev/shm/tmp/ade ./semantic_segmentation/data/ade

cd /youtu_fuxi_team1_ceph/vtzhang/codes/pts_cls/refs/apex
pip3 install -v --disable-pip-version-check --no-cache-dir ./
cd /youtu_fuxi_team1_ceph/vtzhang/codes/pts_cls/refs/mmdetection

#./tools/dist_train_vtzhang.sh configs/ssd_emo/ssdlite_metamobile_scratch_600e_coco.py 8
#PORT=29501 ./tools/dist_train_vtzhang.sh configs/ssd_emo/ssdlite_emo_5M_pretrain_coco.py 8

#./tools/dist_train_vtzhang.sh configs/ssd_emo/ssdlite_emo_1M_pretrain_coco.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/ssd_emo/ssdlite_emo_2M_pretrain_coco.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/ssd_emo/ssdlite_emo_5M_pretrain_coco.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/ssd_emo/ssdlite_metamobile6M_pretrain_coco.py 8 --auto-resume

#./tools/dist_train_vtzhang.sh configs/retinanet_emo/retinanet_emo_1M_fpn_1x_coco.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/retinanet_emo/retinanet_emo_2M_fpn_1x_coco.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/retinanet_emo/retinanet_emo_5M_fpn_1x_coco.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/retinanet_emo/retinanet_metamobile6M_fpn_1x_coco.py 8 --auto-resume

#./tools/dist_train_vtzhang.sh configs/mask_rcnn_metamobile/mask_rcnn_metamobile1M_fpn_1x_coco.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/mask_rcnn_metamobile/mask_rcnn_metamobile2M_fpn_1x_coco.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/mask_rcnn_metamobile/mask_rcnn_metamobile5M_fpn_1x_coco.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/mask_rcnn_metamobile/mask_rcnn_metamobile6M_fpn_1x_coco.py 8 --auto-resume

#./tools/dist_train_vtzhang.sh configs/mask_rcnn_metamobile/mask_rcnn_metamobile1M_fpn_mstrain-poly_3x_coco.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/mask_rcnn_metamobile/mask_rcnn_metamobile2M_fpn_mstrain-poly_3x_coco.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/mask_rcnn_metamobile/mask_rcnn_metamobile5M_fpn_mstrain-poly_3x_coco.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/mask_rcnn_metamobile/mask_rcnn_metamobile6M_fpn_mstrain-poly_3x_coco.py 8 --auto-resume


#PYTHONPATH=".":$PYTHONPATH python3 tools/analysis_tools/get_flops.py configs/ssd_emo/ssdlite_emo_1M_pretrain_coco.py --shape 320 320
#PYTHONPATH=".":$PYTHONPATH python3 tools/analysis_tools/get_flops.py configs/retinanet_emo/retinanet_emo_1M_fpn_1x_coco.py --shape 1333 800
#PYTHONPATH=".":$PYTHONPATH python3 tools/analysis_tools/get_flops.py configs/mask_rcnn_metamobile/mask_rcnn_metamobile1M_fpn_1x_coco.py --shape 1333 800

#PYTHONPATH=".":$PYTHONPATH python3 tools/analysis_tools/get_flops.py configs/ssd_emo/ssdlite_r50_coco.py --shape 320 320
#PYTHONPATH=".":$PYTHONPATH python3 tools/analysis_tools/get_flops.py configs/pvt/retinanet_pvtv2-b0_fpn_1x_coco.py --shape 1333 800

#PYTHONPATH=".":$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python3 tools/test.py configs/ssd_emo/ssdlite_emo_5M_pretrain_coco.py work_dirs/ssdlite_metamobile5M_pretrain_coco/latest.pth --show-dir work_dirs/ssdlite_metamobile5M_pretrain_coco/show
#PYTHONPATH=".":$PYTHONPATH CUDA_VISIBLE_DEVICES=1 python3 tools/test.py configs/retinanet_emo/retinanet_emo_5M_fpn_1x_coco.py work_dirs/retinanet_metamobile5M_fpn_1x_coco/latest.pth --show-dir work_dirs/retinanet_metamobile5M_fpn_1x_coco/show
#PYTHONPATH=".":$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python3 tools/test.py configs/mask_rcnn_metamobile/mask_rcnn_metamobile5M_fpn_1x_coco.py work_dirs/mask_rcnn_metamobile5M_fpn_1x_coco/latest.pth --show-dir work_dirs/mask_rcnn_metamobile5M_fpn_1x_coco/show
#PYTHONPATH=".":$PYTHONPATH CUDA_VISIBLE_DEVICES=1 python3 tools/test.py configs/mask_rcnn_metamobile/mask_rcnn_metamobile5M_fpn_mstrain-poly_3x_coco.py work_dirs/mask_rcnn_metamobile5M_fpn_mstrain-poly_3x_coco/latest.pth --show-dir work_dirs/mask_rcnn_metamobile5M_fpn_mstrain-poly_3x_coco/show

#./tools/dist_test_vtzhang.sh configs/ssd_emo/ssdlite_emo_5M_pretrain_coco.py work_dirs/ssdlite_metamobile5M_pretrain_coco/latest.pth 2 --eval bbox
