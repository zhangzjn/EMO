#!/bin/zsh
echo 'Install packages ...'
pip3 install terminaltables mmpycocotools prettytable xtcocotools timm==0.6.5
pip3 install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip3 install --upgrade numpy

echo 'Create VM ...'
mkdir /dev/shm/tmp
mount -t tmpfs -o size=1200G -o nr_inodes=10000000 tmpfs /dev/shm/tmp

mkdir /dev/shm/tmp/VOCdevkit
cd /youtu_fuxi_team1_ceph/vtzhang/codes/pts_cls/down-stream-tasks/mmsegmentation
echo 'copy [VOC2012] ...'
rsync -av /youtu_fuxi_team1_ceph/vtzhang/codes/data/voc/VOCdevkit/VOC2012 /dev/shm/tmp/VOCdevkit
echo '[VOC2012] finished ...'
#rsync -av /youtu_fuxi_team1_ceph/vtzhang/codes/data/voc/VOCdevkit/VOC2007 /dev/shm/tmp/VOCdevkit
#echo '[VOC2007] finished ...'

echo 'copy [coco_2017] ...'
rsync -av /youtu_fuxi_team1_ceph/vtzhang/codes/data/voc/VOCdevkit/coco2017tovoc /dev/shm/tmp/VOCdevkit
echo '[coco_2017] finished ...'

#echo 'copy [coco_2017] ...'
#mkdir /dev/shm/tmp/VOCdevkit/coco2017tovoc
#python3 copy_coco.py -s /youtu_fuxi_team1_ceph/vtzhang/codes/data/voc/VOCdevkit/coco2017tovoc -m train -y 2017 -t 20
#echo '[coco_2017] train finished ...'
#python3 copy_coco.py -s /youtu_fuxi_team1_ceph/vtzhang/codes/data -m val -y 2017 -t 20
#echo '[coco_2017] val finished ...'
#python3 copy_coco.py -s /youtu_fuxi_team1_ceph/vtzhang/codes/data -m test -y 2017 -t 20
#echo '[coco_2017] test finished ..'
#rsync -av /youtu_fuxi_team1_ceph/vtzhang/codes/data/coco_2017/annotations /dev/shm/tmp/coco_2017
#rsync -av /youtu_fuxi_team1_ceph/vtzhang/codes/data/coco_2017/person_detection_results /dev/shm/tmp/coco_2017


#ln -sf /dev/shm/tmp/VOCdevkit ./data/VOCdevkit

#echo 'copy [ADEChallengeData2016] ...'
#mkdir /dev/shm/tmp/ade
#cd /youtu_fuxi_team1_ceph/vtzhang/codes/pts_cls/EAFormer_det_seg
#rsync -av /youtu_fuxi_team1_ceph/vtzhang/codes/data/ade/ADEChallengeData2016 /dev/shm/tmp/ade
#ln -sf /dev/shm/tmp/ade ./semantic_segmentation/data/ade

cd /youtu_fuxi_team1_ceph/vtzhang/codes/pts_cls/refs/apex
pip3 install -v --disable-pip-version-check --no-cache-dir ./
cd /youtu_fuxi_team1_ceph/vtzhang/codes/pts_cls/refs/mmsegmentation

# python3 tools/convert_datasets/voc_aug.py /youtu_fuxi_team1_ceph/vtzhang/codes/data/voc/VOCdevkit /youtu_fuxi_team1_ceph/vtzhang/codes/data/voc/VOCdevkit/VOCaug --nproc 8

#./tools/dist_train_vtzhang.sh configs/deeplabv3_emo/deeplabv3_metamobile_512x512_20k_voc12.py 8
#./tools/dist_train_vtzhang.sh configs/deeplabv3_emo/deeplabv3_metamobile_512x512_20k_voc12_pretrain_coco.py 8
#PORT=29501 ./tools/dist_train_vtzhang.sh configs/deeplabv3_emo/deeplabv3_r50-d8_512x512_20k_voc12aug.py 8