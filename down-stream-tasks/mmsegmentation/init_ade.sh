#!/bin/zsh
echo 'Install packages ...'
pip3 install terminaltables mmpycocotools prettytable xtcocotools timm==0.6.5
pip3 install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip3 install --upgrade numpy

echo 'Create VM ...'
mkdir /dev/shm/tmp
mount -t tmpfs -o size=1200G -o nr_inodes=10000000 tmpfs /dev/shm/tmp

echo 'copy [ADEChallengeData2016] ...'
mkdir /dev/shm/tmp/ade
cd /youtu_fuxi_team1_ceph/vtzhang/codes/pts_cls/down-stream-tasks/mmsegmentation
rsync -av /youtu_fuxi_team1_ceph/vtzhang/codes/data/ade/ADEChallengeData2016 /dev/shm/tmp/ade
#ln -sf /dev/shm/tmp/ade ./data/ade

cd /youtu_fuxi_team1_ceph/vtzhang/codes/pts_cls/refs/apex
pip3 install -v --disable-pip-version-check --no-cache-dir ./
cd /youtu_fuxi_team1_ceph/vtzhang/codes/pts_cls/refs/mmsegmentation

# python3 tools/convert_datasets/voc_aug.py /youtu_fuxi_team1_ceph/vtzhang/codes/data/voc/VOCdevkit /youtu_fuxi_team1_ceph/vtzhang/codes/data/voc/VOCdevkit/VOCaug --nproc 8

#./tools/dist_train_vtzhang.sh configs/deeplabv3_emo/deeplabv3_metamobile1M_pretrain_512x512_80k_ade20k.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/deeplabv3_emo/deeplabv3_metamobile2M_pretrain_512x512_80k_ade20k.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/deeplabv3_emo/deeplabv3_metamobile5M_pretrain_512x512_80k_ade20k.py 8
#./tools/dist_train_vtzhang.sh configs/deeplabv3_emo/deeplabv3_metamobile6M_pretrain_512x512_80k_ade20k.py 8 --auto-resume
#PORT=29501 ./tools/dist_train_vtzhang.sh configs/deeplabv3_emo/deeplabv3_metamobile5M_pretrain_512x512_80k_ade20k.py 8

#./tools/dist_train_vtzhang.sh configs/sem_fpn_metamobile/fpn_metamobile1M_512x512_80k_ade20k.py 8 --auto-resume
#./tools/dist_train_vtzhang.sh configs/sem_fpn_metamobile/fpn_metamobile2M_512x512_80k_ade20k.py 8
#./tools/dist_train_vtzhang.sh configs/sem_fpn_metamobile/fpn_metamobile5M_512x512_80k_ade20k.py 8
#./tools/dist_train_vtzhang.sh configs/sem_fpn_metamobile/fpn_metamobile6M_512x512_80k_ade20k.py 8

#./tools/dist_train_vtzhang.sh configs/pspnet_emo/pspnet_metamobile1M_512x512_80k_ade20k.py 8
#./tools/dist_train_vtzhang.sh configs/pspnet_emo/pspnet_metamobile2M_512x512_80k_ade20k.py 8
#./tools/dist_train_vtzhang.sh configs/pspnet_emo/pspnet_metamobile5M_512x512_80k_ade20k.py 8
#./tools/dist_train_vtzhang.sh configs/pspnet_emo/pspnet_metamobile6M_512x512_80k_ade20k.py 8 --auto-resume


#PYTHONPATH=".":$PYTHONPATH python3 tools/get_flops.py configs/deeplabv3_emo/deeplabv3_metamobile1M_pretrain_512x512_80k_ade20k.py --shape 512 512
#PYTHONPATH=".":$PYTHONPATH python3 tools/get_flops.py configs/sem_fpn_metamobile/fpn_metamobile1M_512x512_80k_ade20k.py --shape 512 512
#PYTHONPATH=".":$PYTHONPATH python3 tools/get_flops.py configs/pspnet_emo/pspnet_metamobile1M_512x512_80k_ade20k.py --shape 512 512


#PYTHONPATH=".":$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python3 tools/test.py configs/deeplabv3_emo/deeplabv3_emo_5M_pretrain_512x512_80k_ade20k.py work_dirs/deeplabv3_metamobile5M_pretrain_512x512_80k_ade20k_256/latest.pth --show-dir work_dirs/deeplabv3_metamobile5M_pretrain_512x512_80k_ade20k_256/show
#PYTHONPATH=".":$PYTHONPATH CUDA_VISIBLE_DEVICES=1 python3 tools/test.py configs/sem_fpn_metamobile/fpn_metamobile5M_512x512_80k_ade20k.py work_dirs/fpn_metamobile5M_512x512_80k_ade20k/latest.pth --show-dir work_dirs/fpn_metamobile5M_512x512_80k_ade20k/show
#PYTHONPATH=".":$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python3 tools/test.py configs/pspnet_emo/pspnet_emo_5M_512x512_80k_ade20k.py work_dirs/pspnet_metamobile5M_512x512_80k_ade20k_256/latest.pth --show-dir work_dirs/pspnet_metamobile5M_512x512_80k_ade20k_256/show

#./tools/dist_test_vtzhang.sh configs/deeplabv3_emo/deeplabv3_emo_5M_pretrain_512x512_80k_ade20k.py work_dirs/deeplabv3_metamobile5M_pretrain_512x512_80k_ade20k_256/latest.pth 2 --eval mIoU

