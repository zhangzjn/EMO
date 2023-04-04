import os
import shutil
import threading
import glob
import argparse
import paramiko

cur_cnt = 0
all_cnt = 0


def copy_s_imgs(s_imgs, t_imgs, mode):
	global cur_cnt
	for s_img, t_img in zip(s_imgs, t_imgs):
		shutil.copy(s_img, t_img)
		cur_cnt += 1
		print(f'{mode}: {cur_cnt}/{all_cnt}')

	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--source', type=str, default='/youtu_fuxi_team1_ceph/vtzhang/codes/data')
	parser.add_argument('-m', '--mode', type=str, default='train')
	parser.add_argument('-y', '--year', type=str, default='2017')
	parser.add_argument('-t', '--thr', type=int, default=20)
	parser.add_argument('--ip', type=str, default='')
	cfg = parser.parse_args()
	
	# =====> train
	thr = cfg.thr
	s_root = f'{cfg.source}/{cfg.mode}{cfg.year}'
	t_root = f'/dev/shm/tmp/VOCdevkit/coco2017tovoc/{cfg.mode}{cfg.year}'
	os.makedirs(t_root, exist_ok=True)
	s_imgs = glob.glob(f'{s_root}/*', recursive=True)
	t_imgs = [s_img.replace(s_root, t_root) for s_img in s_imgs]
	all_cnt = len(s_imgs)
	thr_num = all_cnt // thr
	
	s_imgs_sub = [s_imgs[i * thr_num:(i + 1) * thr_num] for i in range(thr-1)]
	s_imgs_sub.append(s_imgs[(thr - 1) * thr_num:])
	t_imgs_sub = [t_imgs[i * thr_num:(i + 1) * thr_num] for i in range(thr - 1)]
	t_imgs_sub.append(t_imgs[(thr - 1) * thr_num:])
	
	for i in range(thr):
		t = threading.Thread(target=copy_s_imgs, args=(s_imgs_sub[i], t_imgs_sub[i], cfg.mode))
		t.start()
