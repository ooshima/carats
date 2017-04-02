# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import linalg as LA
import random
import shutil

#////////// �H�c���Ӂi�ܓx35�x�C�o�x139�x���Ӂj�̃f�[�^�𒊏o�i���x10000ft�ȏ�/�H�c��A�q�ւ��܂ށj ///////////

#�f�[�^�ۑ��ꏊ�ֈړ�
folder_root = '/Users/fragrantflower/Desktop/OpenData/data'
os.chdir(folder_root)

#�����Ƃ̃t�H���_���ꗗ���擾
monthlist = os.listdir(folder_root)
	if '.DS_Store' in monthlist:
		monthlist.remove(".DS_Store")

#�����ƂɃt�@�C������
for month in monthlist:
	foldername = folder_root + '/' + month
	#�����Ƃ̃t�H���_�ֈړ�
	os.chdir(foldername)
	#�����Ƃ̃t�@�C���ꗗ���擾
	filelist = os.listdir(os.getcwd())
	#�t�@�C���ꗗ����s�v�ȍ��ڂ��폜
	for (i, file) in enumerate(filelist):
		if file[0:3] <> 'trk':
			del filelist[i]
	#�t�@�C�����ꗗ��1��3�t�@�C���~7���ɐ��`
	filelist_reshape = np.reshape(filelist,(7,3))
	print '----------'
	
	#1��3�t�@�C������7����������
	for day in range(0,7):
		print file
		dfday = pd.DataFrame(columns=['time', 'flight', 'lat', 'long', 'height', 'aircraft'])
		for file in filelist_reshape[day]:
			df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')
			#�k��35�x��C���o139�x��̎��ӂ�K���ɒ��o
			df = df[df['lat'] < 35.85]
			df = df[df['lat'] >= 35.25]
			df = df[df['long'] < 140.1]
			df = df[df['long'] >= 139.5]
			#1�����̃f�[�^�t���[��
			dfday = pd.concat([dfday, df])

		#���x��float��int�Ɍ^�ϊ�
		dfday['height']= dfday['height'].astype(int)

		#csv�ɏ����o��
		fn = file.split('_', 1)[0]
		dfday.to_csv('/Users/fragrantflower/Desktop/carats/00_N35+E139/' + fn + '_N35+E139.csv', header=False, index=False)


#////////// �H�c���Ӄf�[�^���t���C�g���ƂɃv���b�g�i���n��Őԁ��ɐF�����C�����H���݁j //////////

#�f�[�^�ۑ��ꏊ�ֈړ�
dir_data = '/Users/fragrantflower/Desktop/carats/00_N35+E139/'
os.chdir(dir_data)

#�t�@�C�����ꗗ���擾
filelist = os.listdir(dir_data)
if '.DS_Store' in filelist:
	filelist.remove(".DS_Store")
		
filelist = ['trk20130110_N35+E139.csv']	#�k��
#filelist = ['trk20120711_N35+E139.csv']	#�앗

#�����H�[�̍��W lat, long, 16L, 34R, 16R, 34L, 04, 22, 05, 23
runway = np.array([
['16L', 35.565897, 139.786553], ['34R', 35.539694, 139.805136], ['16R', 35.559986, 139.769067], ['34L', 35.5366, 139.786559],
['04', 35.549019, 139.761278], ['22', 35.567467, 139.777114], ['05', 35.524003, 139.803464], ['23', 35.540597, 139.822114]
])

for file in filelist:
	print file
	
	#�H�c���Ӄf�[�^��ǂݍ���ŕւ��ƁC���n��Ń\�[�g
	df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')
	df = df.sort_index(by = ['flight', 'time'])
	
	#�֖����d���Ȃ��̃��X�g�ɂ��ă\�[�g
	ls_flight = sorted(set(list(df['flight'].values.flatten())))
	
	#�f�[�^�t���[������z��ɕϊ�
	ary_data = df.as_matrix()
	
	#�ܓx�o�x�̒l����w��
	latmin = 35.3
	latmax = 35.8
	longmin = 139.55
	longmax = 140.05
	
	#�Y�����̃f�[�^�̂݃X�g�b�N����z��
	ary_day = np.array([])
	
	for flight in ls_flight:
		lat = []
		long = []
		time = []
		n = 0
		
		for data in ary_data:
			if flight == data[1]:
				n += 1
				lat.append(data[2])
				long.append(data[3])
				time.append(n)
				
		#time�𐳋K��
		time_norm = numpy.array(time, numpy.float)
		time_norm = time_norm/max(time_norm)
		time_norm = np.round(time_norm*100)/100

		fig = plt.figure(figsize = (5,3))

		plt.xlim([longmin, longmax])
		plt.ylim([latmin, latmax])

		plt.scatter(long, lat, c=time_norm)

		#�����H���v���b�g
		#�����HA
		plt.plot(runway[:,2][0:2], runway[:,1][0:2], color = 'k', linewidth = 3, alpha = 0.6)
		#�����HB
		plt.plot(runway[:,2][2:4], runway[:,1][2:4], color = 'k', linewidth = 3, alpha = 0.6)
		#�����HC
		plt.plot(runway[:,2][4:6], runway[:,1][4:6], color = 'k', linewidth = 3, alpha = 0.6)
		#�����HD
		plt.plot(runway[:,2][6:8], runway[:,1][6:8], color = 'k', linewidth = 3, alpha = 0.6)

		plt.colorbar()

		#png�t�@�C���ɏ����o��
		fn = file.split('_', 1)[0]				
		fig.savefig('/Users/fragrantflower/Desktop/carats/10_N35+E139_plot/' + fn + '_' + flight + '_N35+E139.png', dpi = 45)
				
		plt.close(fig)
		

#////////// �f�[�^�Z�b�g���P���CCV�C�e�X�g�p�ɕ����� �앗�p ///////////

import random
import shutil

#�f�[�^�ۑ��ꏊ�ֈړ�
dir_data = '/Users/fragrantflower/Desktop/carats/10_N35+E139_plot/south/'
os.chdir(dir_data)

#�t�@�C�����ꗗ���擾
filelist = os.listdir(dir_data)
if '.DS_Store' in filelist:
	filelist.remove(".DS_Store")

#���łɃt�H���_������Ύd�����ς݂ƌ��Ȃ��ď����Ώۂ��珜�O
if os.path.isdir('./train/') is True:
	filelist.remove("train")
if os.path.isdir('./cv/') is True:
	filelist.remove("cv")
if os.path.isdir('./test/') is True:
	filelist.remove("test")

filelist_random = []
train_data = []
cv_data = []
test_data = []

#train_data = random.choice(i)	#�����_���Ɉ�������o��
#train_data = random.sample(filelist, 720)	#�����_���ɕ������o��
random.shuffle(filelist)	#�����_���ɃV���b�t��

#���X�g��n�Ԗڂ���m-1�Ԗڂ̗v�f�����o��
train_data = filelist[0:720]
cv_data = filelist[720:960]
test_data = filelist[960:(len(filelist)+1)]

path = './train/'
if os.path.isdir(path) is False:
	os.mkdir(path)
for file in train_data:
	shutil.move('./'+file, path)

path = './cv/'
if os.path.isdir(path) is False:
	os.mkdir(path)
for file in cv_data:
	shutil.move('./'+file, path)

path = './test/'
if os.path.isdir(path) is False:
	os.mkdir(path)
for file in test_data:
	shutil.move('./'+file, path)


#////////// �d�������ʂ�֖��ƕ��L�������X�g�̍쐬�i�����f�[�^�j //////////

#�f�[�^�ۑ��ꏊ�ֈړ�
dir_data = '/Users/fragrantflower/Desktop/carats/20_south/test/'
os.chdir(dir_data)

#�t�H���_���ꗗ���擾
dirlist = os.listdir(dir_data)
del dirlist[0]

ls_answer = []

#�������ƂɃt�@�C������
for dir in dirlist:
	#�t�@�C�����ꗗ���擾
	filelist = os.listdir(dir)
	if '.DS_Store' in filelist:
		filelist.remove(".DS_Store")
	#�֖��Ɏd�������ʂ𕹋L�������X�g�̍쐬
	for file in filelist:
		ls_answer.append([file, dir])
		
#csv�����o���p�Ƀ��X�g���f�[�^�t���[���ɕϊ�
df = pd.DataFrame(ls_answer)

#csv�t�@�C���ɏ����o��
fn = file.split('_', 1)[0]
df.to_csv(dir_data + '/' + fn + '_answer.csv', header=False, index=False)


#-------�@�����f�[�^�̐�������Ȃ��Ƃ��Ɍ������Ă���t�@�C������肷��@-----

#��j�P���f�[�^�@set�ɂ���Əd�������v�f�͍폜�����
#�����d�����O�̌P���f�[�^�̈ꗗ
old_set = set(train_data)	#�������͌P���f�[�^�ꗗ�̓������t�H���_�̃t�@�C�����ꗗ�����

#�����d������̃f�[�^�ꗗ
ary_answer = np.array(ls_answer)	#���ڂ������o�����߂Ƀ��X�g����z��ɕϊ�
new_set = set(ary_answer[:,0])	#�t�@�C�����������o��

#old_set�ɂ͊܂܂�邪new_set�ɂ͊܂܂�Ȃ��v�f���������V����set�̍쐬
old_set.difference(new_set)


