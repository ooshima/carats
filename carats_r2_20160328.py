# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import linalg as LA

#////////// �H�c���Ӂi�ܓx35�x�C�o�x139�x���Ӂj�̃f�[�^�𒊏o�i���x10000ft�ȏ�/�H�c��A�q�ւ��܂ށj ///////////

#�f�[�^�ۑ��ꏊ�ֈړ�
folder_root = '/Users/fragrantflower/Desktop/OpenData/data'
os.chdir(folder_root)

#�����Ƃ̃t�H���_���ꗗ���擾
monthlist = os.listdir(folder_root)
del monthlist[0]

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
			df = df[df['lat'] < 35.75]
			df = df[df['lat'] >= 35.25]
			df = df[df['long'] < 140]
			df = df[df['long'] >= 139.5]
			#1�����̃f�[�^�t���[��
			dfday = pd.concat([dfday, df])

		#���x��float��int�Ɍ^�ϊ�
		dfday['height']= dfday['height'].astype(int)

		#csv�ɏ����o��
		fn = file.split('_', 1)[0]
		dfday.to_csv('/Users/fragrantflower/Desktop/carats/00_N35+E139/' + fn + '_N35+E139.csv', header=False, index=False)


#////////// �H�c���Ӄf�[�^���t���C�g���ƂɃv���b�g�i�F�����Ȃ��C�����H���݁j //////////

#�f�[�^�ۑ��ꏊ�ֈړ�
dir_data = '/Users/fragrantflower/Desktop/carats/00_N35+E139/'
os.chdir(dir_data)

#�t�@�C�����ꗗ���擾
filelist = os.listdir(dir_data)
del filelist[0]

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
	latmin = 35.25
	latmax = 35.75
	longmin = 139.5
	longmax = 140
	
	#�Y�����̃f�[�^�̂݃X�g�b�N����z��
	ary_day = np.array([])
	
	for flight in ls_flight:
		lat = []
		long = []
		for data in ary_data:
			if flight == data[1]:
				lat.append(data[2])
				long.append(data[3])
				
				fig = plt.figure(figsize = (5,3))

				plt.xlim([longmin, longmax])
				plt.ylim([latmin, latmax])

				plt.plot(long, lat)

				#�����H���v���b�g
				#�����HA
				plt.plot(runway[:,2][0:2], runway[:,1][0:2], color = 'k', linewidth = 3, alpha = 0.6)
				#�����HB
				plt.plot(runway[:,2][2:4], runway[:,1][2:4], color = 'k', linewidth = 3, alpha = 0.6)
				#�����HC
				plt.plot(runway[:,2][4:6], runway[:,1][4:6], color = 'k', linewidth = 3, alpha = 0.6)
				#�����HD
				plt.plot(runway[:,2][6:8], runway[:,1][6:8], color = 'k', linewidth = 3, alpha = 0.6)



				#png�t�@�C���ɏ����o��
				fn = file.split('_', 1)[0]				
				fig.savefig('/Users/fragrantflower/Desktop/carats/10_N35+E139_plot/' + fn + '_' + flight + '_N35+E139.png', dpi = 45)
				
				plt.close(fig)










