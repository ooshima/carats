# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import linalg as LA

#///// ���x10000ft�ȉ��C�k��35�x��C���o139�x��̃f�[�^�𒊏o�@/////

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
			#���x10000ft�ȉ��C�k��35�x��C���o139�x��𒊏o
			df = df[df['height']<=10000]
			df = df[df['lat'] < 36]
			df = df[df['lat'] >= 35]
			df = df[df['long'] < 140]
			df = df[df['long'] >= 139]
			#1�����̃f�[�^�t���[��
			dfday = pd.concat([dfday, df])

		#���x��float��int�Ɍ^�ϊ�
		dfday['height']= dfday['height'].astype(int)

		#csv�ɏ����o��
		fn = file.split('_', 1)[0]
		dfday.to_csv('/Users/fragrantflower/Desktop/carats/10-below10000ft+lat35+long139/' + fn + '_below10000ft+lat35+long139.csv', header=False, index=False)

#///// �����E�o���̔���@/////

#�f�[�^�ۑ��ꏊ�ֈړ�
dir_data = '/Users/fragrantflower/Desktop/carats/10-below10000ft+lat35+long139'
os.chdir(dir_data)

#�����Ώۃt�@�C���ꗗ���擾
filelist = os.listdir(dir_data)

#�t�@�C���ꗗ����s�v�ȍ��ڂ��폜
for (i, file) in enumerate(filelist):
	if file.split('.', 1)[1] <> 'csv':
		del filelist[i]

#���ʂɏ���
for file in filelist:
	print '------'
	print file

	df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')
	
	#�ւ��ƁC���n��Ń\�[�g
	df = df.sort_index(by = ['flight', 'time'])
	
	#�֖����d���Ȃ��̃��X�g�ɂ��ă\�[�g
	ls_flight = sorted(set(list(df['flight'].values.flatten())))
	
	arrdep = []
	ls_flt = []
	
	#�ւ��Ƃɔ���
	for i, flight in enumerate(ls_flight):

		#�Y���ւ̃f�[�^�̂ݒ��o
		df_tmp = df[df['flight']==flight]
		#�f�[�^�t���[������z��ɕϊ�
		ary_data = df_tmp.as_matrix()

		#���x�̗�̂ݒ��o
		ary_height = ary_data[:,4]

		#�㏸�񐔁C���~�񐔃J�E���g�p
		n_up = 0
		n_down = 0
		
		#�����Y���ւ̃f�[�^��1���Ȃ�G���[�ŏI��
		if len(ary_data) < 2:
			ls_flt.append(flight)
			arrdep.append('error')
		#2���ȏ゠��Ȃ獷�����v�Z�i���X�g�ɂ���j
		else:
			#ary1�͐擪�f�[�^�폜�Cary2�͖����f�[�^�폜���āCary1-ary2���v�Z(t+1��t�̍���)
			ary1 = ary_height
			ary2 = ary_height
			ary1 = np.delete(ary1, 0, 0)
			ary2 = np.delete(ary2, len(ary2)-1, 0)
			ary_diff = ary1-ary2

			#�㏸�C���~���J�E���g
			n_up = len(np.where(ary_diff>0)[0])
			n_down = len(np.where(ary_diff<0)[0])

			#�㏸�̕����������dep,���~�̕����������arr�C�����Ȃ�error		
			if n_up > n_down:
				ls_flt.append(flight)
				arrdep.append('dep')
			elif n_up < n_down:
				ls_flt.append(flight)
				arrdep.append('arr')
			else:
				ls_flt.append(flight)
				arrdep.append('error')
		
	df_new = pd.DataFrame(ls_flt)
	df_new.columns = ['flight']
	df_new['arr/dep'] = pd.DataFrame(arrdep)
	
	#csv�ɏ����o��
	fn = file.split('_', 1)[0]
	df_new.to_csv('/Users/fragrantflower/Desktop/carats/20-arr+dep+error(flight)/' + fn + '_arr+dep+error(flight).csv', header=False, index=False)


#///// �o���E�����ʂɉH�c���Ӄf�[�^�𕪂���@/////

#�f�[�^�ۑ��ꏊ�ֈړ�
dir_data = '/Users/fragrantflower/Desktop/carats/10-below10000ft+lat35+long139'
os.chdir(dir_data)

#�����Ώۃt�@�C���ꗗ���擾
filelist = os.listdir(dir_data)

#�t�@�C���ꗗ����s�v�ȍ��ڂ��폜
for (i, file) in enumerate(filelist):
	if file.split('.', 1)[1] <> 'csv':
		del filelist[i]

#���ʂɏ���
for file in filelist:
	print '------'
	print file

	#�����ʍq�Ճf�[�^�i�[�p�f�[�^�t���[��
	df_arr = pd.DataFrame(columns=['time', 'flight', 'lat', 'long', 'height', 'aircraft'])
	df_dep = pd.DataFrame(columns=['time', 'flight', 'lat', 'long', 'height', 'aircraft'])
	
	#�q�Ճf�[�^��ǂݍ���
	df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')

	#�����֖����X�g��ǂݍ���
	df_arrdep = pd.read_csv('/Users/fragrantflower/Desktop/carats/20-arr+dep+error(flight)/' + file[0:12] + 'arr+dep+error(flight).csv', header=None, names=['flight', 'arr/dep'], encoding='Shift_JIS')

	#�����փ��X�g�𔭒����Ƃɕ�����
	ary_arr = np.array(df_arrdep[df_arrdep['arr/dep'] == 'arr'].iloc[:,0])
	ary_dep = np.array(df_arrdep[df_arrdep['arr/dep'] == 'dep'].iloc[:,0])

	#�q�Ճf�[�^�֖̕����d���Ȃ��̃��X�g�ɂ��ă\�[�g
	ls_flight = sorted(set(list(df['flight'].values.flatten())))

	#�������Ƃɍq�Ճf�[�^�𕪂���
	for flt in ls_flight:
		df_tmp = df[df['flight']==flt]
		#�֖��������փ��X�g�ɂ���Γ����C�o���փ��X�g�ɂ���Ώo��
		if flt in ary_arr:
			df_arr = pd.concat([df_arr, df_tmp])
		elif flt in ary_dep:
			df_dep = pd.concat([df_dep, df_tmp])

	#���n��Ń\�[�g
#	df_arr = df_arr.sort_index(by = ['time', 'flight'])
#	df_dep = df_dep.sort_index(by = ['time', 'flight'])

	#���x��float��int�Ɍ^�ϊ�
	df_arr['height']= df_arr['height'].astype(int)
	df_dep['height']= df_dep['height'].astype(int)

	#csv�ɏ����o��
	fn = file.split('_', 1)[0]
	df_arr.to_csv('/Users/fragrantflower/Desktop/carats/30-arr+dep/' + fn + '_arr.csv', header=False, index=False)
	df_dep.to_csv('/Users/fragrantflower/Desktop/carats/30-arr+dep/' + fn + '_dep.csv', header=False, index=False)


#/////�@��������ڎ��Ŕ��肷�邽�߃v���b�g�p�̖��[�f�[�^�𒊏o�@/////

#�f�[�^�ۑ��ꏊ�ֈړ�
dir_data = '/Users/fragrantflower/Desktop/carats/30-arr+dep'
os.chdir(dir_data)

#�����Ώۃt�@�C���ꗗ���擾
filelist = os.listdir(dir_data)

#�t�@�C���ꗗ����s�v�ȍ��ڂ��폜
for (i, file) in enumerate(filelist):
	if file.split('.', 1)[1] <> 'csv':
		del filelist[i]

#�t�@�C���ꗗ��arr��dep�ŕ�����
arrlist = filter((lambda file: file[12:15]=='arr'), filelist)
deplist = filter((lambda file: file[12:15]=='dep'), filelist)

#�����ւ������ilatest�̂ݒ��o�j
for file in arrlist:
	print file

	#���o�f�[�^�̊i�[�p
	df_edge = pd.DataFrame(columns=['time', 'flight', 'lat', 'long', 'height', 'aircraft'])

	#�q�Ճf�[�^��ǂݍ���
	df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')

	#latest�f�[�^�̂ݒ��o
	df_grouped = df.groupby('flight', as_index = False).agg({'time' :[np.max]})

	for i in range(len(df_grouped)):
		df_tmp1 = df[df['flight']==df_grouped.ix[i,0]]
		df_tmp2 = df_tmp1[df_tmp1['time']==df_grouped.ix[i,1]]
		df_edge = pd.concat([df_edge, df_tmp2])
	
	#csv�ɏ����o��
	fn = file.split('_', 1)[0]
	df_edge.to_csv('/Users/fragrantflower/Desktop/carats/40-edge/' + fn + '_arr.csv', header=False, index=False)

#�o���ւ������ifastest�̂ݒ��o�j
for file in deplist:
	print file

	#���o�f�[�^�̊i�[�p
	df_edge = pd.DataFrame(columns=['time', 'flight', 'lat', 'long', 'height', 'aircraft'])

	#�q�Ճf�[�^��ǂݍ���
	df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')

	#fasest�f�[�^�̂ݒ��o
	df_grouped = df.groupby('flight', as_index = False).agg({'time' :[np.min]})

	for i in range(len(df_grouped)):
		df_tmp1 = df[df['flight']==df_grouped.ix[i,0]]
		df_tmp2 = df_tmp1[df_tmp1['time']==df_grouped.ix[i,1]]
		df_edge = pd.concat([df_edge, df_tmp2])
	
	#csv�ɏ����o��
	fn = file.split('_', 1)[0]
	df_edge.to_csv('/Users/fragrantflower/Desktop/carats/40-edge/' + fn + '_fastest_dep.csv', header=False, index=False)

	
#/////�@��������ڎ��Ŕ��肷�邽�ߖ��[�f�[�^���W�����̂܂܃v���b�g�@/////

#�f�[�^�ۑ��ꏊ�ֈړ�
dir_data = '/Users/fragrantflower/Desktop/carats/40-edge'
os.chdir(dir_data)

#�����Ώۃt�@�C���ꗗ���擾
filelist = os.listdir(dir_data)

#�t�@�C���ꗗ����s�v�ȍ��ڂ��폜
for (i, file) in enumerate(filelist):
	if file.split('.', 1)[1] <> 'csv':
		del filelist[i]

#�t�@�C���ꗗ��arr��dep�ŕ�����
arrlist = filter((lambda file: file[12:15]=='arr'), filelist)
deplist = filter((lambda file: file[12:15]=='dep'), filelist)

#�����H�[�̍��W lat, long, 16L, 34R, 16R, 34L, 04, 22, 05, 23
runway = np.array([
['16L', 35.565897, 139.786553], ['34R', 35.539694, 139.805136], ['16R', 35.559986, 139.769067], ['34L', 35.5366, 139.786559],
['04', 35.549019, 139.761278], ['22', 35.567467, 139.777114], ['05', 35.524003, 139.803464], ['23', 35.540597, 139.822114]
])

#�����ւ�����
fig, axes = plt.subplots(nrows = 6, ncols = 7, figsize = (20, 12))
k = 0;
for i in range(6):
	for j in range(7):
		lat = []
		long = []
		height = []
			
		#�q�Ճf�[�^��ǂݍ���
		file = arrlist[k]
		df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')

		#�ܓx�C�o�x�C���x
		lat.extend(df['lat'].values.flatten())
		long.extend(df['long'].values.flatten())
		height.extend(df['height'].values.flatten())
	
		axes[i,j].scatter(long, lat, color = 'r', edgecolors = 'k', s=10)
		axes[i,j].set_title('arr at ' + file[3:11])
		
		print file
		k += 1

#�����H���v���b�g�C�l������낦��
for ax in axes.flatten():
	#�����HA
		ax.plot(runway[:,2][0:2], runway[:,1][0:2], color = 'k', linewidth = 3, alpha = 0.6)
	#�����HB
		ax.plot(runway[:,2][2:4], runway[:,1][2:4], color = 'k', linewidth = 3, alpha = 0.6)
	#�����HC
		ax.plot(runway[:,2][4:6], runway[:,1][4:6], color = 'k', linewidth = 3, alpha = 0.6)
	#�����HD
		ax.plot(runway[:,2][6:8], runway[:,1][6:8], color = 'k', linewidth = 3, alpha = 0.6)
		
		ax.set_xlim([139.74, 139.84])
		ax.set_ylim([35.48, 35.6])
		ax.grid(True)

plt.tight_layout()
plt.show()

#png�t�@�C���ɏ����o��
fig.savefig('/Users/fragrantflower/Desktop/carats/40-edge/' + fn + '_latest_arr.png', dpi=90)
plt.close(fig)

#�o���ւ�����
fig, axes = plt.subplots(nrows = 6, ncols = 7, figsize = (20, 12))
k = 0;
for i in range(6):
	for j in range(7):
		lat = []
		long = []
		height = []
			
		#�q�Ճf�[�^��ǂݍ���
		file = deplist[k]
		df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')

		#�ܓx�C�o�x�C���x
		lat.extend(df['lat'].values.flatten())
		long.extend(df['long'].values.flatten())
		height.extend(df['height'].values.flatten())
	
		axes[i,j].scatter(long, lat, color = 'b', edgecolors='k', s=10)
		axes[i,j].set_title('dep at ' + file[3:11])
		
		print file
		k += 1

#�����H���v���b�g�C�l������낦��
for ax in axes.flatten():
	#�����HA
		ax.plot(runway[:,2][0:2], runway[:,1][0:2], color = 'k', linewidth = 3, alpha = 0.6)
	#�����HB
		ax.plot(runway[:,2][2:4], runway[:,1][2:4], color = 'k', linewidth = 3, alpha = 0.6)
	#�����HC
		ax.plot(runway[:,2][4:6], runway[:,1][4:6], color = 'k', linewidth = 3, alpha = 0.6)
	#�����HD
		ax.plot(runway[:,2][6:8], runway[:,1][6:8], color = 'k', linewidth = 3, alpha = 0.6)
		
		ax.set_xlim([139.74, 139.88])
		ax.set_ylim([35.45, 35.63])
		ax.grid(True)

plt.tight_layout()
plt.show()

#png�t�@�C���ɏ����o���i����p�ɂ�dpi350�`400�j
fig.savefig('/Users/fragrantflower/Desktop/carats/40-edge/' + fn + '_dep.png', dpi=90)
plt.close(fig)


#/////�@�Ώۓ���edge�f�[�^�𕗌����ʁ~�����ʂɓ����i���p�����H�̔���p�j�@/////

#�f�[�^�ۑ��ꏊ�ֈړ�
dir_data = '/Users/fragrantflower/Desktop/carats/50-targetday'
os.chdir(dir_data)

#�k���C�앗���Ƃɓ����C�o���փf�[�^����̃t�@�C���ɓ���
ary = ['n', 's']

for x in ary:

	#�����Ώۃt�@�C���ꗗ���擾
	os.chdir(dir_data + '/' + x)
	filelist = os.listdir(dir_data + '/' + x)

	#�t�@�C���ꗗ����s�v�ȍ��ڂ��폜
	for (i, file) in enumerate(filelist):
		if file.split('.', 1)[1] <> 'csv':
			del filelist[i]

	#�f�[�^�~�ϗp
	df_arr = pd.DataFrame(columns=['time', 'flight', 'lat', 'long', 'height', 'aircraft'])
	df_dep = pd.DataFrame(columns=['time', 'flight', 'lat', 'long', 'height', 'aircraft'])
	
	for file in filelist:
		#edge�f�[�^�̓ǂݍ���
		df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')
		
		if file[12:15] == 'arr':
			df_arr = pd.concat([df_arr, df])
		elif file[12:15] == 'dep':
			df_dep = pd.concat([df_dep, df])
			
	#csv�ɏ����o��
	fn = file.split('_', 1)[0]
	df_arr.to_csv('/Users/fragrantflower/Desktop/carats/50-targetday/' + x + '_targets_arr_latest.csv', header=False, index=False)
	df_dep.to_csv('/Users/fragrantflower/Desktop/carats/50-targetday/' + x + '_targets_dep_fastest.csv', header=False, index=False)


#/////�@�Ώۓ���edge�f�[�^�𕗌����ʁ~�����ʂɃv���b�g�i���p�����H�̔���p�j�@/////

#�f�[�^�ۑ��ꏊ�ֈړ�
dir_data = '/Users/fragrantflower/Desktop/carats/50-targetday/targets_edge'
os.chdir(dir_data)

#�����Ώۃt�@�C���ꗗ���擾
filelist = os.listdir(dir_data)

#�t�@�C���ꗗ����s�v�ȍ��ڂ��폜
for (i, file) in enumerate(filelist):
	if file.split('.', 1)[1] <> 'csv':
		del filelist[i]

#�����H�[�̍��W lat, long, 16L, 34R, 16R, 34L, 04, 22, 05, 23
runway = np.array([
['16L', 35.565897, 139.786553], ['34R', 35.539694, 139.805136], ['16R', 35.559986, 139.769067], ['34L', 35.5366, 139.786559],
['04', 35.549019, 139.761278], ['22', 35.567467, 139.777114], ['05', 35.524003, 139.803464], ['23', 35.540597, 139.822114]
])

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 10))

k = 0
for i in range(2):
	for j in range(2):
		lat = []
		long = []
		
		file = filelist[k]
		#�Ώۓ���edge�����f�[�^��ǂݍ���
		df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')
		
		#�ܓx�C�o�x
		lat.extend(df['lat'].values.flatten())
		long.extend(df['long'].values.flatten())
		
		axes[i, j].scatter(long, lat)
		axes[i, j].set_title(file)
		
		print file
		k += 1

#�����H���v���b�g
for ax in axes.flatten():
	#�����HA
		ax.plot(runway[:,2][0:2], runway[:,1][0:2], color = 'k', linewidth = 3, alpha = 0.6)
	#�����HB
		ax.plot(runway[:,2][2:4], runway[:,1][2:4], color = 'k', linewidth = 3, alpha = 0.6)
	#�����HC
		ax.plot(runway[:,2][4:6], runway[:,1][4:6], color = 'k', linewidth = 3, alpha = 0.6)
	#�����HD
		ax.plot(runway[:,2][6:8], runway[:,1][6:8], color = 'k', linewidth = 3, alpha = 0.6)

#�e�O���t�̃v���b�g�͈͂ƃX�P�[���̒���
#�k���~����
axes[0,0].set_xlim([139.77, 139.82])
axes[0,0].set_ylim([35.514, 35.5475])
axes[0,0].grid(True)

#�k���~�o��
axes[0,1].set_xlim([139.75, 139.86])
axes[0,1].set_ylim([35.51, 35.6])
axes[0,1].grid(True)

#�앗�~����
axes[1,0].set_xlim([139.75, 139.83])
axes[1,0].set_ylim([35.52, 35.58])
axes[1,0].grid(True)

#�앗�~�o��
axes[1,1].set_xlim([139.75, 139.86])
axes[1,1].set_ylim([35.476, 35.557])
axes[1,1].grid(True)

plt.tight_layout()
plt.show()

#png�t�@�C���ɏ����o���i����p�ɂ�dpi350�`400�j
#fig.savefig('/Users/fragrantflower/Desktop/carats/40-edge/' + fn + '_dep.png', dpi=90)
#plt.close(fig)

