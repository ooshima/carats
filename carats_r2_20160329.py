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

#////////// 羽田周辺（緯度35度，経度139度周辺）のデータを抽出（高度10000ft以上/羽田非就航便も含む） ///////////

#データ保存場所へ移動
folder_root = '/Users/fragrantflower/Desktop/OpenData/data'
os.chdir(folder_root)

#月ごとのフォルダ名一覧を取得
monthlist = os.listdir(folder_root)
	if '.DS_Store' in monthlist:
		monthlist.remove(".DS_Store")

#月ごとにファイル処理
for month in monthlist:
	foldername = folder_root + '/' + month
	#月ごとのフォルダへ移動
	os.chdir(foldername)
	#月ごとのファイル一覧を取得
	filelist = os.listdir(os.getcwd())
	#ファイル一覧から不要な項目を削除
	for (i, file) in enumerate(filelist):
		if file[0:3] <> 'trk':
			del filelist[i]
	#ファイル名一覧を1日3ファイル×7日に整形
	filelist_reshape = np.reshape(filelist,(7,3))
	print '----------'
	
	#1日3ファイルずつ7日分を処理
	for day in range(0,7):
		print file
		dfday = pd.DataFrame(columns=['time', 'flight', 'lat', 'long', 'height', 'aircraft'])
		for file in filelist_reshape[day]:
			df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')
			#北緯35度台，東経139度台の周辺を適当に抽出
			df = df[df['lat'] < 35.85]
			df = df[df['lat'] >= 35.25]
			df = df[df['long'] < 140.1]
			df = df[df['long'] >= 139.5]
			#1日分のデータフレーム
			dfday = pd.concat([dfday, df])

		#高度をfloat→intに型変換
		dfday['height']= dfday['height'].astype(int)

		#csvに書き出し
		fn = file.split('_', 1)[0]
		dfday.to_csv('/Users/fragrantflower/Desktop/carats/00_N35+E139/' + fn + '_N35+E139.csv', header=False, index=False)


#////////// 羽田周辺データをフライトごとにプロット（時系列で赤→青に色分け，滑走路込み） //////////

#データ保存場所へ移動
dir_data = '/Users/fragrantflower/Desktop/carats/00_N35+E139/'
os.chdir(dir_data)

#ファイル名一覧を取得
filelist = os.listdir(dir_data)
if '.DS_Store' in filelist:
	filelist.remove(".DS_Store")
		
filelist = ['trk20130110_N35+E139.csv']	#北風
#filelist = ['trk20120711_N35+E139.csv']	#南風

#滑走路端の座標 lat, long, 16L, 34R, 16R, 34L, 04, 22, 05, 23
runway = np.array([
['16L', 35.565897, 139.786553], ['34R', 35.539694, 139.805136], ['16R', 35.559986, 139.769067], ['34L', 35.5366, 139.786559],
['04', 35.549019, 139.761278], ['22', 35.567467, 139.777114], ['05', 35.524003, 139.803464], ['23', 35.540597, 139.822114]
])

for file in filelist:
	print file
	
	#羽田周辺データを読み込んで便ごと，時系列でソート
	df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')
	df = df.sort_index(by = ['flight', 'time'])
	
	#便名を重複なしのリストにしてソート
	ls_flight = sorted(set(list(df['flight'].values.flatten())))
	
	#データフレームから配列に変換
	ary_data = df.as_matrix()
	
	#緯度経度の値域を指定
	latmin = 35.3
	latmax = 35.8
	longmin = 139.55
	longmax = 140.05
	
	#該当日のデータのみストックする配列
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
				
		#timeを正規化
		time_norm = numpy.array(time, numpy.float)
		time_norm = time_norm/max(time_norm)
		time_norm = np.round(time_norm*100)/100

		fig = plt.figure(figsize = (5,3))

		plt.xlim([longmin, longmax])
		plt.ylim([latmin, latmax])

		plt.scatter(long, lat, c=time_norm)

		#滑走路をプロット
		#滑走路A
		plt.plot(runway[:,2][0:2], runway[:,1][0:2], color = 'k', linewidth = 3, alpha = 0.6)
		#滑走路B
		plt.plot(runway[:,2][2:4], runway[:,1][2:4], color = 'k', linewidth = 3, alpha = 0.6)
		#滑走路C
		plt.plot(runway[:,2][4:6], runway[:,1][4:6], color = 'k', linewidth = 3, alpha = 0.6)
		#滑走路D
		plt.plot(runway[:,2][6:8], runway[:,1][6:8], color = 'k', linewidth = 3, alpha = 0.6)

		plt.colorbar()

		#pngファイルに書き出し
		fn = file.split('_', 1)[0]				
		fig.savefig('/Users/fragrantflower/Desktop/carats/10_N35+E139_plot/' + fn + '_' + flight + '_N35+E139.png', dpi = 45)
				
		plt.close(fig)
		

#////////// データセットを訓練，CV，テスト用に分ける 南風用 ///////////

import random
import shutil

#データ保存場所へ移動
dir_data = '/Users/fragrantflower/Desktop/carats/10_N35+E139_plot/south/'
os.chdir(dir_data)

#ファイル名一覧を取得
filelist = os.listdir(dir_data)
if '.DS_Store' in filelist:
	filelist.remove(".DS_Store")

#すでにフォルダがあれば仕分け済みと見なして処理対象から除外
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

#train_data = random.choice(i)	#ランダムに一つだけ取り出す
#train_data = random.sample(filelist, 720)	#ランダムに複数取り出す
random.shuffle(filelist)	#ランダムにシャッフル

#リストのn番目からm-1番目の要素を取り出す
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


#////////// 仕分け結果を便名と併記したリストの作成（正解データ） //////////

#データ保存場所へ移動
dir_data = '/Users/fragrantflower/Desktop/carats/20_south/test/'
os.chdir(dir_data)

#フォルダ名一覧を取得
dirlist = os.listdir(dir_data)
del dirlist[0]

ls_answer = []

#正解ごとにファイル処理
for dir in dirlist:
	#ファイル名一覧を取得
	filelist = os.listdir(dir)
	if '.DS_Store' in filelist:
		filelist.remove(".DS_Store")
	#便名に仕分け結果を併記したリストの作成
	for file in filelist:
		ls_answer.append([file, dir])
		
#csv書き出し用にリストをデータフレームに変換
df = pd.DataFrame(ls_answer)

#csvファイルに書き出し
fn = file.split('_', 1)[0]
df.to_csv(dir_data + '/' + fn + '_answer.csv', header=False, index=False)


#-------　正解データの数が合わないときに欠落しているファイルを特定する　-----

#例）訓練データ　setにすると重複した要素は削除される
#正解仕分け前の訓練データの一覧
old_set = set(train_data)	#もしくは訓練データ一覧の入ったフォルダのファイル名一覧を作る

#正解仕分け後のデータ一覧
ary_answer = np.array(ls_answer)	#一列目だけ取り出すためにリストから配列に変換
new_set = set(ary_answer[:,0])	#ファイル名だけ取り出す

#old_setには含まれるがnew_setには含まれない要素を持った新しいsetの作成
old_set.difference(new_set)


