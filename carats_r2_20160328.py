# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import linalg as LA

#////////// 羽田周辺（緯度35度，経度139度周辺）のデータを抽出（高度10000ft以上/羽田非就航便も含む） ///////////

#データ保存場所へ移動
folder_root = '/Users/fragrantflower/Desktop/OpenData/data'
os.chdir(folder_root)

#月ごとのフォルダ名一覧を取得
monthlist = os.listdir(folder_root)
del monthlist[0]

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
			df = df[df['lat'] < 35.75]
			df = df[df['lat'] >= 35.25]
			df = df[df['long'] < 140]
			df = df[df['long'] >= 139.5]
			#1日分のデータフレーム
			dfday = pd.concat([dfday, df])

		#高度をfloat→intに型変換
		dfday['height']= dfday['height'].astype(int)

		#csvに書き出し
		fn = file.split('_', 1)[0]
		dfday.to_csv('/Users/fragrantflower/Desktop/carats/00_N35+E139/' + fn + '_N35+E139.csv', header=False, index=False)


#////////// 羽田周辺データをフライトごとにプロット（色分けなし，滑走路込み） //////////

#データ保存場所へ移動
dir_data = '/Users/fragrantflower/Desktop/carats/00_N35+E139/'
os.chdir(dir_data)

#ファイル名一覧を取得
filelist = os.listdir(dir_data)
del filelist[0]

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
	latmin = 35.25
	latmax = 35.75
	longmin = 139.5
	longmax = 140
	
	#該当日のデータのみストックする配列
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

				#滑走路をプロット
				#滑走路A
				plt.plot(runway[:,2][0:2], runway[:,1][0:2], color = 'k', linewidth = 3, alpha = 0.6)
				#滑走路B
				plt.plot(runway[:,2][2:4], runway[:,1][2:4], color = 'k', linewidth = 3, alpha = 0.6)
				#滑走路C
				plt.plot(runway[:,2][4:6], runway[:,1][4:6], color = 'k', linewidth = 3, alpha = 0.6)
				#滑走路D
				plt.plot(runway[:,2][6:8], runway[:,1][6:8], color = 'k', linewidth = 3, alpha = 0.6)



				#pngファイルに書き出し
				fn = file.split('_', 1)[0]				
				fig.savefig('/Users/fragrantflower/Desktop/carats/10_N35+E139_plot/' + fn + '_' + flight + '_N35+E139.png', dpi = 45)
				
				plt.close(fig)










