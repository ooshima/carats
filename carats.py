# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import linalg as LA

#///// 高度10000ft以下，北緯35度台，東経139度台のデータを抽出　/////

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
			#高度10000ft以下，北緯35度台，東経139度台を抽出
			df = df[df['height']<=10000]
			df = df[df['lat'] < 36]
			df = df[df['lat'] >= 35]
			df = df[df['long'] < 140]
			df = df[df['long'] >= 139]
			#1日分のデータフレーム
			dfday = pd.concat([dfday, df])

		#高度をfloat→intに型変換
		dfday['height']= dfday['height'].astype(int)

		#csvに書き出し
		fn = file.split('_', 1)[0]
		dfday.to_csv('/Users/fragrantflower/Desktop/carats/10-below10000ft+lat35+long139/' + fn + '_below10000ft+lat35+long139.csv', header=False, index=False)

#///// 到着・出発の判定　/////

#データ保存場所へ移動
dir_data = '/Users/fragrantflower/Desktop/carats/10-below10000ft+lat35+long139'
os.chdir(dir_data)

#処理対象ファイル一覧を取得
filelist = os.listdir(dir_data)

#ファイル一覧から不要な項目を削除
for (i, file) in enumerate(filelist):
	if file.split('.', 1)[1] <> 'csv':
		del filelist[i]

#日別に処理
for file in filelist:
	print '------'
	print file

	df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')
	
	#便ごと，時系列でソート
	df = df.sort_index(by = ['flight', 'time'])
	
	#便名を重複なしのリストにしてソート
	ls_flight = sorted(set(list(df['flight'].values.flatten())))
	
	arrdep = []
	ls_flt = []
	
	#便ごとに判定
	for i, flight in enumerate(ls_flight):

		#該当便のデータのみ抽出
		df_tmp = df[df['flight']==flight]
		#データフレームから配列に変換
		ary_data = df_tmp.as_matrix()

		#高度の列のみ抽出
		ary_height = ary_data[:,4]

		#上昇回数，下降回数カウント用
		n_up = 0
		n_down = 0
		
		#もし該当便のデータが1件ならエラーで終了
		if len(ary_data) < 2:
			ls_flt.append(flight)
			arrdep.append('error')
		#2件以上あるなら差分を計算（リストにする）
		else:
			#ary1は先頭データ削除，ary2は末尾データ削除して，ary1-ary2を計算(t+1とtの差分)
			ary1 = ary_height
			ary2 = ary_height
			ary1 = np.delete(ary1, 0, 0)
			ary2 = np.delete(ary2, len(ary2)-1, 0)
			ary_diff = ary1-ary2

			#上昇，下降をカウント
			n_up = len(np.where(ary_diff>0)[0])
			n_down = len(np.where(ary_diff<0)[0])

			#上昇の方が多ければdep,下降の方が多ければarr，同数ならerror		
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
	
	#csvに書き出し
	fn = file.split('_', 1)[0]
	df_new.to_csv('/Users/fragrantflower/Desktop/carats/20-arr+dep+error(flight)/' + fn + '_arr+dep+error(flight).csv', header=False, index=False)


#///// 出発・到着別に羽田周辺データを分ける　/////

#データ保存場所へ移動
dir_data = '/Users/fragrantflower/Desktop/carats/10-below10000ft+lat35+long139'
os.chdir(dir_data)

#処理対象ファイル一覧を取得
filelist = os.listdir(dir_data)

#ファイル一覧から不要な項目を削除
for (i, file) in enumerate(filelist):
	if file.split('.', 1)[1] <> 'csv':
		del filelist[i]

#日別に処理
for file in filelist:
	print '------'
	print file

	#発着別航跡データ格納用データフレーム
	df_arr = pd.DataFrame(columns=['time', 'flight', 'lat', 'long', 'height', 'aircraft'])
	df_dep = pd.DataFrame(columns=['time', 'flight', 'lat', 'long', 'height', 'aircraft'])
	
	#航跡データを読み込み
	df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')

	#発着便名リストを読み込み
	df_arrdep = pd.read_csv('/Users/fragrantflower/Desktop/carats/20-arr+dep+error(flight)/' + file[0:12] + 'arr+dep+error(flight).csv', header=None, names=['flight', 'arr/dep'], encoding='Shift_JIS')

	#発着便リストを発着ごとに分ける
	ary_arr = np.array(df_arrdep[df_arrdep['arr/dep'] == 'arr'].iloc[:,0])
	ary_dep = np.array(df_arrdep[df_arrdep['arr/dep'] == 'dep'].iloc[:,0])

	#航跡データの便名を重複なしのリストにしてソート
	ls_flight = sorted(set(list(df['flight'].values.flatten())))

	#発着ごとに航跡データを分ける
	for flt in ls_flight:
		df_tmp = df[df['flight']==flt]
		#便名が到着便リストにあれば到着，出発便リストにあれば出発
		if flt in ary_arr:
			df_arr = pd.concat([df_arr, df_tmp])
		elif flt in ary_dep:
			df_dep = pd.concat([df_dep, df_tmp])

	#時系列でソート
#	df_arr = df_arr.sort_index(by = ['time', 'flight'])
#	df_dep = df_dep.sort_index(by = ['time', 'flight'])

	#高度をfloat→intに型変換
	df_arr['height']= df_arr['height'].astype(int)
	df_dep['height']= df_dep['height'].astype(int)

	#csvに書き出し
	fn = file.split('_', 1)[0]
	df_arr.to_csv('/Users/fragrantflower/Desktop/carats/30-arr+dep/' + fn + '_arr.csv', header=False, index=False)
	df_dep.to_csv('/Users/fragrantflower/Desktop/carats/30-arr+dep/' + fn + '_dep.csv', header=False, index=False)


#/////　風向きを目視で判定するためプロット用の末端データを抽出　/////

#データ保存場所へ移動
dir_data = '/Users/fragrantflower/Desktop/carats/30-arr+dep'
os.chdir(dir_data)

#処理対象ファイル一覧を取得
filelist = os.listdir(dir_data)

#ファイル一覧から不要な項目を削除
for (i, file) in enumerate(filelist):
	if file.split('.', 1)[1] <> 'csv':
		del filelist[i]

#ファイル一覧をarrとdepで分ける
arrlist = filter((lambda file: file[12:15]=='arr'), filelist)
deplist = filter((lambda file: file[12:15]=='dep'), filelist)

#到着便を処理（latestのみ抽出）
for file in arrlist:
	print file

	#抽出データの格納用
	df_edge = pd.DataFrame(columns=['time', 'flight', 'lat', 'long', 'height', 'aircraft'])

	#航跡データを読み込み
	df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')

	#latestデータのみ抽出
	df_grouped = df.groupby('flight', as_index = False).agg({'time' :[np.max]})

	for i in range(len(df_grouped)):
		df_tmp1 = df[df['flight']==df_grouped.ix[i,0]]
		df_tmp2 = df_tmp1[df_tmp1['time']==df_grouped.ix[i,1]]
		df_edge = pd.concat([df_edge, df_tmp2])
	
	#csvに書き出し
	fn = file.split('_', 1)[0]
	df_edge.to_csv('/Users/fragrantflower/Desktop/carats/40-edge/' + fn + '_arr.csv', header=False, index=False)

#出発便を処理（fastestのみ抽出）
for file in deplist:
	print file

	#抽出データの格納用
	df_edge = pd.DataFrame(columns=['time', 'flight', 'lat', 'long', 'height', 'aircraft'])

	#航跡データを読み込み
	df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')

	#fasestデータのみ抽出
	df_grouped = df.groupby('flight', as_index = False).agg({'time' :[np.min]})

	for i in range(len(df_grouped)):
		df_tmp1 = df[df['flight']==df_grouped.ix[i,0]]
		df_tmp2 = df_tmp1[df_tmp1['time']==df_grouped.ix[i,1]]
		df_edge = pd.concat([df_edge, df_tmp2])
	
	#csvに書き出し
	fn = file.split('_', 1)[0]
	df_edge.to_csv('/Users/fragrantflower/Desktop/carats/40-edge/' + fn + '_fastest_dep.csv', header=False, index=False)

	
#/////　風向きを目視で判定するため末端データ座標をそのままプロット　/////

#データ保存場所へ移動
dir_data = '/Users/fragrantflower/Desktop/carats/40-edge'
os.chdir(dir_data)

#処理対象ファイル一覧を取得
filelist = os.listdir(dir_data)

#ファイル一覧から不要な項目を削除
for (i, file) in enumerate(filelist):
	if file.split('.', 1)[1] <> 'csv':
		del filelist[i]

#ファイル一覧をarrとdepで分ける
arrlist = filter((lambda file: file[12:15]=='arr'), filelist)
deplist = filter((lambda file: file[12:15]=='dep'), filelist)

#滑走路端の座標 lat, long, 16L, 34R, 16R, 34L, 04, 22, 05, 23
runway = np.array([
['16L', 35.565897, 139.786553], ['34R', 35.539694, 139.805136], ['16R', 35.559986, 139.769067], ['34L', 35.5366, 139.786559],
['04', 35.549019, 139.761278], ['22', 35.567467, 139.777114], ['05', 35.524003, 139.803464], ['23', 35.540597, 139.822114]
])

#到着便を処理
fig, axes = plt.subplots(nrows = 6, ncols = 7, figsize = (20, 12))
k = 0;
for i in range(6):
	for j in range(7):
		lat = []
		long = []
		height = []
			
		#航跡データを読み込み
		file = arrlist[k]
		df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')

		#緯度，経度，高度
		lat.extend(df['lat'].values.flatten())
		long.extend(df['long'].values.flatten())
		height.extend(df['height'].values.flatten())
	
		axes[i,j].scatter(long, lat, color = 'r', edgecolors = 'k', s=10)
		axes[i,j].set_title('arr at ' + file[3:11])
		
		print file
		k += 1

#滑走路をプロット，値域をそろえる
for ax in axes.flatten():
	#滑走路A
		ax.plot(runway[:,2][0:2], runway[:,1][0:2], color = 'k', linewidth = 3, alpha = 0.6)
	#滑走路B
		ax.plot(runway[:,2][2:4], runway[:,1][2:4], color = 'k', linewidth = 3, alpha = 0.6)
	#滑走路C
		ax.plot(runway[:,2][4:6], runway[:,1][4:6], color = 'k', linewidth = 3, alpha = 0.6)
	#滑走路D
		ax.plot(runway[:,2][6:8], runway[:,1][6:8], color = 'k', linewidth = 3, alpha = 0.6)
		
		ax.set_xlim([139.74, 139.84])
		ax.set_ylim([35.48, 35.6])
		ax.grid(True)

plt.tight_layout()
plt.show()

#pngファイルに書き出し
fig.savefig('/Users/fragrantflower/Desktop/carats/40-edge/' + fn + '_latest_arr.png', dpi=90)
plt.close(fig)

#出発便を処理
fig, axes = plt.subplots(nrows = 6, ncols = 7, figsize = (20, 12))
k = 0;
for i in range(6):
	for j in range(7):
		lat = []
		long = []
		height = []
			
		#航跡データを読み込み
		file = deplist[k]
		df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')

		#緯度，経度，高度
		lat.extend(df['lat'].values.flatten())
		long.extend(df['long'].values.flatten())
		height.extend(df['height'].values.flatten())
	
		axes[i,j].scatter(long, lat, color = 'b', edgecolors='k', s=10)
		axes[i,j].set_title('dep at ' + file[3:11])
		
		print file
		k += 1

#滑走路をプロット，値域をそろえる
for ax in axes.flatten():
	#滑走路A
		ax.plot(runway[:,2][0:2], runway[:,1][0:2], color = 'k', linewidth = 3, alpha = 0.6)
	#滑走路B
		ax.plot(runway[:,2][2:4], runway[:,1][2:4], color = 'k', linewidth = 3, alpha = 0.6)
	#滑走路C
		ax.plot(runway[:,2][4:6], runway[:,1][4:6], color = 'k', linewidth = 3, alpha = 0.6)
	#滑走路D
		ax.plot(runway[:,2][6:8], runway[:,1][6:8], color = 'k', linewidth = 3, alpha = 0.6)
		
		ax.set_xlim([139.74, 139.88])
		ax.set_ylim([35.45, 35.63])
		ax.grid(True)

plt.tight_layout()
plt.show()

#pngファイルに書き出し（印刷用にはdpi350～400）
fig.savefig('/Users/fragrantflower/Desktop/carats/40-edge/' + fn + '_dep.png', dpi=90)
plt.close(fig)


#/////　対象日のedgeデータを風向き別×発着別に統合（利用滑走路の判定用）　/////

#データ保存場所へ移動
dir_data = '/Users/fragrantflower/Desktop/carats/50-targetday'
os.chdir(dir_data)

#北風，南風ごとに到着，出発便データを一つのファイルに統合
ary = ['n', 's']

for x in ary:

	#処理対象ファイル一覧を取得
	os.chdir(dir_data + '/' + x)
	filelist = os.listdir(dir_data + '/' + x)

	#ファイル一覧から不要な項目を削除
	for (i, file) in enumerate(filelist):
		if file.split('.', 1)[1] <> 'csv':
			del filelist[i]

	#データ蓄積用
	df_arr = pd.DataFrame(columns=['time', 'flight', 'lat', 'long', 'height', 'aircraft'])
	df_dep = pd.DataFrame(columns=['time', 'flight', 'lat', 'long', 'height', 'aircraft'])
	
	for file in filelist:
		#edgeデータの読み込み
		df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')
		
		if file[12:15] == 'arr':
			df_arr = pd.concat([df_arr, df])
		elif file[12:15] == 'dep':
			df_dep = pd.concat([df_dep, df])
			
	#csvに書き出し
	fn = file.split('_', 1)[0]
	df_arr.to_csv('/Users/fragrantflower/Desktop/carats/50-targetday/' + x + '_targets_arr_latest.csv', header=False, index=False)
	df_dep.to_csv('/Users/fragrantflower/Desktop/carats/50-targetday/' + x + '_targets_dep_fastest.csv', header=False, index=False)


#/////　対象日のedgeデータを風向き別×発着別にプロット（利用滑走路の判定用）　/////

#データ保存場所へ移動
dir_data = '/Users/fragrantflower/Desktop/carats/50-targetday/targets_edge'
os.chdir(dir_data)

#処理対象ファイル一覧を取得
filelist = os.listdir(dir_data)

#ファイル一覧から不要な項目を削除
for (i, file) in enumerate(filelist):
	if file.split('.', 1)[1] <> 'csv':
		del filelist[i]

#滑走路端の座標 lat, long, 16L, 34R, 16R, 34L, 04, 22, 05, 23
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
		#対象日のedge統合データを読み込み
		df = pd.read_csv(file, header=None, names=['time', 'flight', 'lat', 'long', 'height', 'aircraft'], encoding='Shift_JIS')
		
		#緯度，経度
		lat.extend(df['lat'].values.flatten())
		long.extend(df['long'].values.flatten())
		
		axes[i, j].scatter(long, lat)
		axes[i, j].set_title(file)
		
		print file
		k += 1

#滑走路をプロット
for ax in axes.flatten():
	#滑走路A
		ax.plot(runway[:,2][0:2], runway[:,1][0:2], color = 'k', linewidth = 3, alpha = 0.6)
	#滑走路B
		ax.plot(runway[:,2][2:4], runway[:,1][2:4], color = 'k', linewidth = 3, alpha = 0.6)
	#滑走路C
		ax.plot(runway[:,2][4:6], runway[:,1][4:6], color = 'k', linewidth = 3, alpha = 0.6)
	#滑走路D
		ax.plot(runway[:,2][6:8], runway[:,1][6:8], color = 'k', linewidth = 3, alpha = 0.6)

#各グラフのプロット範囲とスケールの調整
#北風×到着
axes[0,0].set_xlim([139.77, 139.82])
axes[0,0].set_ylim([35.514, 35.5475])
axes[0,0].grid(True)

#北風×出発
axes[0,1].set_xlim([139.75, 139.86])
axes[0,1].set_ylim([35.51, 35.6])
axes[0,1].grid(True)

#南風×到着
axes[1,0].set_xlim([139.75, 139.83])
axes[1,0].set_ylim([35.52, 35.58])
axes[1,0].grid(True)

#南風×出発
axes[1,1].set_xlim([139.75, 139.86])
axes[1,1].set_ylim([35.476, 35.557])
axes[1,1].grid(True)

plt.tight_layout()
plt.show()

#pngファイルに書き出し（印刷用にはdpi350～400）
#fig.savefig('/Users/fragrantflower/Desktop/carats/40-edge/' + fn + '_dep.png', dpi=90)
#plt.close(fig)

