#['BOD5_ET', 'COD_ET', 'TN_ET', 'TP_ET', 'PH_ET', 'NH4_N_ET']
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.dates as mdates
#z分数的计算
def zscore(std,mean,data):
	zscore = abs(data - mean)/std
	return np.float64(zscore).item()
#时间序列
def drawtime(data,name):
	list = []
	for i in range(len(data)):
		list.append(3)
	ax = plt.gca()
	xfmt = mdates.DateFormatter('%y-%m-%d')
	ax.xaxis.set_major_formatter(xfmt)
	plt.plot(data.index,data[name])
	plt.plot(data.index,list,color='red', linewidth=1.0, linestyle='--')
	plt.xlim(('2008-3-1','2010-4-1')) 
	plt.ylabel(name,fontproperties='SimHei')
	plt.show()

data = pd.read_csv('../503input.csv',encoding='utf-8',index_col='YMD')#变成了dataframehahaxixi
data.index = pd.to_datetime(data.index)#这一列作为时间序列
dataout = data.copy()

#计算每个元素的z分数
def calculatezscore(dataout,data):
	k=0
	for i in dataout.columns:
		std = dataout[i].std()
		mean = dataout[i].mean()
		for j in range(len(dataout)):
			if np.isnan(dataout[i][j]):
				data[i][j] = -1
			else:
				data[i][j] = round(zscore(std,mean,dataout[i][j]),3)
		k = k+1
#检查z分数
def checkzscore(data):
	list = []
	for i in data.columns:
		print(data[i][data[i]>3])
		list.append(len(data[i][data[i]>3]))
	plt.bar(data.columns, list)
	plt.xlabel(u"The water quality parameters")
	plt.ylabel(u"Z-score abnormal statistics")
	for a,b in zip(data.columns,list):
		plt.text(a, b+0.01, '%.0f' % b, ha='center', va= 'bottom',fontsize=12)
	plt.show()
calculatezscore(dataout,data)
# print(data.head())
# print(data.tail())
#查看z分数超标的数据
print(20*'*','chakanzfenshuchaobiaodeshuju1')
checkzscore(data)
drawtime(data,'TP_ET')
#查看这个时间段的tp的数据
# print(data.loc['2010-03-01':'2010-03-31','TP_ET'])
#经过分析将这个时间段的数据全部除以10
dataout.loc['2010-03-01':'2010-03-31','TP_ET'] = dataout.loc['2010-03-01':'2010-03-31','TP_ET']/10
#print(dataout.loc['2010-03-01':'2010-03-31','TP_ET'])

#清洗后再次求z分数
calculatezscore(dataout,data)
#查看z分数超标的数据
print(20*'*','chakanzfenshuchaobiaodeshuju2')
checkzscore(data)
#拿到z分数的list
#print(data['DLQLTY'])
#print(datanp)
#前后三天的统计学均值填补
def changedata(dataout,data):
	datanp = pd.DataFrame(np.array(data))
	k = 0
	for i in dataout.columns:
		list = []
		scoreindex = datanp[k][datanp[k]>3].index.tolist()
		#print(scoreindex)
		for j in scoreindex:
			for n in range(1,4):
				if j+n>704:
					break
				if data[i][j+n]!=-1:
					list.append(dataout[i][j+n])
			for n in range(1,4):
				if j-n<0:
					break
				if data[i][j-n]!=-1:
					list.append(dataout[i][j-n])
			dataout[i][j] = round(np.mean(np.array(list)),4)
		k = k+1
#***************第一次优化z分数**********************
print(20*'*','changedata1')
changedata(dataout,data)
calculatezscore(dataout,data)
checkzscore(data)
print(20*'*','changedata2')
changedata(dataout,data)
calculatezscore(dataout,data)
checkzscore(data)
#时间序列散点图
dataout.to_csv('503afterwash2.csv')
def drawtime1(data,name):
	list = []
	for i in range(len(data)):
		list.append(3)
	ax = plt.gca()
	xfmt = mdates.DateFormatter('%y-%m-%d')
	ax.xaxis.set_major_formatter(xfmt)
	plt.scatter(data.index,data[name])
	# plt.plot(data.index,list,color='red', linewidth=1.0, linestyle='--')
	plt.xlim(('2008-3-1','2010-4-1')) 
	plt.ylabel(name,fontproperties='SimHei')
	plt.show()
drawtime1(dataout,'NH4_N_ET')
#柱状图
def drawbar(data):
	list = []
	for i in data.columns:
		list.append(len(data[i][data[i]>0]))
	plt.bar(data.columns, list)
	plt.xlabel(u"The water quality parameters")
	plt.ylabel(u"The amount of data")
	for a,b in zip(data.columns,list):
		plt.text(a, b+0.01, '%.0f' % b, ha='center', va= 'bottom',fontsize=12)
	plt.show()
drawbar(data)





