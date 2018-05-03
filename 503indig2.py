#['BOD5_ET', 'COD_ET', 'TN_ET', 'TP_ET', 'PH_ET', 'NH4_N_ET']
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('503afterwash2.csv',encoding='utf-8',index_col='YMD')#变成了dataframehahaxixi
data.index = pd.to_datetime(data.index)#这一列作为时间序列
#NH$从367开始有值
datanp = np.array(data)[367:587]
# print(datanp)
# datain,dataout = np.split(datanp,(6,), axis=1)#分割输入输出
# print(datain)
# print(dataout)
#ndarray变成list
# datainlist = datain.tolist()
# dataoutlist = dataout.tolist()
# print(datainlist)
# print(dataoutlist)
#将有值的部分重新组成datafram
data2 = pd.DataFrame(datanp,columns=['BOD5_ET', 'COD_ET', 'TN_ET', 'TP_ET', 'PH_ET', 'NH4_N_ET'])
print(data2.tail())

#***********************维数选择****************************     填补NH4的结果不错
#相关性检验 结合相关性检验选择 DLQLTY   BOD5_ET TN_ET     TP_ET  作为输入
print(data2.corr(method='pearson')['NH4_N_ET']) 
#pca检验需要的维数 效果不明显
# pca = PCA(copy=True, iterated_power='auto', n_components=6, random_state=None,
#   svd_solver='randomized', tol=0.0, whiten=False)
# a,b = np.split(datanp,(6,), axis=1)
# print(a.tolist()) 把输入独立出来
# pca.fit(a)
# print(pca.explained_variance_ratio_) #输出主成分比例,也就是一维的时候，二维的时候，三维的时候
def har():
	index = ['NH4_N_ET','TN_ET','TP_ET','COD_ET','BOD5_ET','']
	data = np.array(data2.corr(method='pearson')['NH4_N_ET']).tolist()
	# print(sorted(data))
	plt.barh(bottom=(0,1,2,3,4,5),height=0.35,width=sorted(data)[::-1],align="center")  
	plt.yticks((0,1,2,3,4,5),index)  
	ax = plt.gca()
	ax.spines['right'].set_color('none')#ax的spines指的就是图形的四个边框，脊梁
	ax.spines['top'].set_color('none')
	ax.yaxis.set_ticks_position('left')#将左边的边框作为y轴
	ax.spines['left'].set_position(('data',0))#挪动边框 在哪个点落位，y轴在x等于-1落位
	plt.text(0.08, 4.85, 'PH_ET', ha='center', va= 'bottom',fontsize=10)
	plt.plot([0.3,0.3,0.3,0.3,0.3,0.3,0.3],[-1,0,1,2,3,4,5],color='red', linewidth=1.0, linestyle='--')
	plt.text(0.32, 4.5, 'corr=0.3', ha='left', va= 'bottom',fontsize=10,color='red')
	plt.show()
har()
datanp2 = np.array(data2.as_matrix(columns = ['BOD5_ET','COD_ET','TN_ET', 'TP_ET','NH4_N_ET']))

datain,dataout = np.split(datanp2,(4,), axis=1)#分割输入输出
#输入输出的list
k=220  #选择220条数据效果最好
datain1 = datain[:k]
dataout1 = dataout[:k]


clf = linear_model.BayesianRidge(alpha_1=1000, alpha_2=1000, compute_score=False,
        copy_X=True, fit_intercept=True, lambda_1=1000, lambda_2=1000,
        n_iter=3000, normalize=False, tol=1000, verbose=False)
clf1 = svm.SVR(kernel='linear', C=1,epsilon=5)#0.1 0.1 0.1就超越了贝叶斯 C影响较大，C越大运行越慢，epsilon越大运行越快 通过pso去优化C和epsilon的值 C和epsilon取在（0，5]优化
clf2 = DecisionTreeRegressor(max_depth=4)
def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)
def testclf(clf):
	cv = ShuffleSplit(n_splits=5, train_size=0.8, random_state=0)#train_size表示的是整个数据数据集选多少train来训练
	scores = cross_val_score(clf, datain1, dataout1, cv=cv)
	print(scores)
	clf.fit(datain1,dataout1)
	predicted = clf.predict(datain[:k])
	#print(r2_score(predicted,dataout[:k]))
#检查预测结果和实际结果的相关性
	b = []
	for n in dataout[:k]:
		b.append(n[0])
	d = {
       	'predict':predicted.tolist(),
       	'true':b
		}
	datacor = pd.DataFrame(d)
	print(datacor.corr(method='pearson'))
	#print(MAPE(b,predicted.tolist()))
# print('beiyesi:','*****************')
# testclf(clf)
# print('jueceshu:','*****************')
# testclf(clf2)
clf2 = DecisionTreeRegressor(max_depth=3)
# print('svr:','*****************')
# testclf(clf1)
def kfoldtest(clf):
	score = []
	r2s = []
	corrs = []
	kf=KFold(5,shuffle=False,random_state=0)
	x = datain[:k]
	y = dataout[:k]
	for train_index, test_index in kf.split(x,y):
		X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
		clf.fit(X_train,y_train)
		predicted = clf.predict(X_test)
		r2 = round(r2_score(predicted,y_test),3)
		r2s.append(r2)
		# print('r2_score2:',round(r2_score(predicted,y_test),3))
		b = []
		for i in y_test.tolist():
			b.append(i[0])
		d = {
       		'predict':predicted.tolist(),
       		'true':b
		}
		datacor = pd.DataFrame(d)
		corr = round(datacor.corr(method='pearson')['predict'][1],5)
		corrs.append(corr)
	score.append(r2s)
	score.append(corrs)
	return score
		# print('coor:',round(datacor.corr(method='pearson')['predict'][1],3))
print('svrkfold:','*****************')
print(np.array(kfoldtest(clf1)[0]),np.array(kfoldtest(clf1)[1]))

print('beiyesi:','*****************')
print(np.array(kfoldtest(clf)[0]).mean(),np.array(kfoldtest(clf)[1]).mean())
print('jueceshu:','*****************')
print(np.array(kfoldtest(clf2)[0]).mean(),np.array(kfoldtest(clf2)[1]).mean())

 

# print('svr',20*'*')
# clf = svm.SVR(kernel='rbf', C=1000)
# cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# scores = cross_val_score(clf, datain1, dataout1, cv=cv)
# print(scores)
# clf.fit(datain1,dataout1)
# predicted = clf.predict(datain[k:])
# # print(predicted)
# print(r2_score(predicted,dataout[k:]))

#**********************聚类分析*******************
# datanp3 = np.array(data2.as_matrix(columns = ['BOD5_ET', 'COD_ET', 'TN_ET', 'TP_ET', 'PH_ET']))
# # print(datanp3.tolist())
# #Kmeans
# clf = KMeans(n_clusters=8,algorithm='full',max_iter=1000)
# km = clf.fit(datanp3.tolist())
# print(km.labels_)
# #评价聚类模型 分成四维
# a,b= np.split(datanp3,(258,), axis=0)#分割输入输出
# def findharabaz_score():
# 	harabaz_score = []
# 	k = [x for x in range(4,121)]
# 	for x in k:
# 		clf = KMeans(n_clusters=x,algorithm='full',max_iter=1000)
# 		km = clf.fit(a)
# 		harabaz_score.append(metrics.calinski_harabaz_score(a,km.labels_))
# 	plt.figure()
# 	plt.plot(k,harabaz_score)#往figure里面装数据
# 	plt.show()
# # findharabaz_score() #分成8维

 


