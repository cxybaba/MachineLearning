from sklearn import linear_model
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import random
import copy

data = pd.read_csv('503afterwash2.csv',encoding='utf-8',index_col='YMD')
data.index = pd.to_datetime(data.index)
datanp = np.array(data.as_matrix(columns = ['BOD5_ET','COD_ET','TN_ET', 'TP_ET','NH4_N_ET']))[367:587]

datain,dataout = np.split(datanp,(4,), axis=1)#分割输入输出
clf = svm.SVR(kernel='linear', C=0.1,epsilon=0.01)#0.1 0.1 0.1就超越了贝叶斯 C影响较大，C越大运行越慢，epsilon越大运行越快 都在(0,0.3]之间取值
#返回五次迭代的平均分 通过决定系数的平均分来评定模型的好坏
def kfoldtest(clf):
	r2s = []
	corrs = []
	kf=KFold(5,shuffle=False,random_state=0)
	x = datain
	y = dataout
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
	# print('r2_array:     ',r2s)
	# print('corrs_array:   ',corrs)
	return np.array(r2s),np.array(corrs)
def getsvr(list):
	return svm.SVR(kernel='linear', C=list[0],epsilon=list[1])
print(kfoldtest(getsvr([0.025413940665258837,0.10838556640556057])))

#***************************LPSO*************************************
birds=8#10只鸟
xcount=2#2个维度
pos=[]#坐标[[],[],[],[],[]...........] 每只鸟的位置
speed=[]#速度[[],[],[],[],[]..........] 每只鸟的速度
bestpos=[]#这只鸟最好的坐标。[[],[],[],[]......]每一个元素装着对应鸟自己的最好的位置
birdsbestpos=[]#所有鸟目前最好的坐标。[]这个list装着所有鸟最好的位置
w=0.8#权重
c1=2 #学习因子，通常为2
c2=2#学习因子
r1=0.6#通常为0到1的随机数
r2=0.3
for i in range(birds):
    pos.append([])
    speed.append([])
    bestpos.append([])

def GenerateRandVec(list):
    for i in range(xcount):
        list.append(random.uniform(0,5))

for i in range(birds):          #initial all birds' pos,speed
    GenerateRandVec(pos[i])
    GenerateRandVec(speed[i])
    bestpos[i]=copy.deepcopy(pos[i])
        
def FindBirdsMostPos():         #找所有鸟的最佳位置，这是下一次迭代的标准
    best=kfoldtest(getsvr((bestpos[0])))
    index=0
    for i in range(birds):
        temp=kfoldtest(getsvr((bestpos[i])))
        if temp>best:
            best=temp
            index=i
    return bestpos[index]

birdsbestpos=FindBirdsMostPos()   #initial birdsbestpos

def NumMulVec(num,list):         # 常数乘以向量
    for i in range(len(list)):
        list[i]*=num             #（vi1,vi2,vi1,vi2,vi1)→w*(vi1,vi2,vi1,vi2,vi1)

    return list

def VecSubVec(list1,list2):       #向量减去向量
    for i in range(len(list1)):
        list1[i]-=list2[i]
    return list1

def VecAddVec(list1,list2):      #向量加上向量
    for i in range(len(list1)):
        list1[i]+=list2[i]
    return list1

def VecAddVec1(list1,list2):      #向量加上向量 更新位置专用
	list3 = list1.copy()
	for i in range(len(list1)):
		list1[i]+=list2[i]
	if list1[0]>0 and list1[1]>0: #and list1[0]<0.4 and list1[1]<0.4:
		return list1
	list1 = list3.copy()
	return list1

def UpdateSpeed():
    #global speed
    for i in range(birds):
        temp1=NumMulVec(w,speed[i][:])
        temp2=VecSubVec(bestpos[i][:],pos[i])
        temp2=NumMulVec(c1*r1,temp2[:])
        temp1=VecAddVec(temp1[:],temp2)
        temp2=VecSubVec(birdsbestpos[:],pos[i])
        temp2=NumMulVec(c2*r2,temp2[:])
        speed[i]=VecAddVec(temp1,temp2)
        
def UpdatePos(n):
    global bestpos,birdsbestpos
    for i in range(birds):
        pos[i] = VecAddVec1(pos[i],speed[i])
        # print(pos[i])
        #如果更新的位置比原来的好，就替换，更新局部最优
        if  kfoldtest(getsvr((pos[i])))>kfoldtest(getsvr((bestpos[i]))):
            bestpos[i]=copy.deepcopy(pos[i])
    #及时跟新全局最优
    birdsbestpos=FindBirdsMostPos()
    w = 0.8+0.5*((1-(n/100)**2)**5)
    
# for i in range(50):
#     #print birdsbestpos
#     print(i,birdsbestpos,kfoldtest(getsvr(birdsbestpos)))
#     UpdateSpeed()
#     UpdatePos(i)
                
