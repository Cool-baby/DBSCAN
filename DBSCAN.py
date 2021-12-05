import random
from sklearn import datasets
from matplotlib import pyplot
import numpy
import math


'''
    loadDataSet(fileName, splitChar='\t')
    数据集加载函数，参数fileN--数据集文件名，splitChar--分隔符（此处给出默认分隔符为“tab”，本数据集所用分隔符为“,"）
'''
def loadDataSet(fileName, splitChar='\t'):
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return dataSet


'''
    distance(data1,data2)
    data1和data2分别是X(x1,x2)，Y(y1,y2)两点的坐标值
    欧氏距离计算函数，distance = [(x1-y1)^2+(x2-y2)^2]^(1/2)
'''
def distance(data1,data2):
    dis = math.sqrt((numpy.power(data1[0] - data2[0],2) + numpy.power(data1[1] - data2[1],2)))
    return dis


'''
    dbscan(data,e,minpts)
    核心函数，参数data--数据集，e--半径参数，minpts--密度阈值
'''
def dbscan(data,e,minpts):
    num = len(data)                                 #数据集中点的个数
    unvisited = [i for i in range(num)]             #没有访问到点的列表
    visited = []                                    #已访问点的列表
    C = [-1 for i in range(num)]                    #C为output，初始值全是-1，代表着没分类，数量是num个
    k = -1                                          #k标记不同簇，-1代表着噪声点

    while len(unvisited) > 0:                       #未访问节点数>0，代表还未完全访问数据集
        p = random.choice(unvisited)                #随机在未访问数据集中选取一点p
        unvisited.remove(p)                         #将p从未访问列表中去除
        visited.append(p)                           #将p加入已访问列表

        N = []                                      #N为p点距离小于e邻域内所有对象的集合
        for i in range(num):                        #在数据集所有样本中循环
            if(distance(data[i],data[p]) <= e):     #当某样本i距离核心对象距离小于e时
                N.append(i)                         #将i样本加入p核心对象集合中

        if len(N) >= minpts:                        #当核心对象p邻域中样本个数不小于minpts时，才认为p是核心对象，才进行下面操作
            k = k+1                                 #划分新簇
            C[p] = k                                #将划分新簇的代号替换掉output集合中初始值-1

            for N_point in N:                       #N_ponit代表着以p为核心对象的距离小于e的样本
                if N_point in unvisited:            #此样本必须是未被标记过的（先访问先标记，此处是DBSCAN不稳定的直接原因）
                    unvisited.remove(N_point)       #将N_ponit从未访问列表中移除
                    visited.append(N_point)         #将N_ponit加入到已访问列表

                    M = []                          #M是位于N_ponit邻域中样本点的列表
                    for j in range(num):            #在数据集所有样本中循环
                        if (distance(data[j],data[N_point]) <= e):      #当某样本j距离核心对象N_ponit距离小于e时
                            M.append(j)             #将j样本加入N_ponit核心对象集合中

                    if len(M) >= minpts:            #当核心对象N_ponit邻域中样本个数大于minpts时，才认为N_ponit是核心对象，才进行下面操作
                        for M_point in M:           #M_ponit代表着以N_ponit为核心对象的距离小于e的样本
                            if M_point not in N:    #M_ponit这个点不在以p为核心对象的列表N中（若在N中说明已将此点放入以p为核心对象的列表中）
                                N.append(M_point)   #将此点放入N中

                if C[N_point] == -1:                #此处如果N_point从未被划分的某个簇，即簇数为默认值-1，才将此点标记为k簇
                    C[N_point] = k                  #将N_point标记为k簇
        else:
            C[p] = -1                               #当核心对象p邻域中样本个数小于minpts时，不满足定义的核心样本邻域的阈值，不划分簇

    return C                                        #返回output


dataset = loadDataSet('DBSCANpoints.txt', splitChar=',')     #加载数据集DBSCANpoints,数据集少，很快出结果
#dataset = loadDataSet('mydata.csv', splitChar=',')            #加载数据集mydata，数据集大，运算时间较长，具体看性能
C = dbscan(dataset,2,10)                                      #调用dbscan函数，通过不断调整参数，来确定最优参数


'''
    可视化过程，利用python的pyplot
'''
x = []
y = []
for data in dataset:                                        #将数据集中所有的样本的x轴y轴加入到列表x和y中
    x.append(data[0])
    y.append(data[1])
pyplot.figure(figsize=(8, 6), dpi=480)                      #figure函数，参数figsize代表画布宽高（英寸），dpi代表分辨率
pyplot.scatter(x,y,c=C,marker='o')                          #scatter函数，参数x,y为输入数据，c为颜色序列（大C为咱们标记的不同的簇，故不同簇颜色不同），marker为标记（'o'为圆圈）
pyplot.show()