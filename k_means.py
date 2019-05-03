import numpy as np
import pandas as pd


# K-均值聚类辅助函数

# 文本数据解析函数
def fileInput(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将每一行的数据映射成float型
        fltLine = map(int, curLine)
        dataMat.append(fltLine)
    return dataMat


# 数据向量计算欧式距离
def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA.astype(int)- vecB.astype(int), 2)))


# 随机初始化K个质心(质心满足数据边界之内)
def randCent(dataSet, k):
    # 得到数据样本的维度
    n = np.shape(dataSet)[1]
    # 初始化为一个(k,n)的矩阵
    centroids = np.mat(np.zeros((k, n)))
    # 遍历数据集的每一维度
    for j in range(n):
        # 得到该列数据的最小值
        minJ = min(dataSet[:, j])
        # 得到该列数据的范围(最大值-最小值)
        rangeJ = float(max(dataSet[:, j]) - minJ)
        # k个质心向量的第j维数据值随机为位于(最小值，最大值)内的某一值
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    # 返回初始化得到的k个质心向量
    print('centroids',centroids)
    return centroids


# k-均值聚类算法
# @dataSet:聚类数据集
# @k:用户指定的k个类
# @distMeas:距离计算方法，默认欧氏距离distEclud()
# @createCent:获得k个质心的方法，默认随机获取randCent()
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    # 获取数据集样本数
    m = np.shape(dataSet)[0]
    print('m',m)
    # 初始化一个(m,2)的矩阵
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 创建初始的k个质心向量
    centroids = createCent(dataSet, k)
    # 聚类结果是否发生变化的布尔类型
    clusterChanged = True
    flg = 0
    # 只要聚类结果一直发生变化，就一直执行聚类算法，直至所有数据点聚类结果不变化
    while clusterChanged:
        # 聚类结果变化布尔类型置为false
        flg +=1
        clusterChanged = False
        # 遍历数据集每一个样本向量
        for i in range(m):
            # 初始化最小距离最正无穷；最小距离对应索引为-1
            minDist = np.inf;
            minIndex = -1
            # 循环k个类的质心
            for j in range(k):
                # 计算数据点到质心的欧氏距离
                distJI = distMeas(np.array(centroids[j, :])[0], dataSet[i, :])
                # 如果距离小于当前最小距离
                if distJI < minDist:
                    # 当前距离定为当前最小距离；最小距离对应索引对应为j(第j个类)
                    minDist = distJI;
                    minIndex = j
        # 当前聚类结果中第i个样本的聚类结果发生变化：布尔类型置为true，继续聚类算法
        if clusterAssment[i, 0] != minIndex:
            clusterChanged = True
        # 更新当前变化样本的聚类结果和平方误差
        clusterAssment[i, :] = minIndex, minDist ** 2
    # 打印k-均值聚类的质心
    # print(centroids)
    # 遍历每一个质心
    for cent in range(k):
        # 将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
        ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
        # print(ptsInClust)
        # 计算这些数据的均值（axis=0：求列的均值），作为该类质心向量
        centroids[cent, :] = np.mean(ptsInClust, axis=0)
    # 返回k个聚类，聚类结果及误差
    return centroids, clusterAssment

#二分k_means
def biKmeans(dataSet,k,distMeas=distEclud):
    #获得数据集的样本数
    m=np.shape(dataSet)[0]
    #初始化一个元素均值0的(m,2)矩阵
    clusterAssment=np.mat(np.zeros((m,2)))
    #获取数据集每一列数据的均值，组成一个长为列数的列表
    centroid0=np.mean(dataSet,axis=0).tolist()
    #当前聚类列表为将数据集聚为一类
    centList=[centroid0]
    print('len',len(centList))
    #遍历每个数据集样本
    # print("centroid0"+str(centroid0))
    # print("typecentroid0" + str(type(centroid0)))
    for j in range(m):
        # print("centroid0" + str(dataSet[j,:]))
        # print("typecentroid0" + str(type(dataSet[j,:])))
        #计算当前聚为一类时各个数据点距离质心的平方距离
        clusterAssment[j,1]=distMeas(np.array(centroid0),dataSet[j,:])** 2
    #循环，直至二分k-均值达到k类为止
    while (len(centList)<k):
        #将当前最小平方误差置为正无穷
        lowerSSE=np.inf
        #遍历当前每个聚类
        for i in range(len(centList)):
            #通过数组过滤筛选出属于第i类的数据集合
            ptsInCurrCluster=\
                dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]
            #对该类利用二分k-均值算法进行划分，返回划分后结果，及误差
            centroidMat,splitClustAss=\
                kMeans(ptsInCurrCluster,2,distMeas)
            #计算该类划分后两个类的误差平方和
            sseSplit=sum(splitClustAss[:,1])
            #计算数据集中不属于该类的数据的误差平方和
            sseNotSplit=\
                sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            #打印这两项误差值
            print('sseSplit'+str(sseSplit)+' notSplit:'+str(sseNotSplit))
            #划分第i类后总误差小于当前最小总误差
            if(sseSplit+sseNotSplit)<lowerSSE:
                #第i类作为本次划分类
                bestCentToSplit=i
                #第i类划分后得到的两个质心向量
                bestNewCents=centroidMat
                #复制第i类中数据点的聚类结果即误差值
                bestClustAss=splitClustAss.copy()
                #将划分第i类后的总误差作为当前最小误差
                lowerSSE=sseSplit+sseNotSplit
        #数组过滤筛选出本次2-均值聚类划分后类编号为1数据点，将这些数据点类编号变为
        #当前类个数+1，作为新的一个聚类
        bestClustAss[np.nonzero(bestClustAss[:,0].A==1)[0],0]=\
                len(centList)
        #同理，将划分数据集中类编号为0的数据点的类编号仍置为被划分的类编号，使类编号
        #连续不出现空缺
        bestClustAss[np.nonzero(bestClustAss[:,0].A==0)[0],0]=\
                bestCentToSplit
        #打印本次执行2-均值聚类算法的类
        print('the bestCentToSplit is:'+str(bestCentToSplit))
        #打印被划分的类的数据个数
        print('the len of bestClustAss is:'+str(len(bestClustAss)))
        #更新质心列表中的变化后的质心向量
        centList[bestCentToSplit]=bestNewCents[0,:]
        #添加新的类的质心向量
        centList.append(bestNewCents[1,:])
        #更新clusterAssment列表中参与2-均值聚类数据点变化后的分类编号，及数据该类的误差平方
        clusterAssment[np.nonzero(clusterAssment[:,0].A==\
                bestCentToSplit)[0],:]=bestClustAss
        #返回聚类结果
        return centList,clusterAssment


#
# label_size = 10
#
# class Data:
#     __slots__ = ["data", "group"]
#     def __init__(self, x=[0]*28*28, group=0):
#         self.x, self.group = x, group
#
# def binaryzation(data):
#     ret = np.empty((data.shape[0]))
#     for i in range(data.shape[0]):
#         ret[i] = 0
#         if (data[i] > 127):
#             ret[i] = 1
#     return ret
#
# def distance(center, data):
#     dist = data - center
#
#     return dist ** 2
#
# def load_data(dataset='training'):
#     filepath = './samples/labels/train_label_fix.csv'
#     datas = pd.read_csv(filepath).values
#     datas, labels = np.concatenate((datas[1:, 1:3],datas[1:, 5:6]),axis=1), datas[1:, -1]
#     idx = np.random.permutation(datas.shape[0])
#     fileInd = []
#     for i in range(datas.shape[0]):
#         fileInd.append(idx[i])
#
#     data = np.empty((datas.shape[0], 4), dtype="uint8")
#     label = np.empty((datas.shape[0]), dtype="uint8")
#     print("loading data...")
#     for c in range(datas.shape[0]):
#         tmpInd = fileInd[c]
#         data[c] = binaryzation(datas[tmpInd])
#         label[c] = labels[tmpInd]
#
#     return data, label
#
# def Bin_k_means(data, k):
#     Orig_cluster_centers = np.mean(data, axis=1)
#     dataD ={}
#     dataD['0'] = {'center' : Orig_cluster_centers, 'data': data}
#
#     while len(dataD.keys)<k:
#         sse =[]
#         for c in range(0, len(dataD.keys)):
#             dt, sseDt = k_means(dataD[c]['data'], 2)
#             sse.append(dt)
#         ##TO DO
#
#
# def k_means(data_img, k, count_SSE =True):
#     data = {}
#     centerlist = []
#     for c in range(0,k):
#         data[str(c)] = {'center' : data_img[np.random.rand(data_img.shape[0])],'data':[]}
#         centerlist.append(data[str(c)]['center'])
#     update_flg = 1
#     while update_flg !=0:
#         update_flg = 0
#
#         for c in range(0, k):
#             data[str(c)]['data'] = []
#
#         for c in data_img:
#             dif_mat = np.tile(c, (k,1)) - centerlist
#             sqr_dif_mat = dif_mat**2
#             sqr_dis = sqr_dif_mat.sum(axis=1)
#
#             centerNum = np.argmin(sqr_dis)
#             data[str(centerNum)]['data'].append(c)
#
#
#
#         for ind, c in enumerate(data):
#             center_update = np.mean(c['data'], axis=1)
#             if center_update != c['center']:
#                 update_flg += 1
#                 c['center'] = center_update
#                 centerlist[ind] = center_update
#
#
#     if count_SSE:
#         sseValue = 0
#         for c in data:
#             sqr_dif_mat = (np.tile(c, (c['data'].shape[0],1)) - c['data'])**2
#             sqr_dis = sqr_dif_mat.sum(axis=1)
#             sseValue += sqr_dis
#
#         return data, sseValue
#
#
#     return data




if __name__ == '__main__':
    # train_img, train_lbl = load_data(dataset='training')
    # test_img, test_lbl = load_data(dataset='testing')
    # kmeans_clu = k_means(train_img, label_size)
    filepath = './samples/labels/train_label_fix.csv'
    datas = pd.read_csv(filepath).values
    print(datas.shape[0])
    dataT, labels = (datas[0:, 5:7]-datas[0:, 1:3]), datas[0:, -1]

    print(kMeans(np.array(dataT), 1)[0])
    # [[52.17904038 51.20594333]]
    # print(biKmeans(dataT,2))
    # None

    # print(kMeans(np.array(dataT), 2)[0])
    # [[52.18028336 51.20703954]
    #  [27.         29.]]