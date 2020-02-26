import pandas as pd
import numpy as np
import heapq

#读取数据
meal_order_detail=pd.read_csv('meal_order_detail.csv',encoding='utf-8')
meal_order_info=pd.read_csv('meal_order_info.csv',encoding='gbk')

#数据预处理

#处理订单详情表中的特殊符号
meal_order_detail["dishes_name"]=meal_order_detail["dishes_name"].apply(str.rstrip)

#保留订单表中订单状态为1的订单
meal_order_info=meal_order_info.loc[meal_order_info["order_status"]==1]

#清除订单详情表中订单表没出现的订单
Max=max(max(meal_order_info["info_id"]),max(meal_order_detail["order_id"]))
flag=[False for i in range(Max+1)]
for i in meal_order_info["info_id"]:
    flag[i]=True
for i in meal_order_detail.index:
    if flag[meal_order_detail["order_id"].loc[i]]==False:
        meal_order_detail=meal_order_detail.drop(i)

#清除订单详情表中含有白饭的订单
meal_order_detail=meal_order_detail.loc[~meal_order_detail["dishes_name"].str.contains("白饭")]

#只保留菜品名称和用户id
meal_order_detail=meal_order_detail[["dishes_name","emp_id"]]

#清除重复数据
meal_order_detail=meal_order_detail[~meal_order_detail.duplicated()]

#保留点过三道菜以上的用户
count=meal_order_detail["emp_id"].value_counts()
for i in meal_order_detail.index:
    if count[meal_order_detail["emp_id"].loc[i]]<=3:
        meal_order_detail=meal_order_detail.drop(i)

#重置数据的标签
meal_order_detail=meal_order_detail.reset_index()

#划分训练集和测试集

#储存所有菜名
dishes_count=meal_order_detail["dishes_name"].value_counts()

#按用户进行划分
count=meal_order_detail["emp_id"].value_counts()
train=count.sample(frac=0.7)

#把meal_order_index中按已经划分好的用户进行分组
train_data={"emp_id":[],"dishes_name":[]}
test_data={"emp_id":[],"dishes_name":[]}
for i in meal_order_detail.index:
    if (meal_order_detail["emp_id"].loc[i] in train.index):
        train_data["emp_id"].append(meal_order_detail["emp_id"].loc[i])
        train_data["dishes_name"].append(meal_order_detail["dishes_name"].loc[i])
    else:
        test_data["emp_id"].append(meal_order_detail["emp_id"].loc[i])
        test_data["dishes_name"].append(meal_order_detail["dishes_name"].loc[i])
train_data=pd.DataFrame(train_data,columns=["emp_id","dishes_name"],index=range(len(train_data["emp_id"])))
test_data=pd.DataFrame(test_data,columns=["emp_id","dishes_name"],index=range(len(test_data["emp_id"])))

#构建菜品的相似度矩阵

#构建基于训练集的所有菜品的二元矩阵data
emp_id=train_data["emp_id"].value_counts().index
dishes_name=dishes_count.index
dishes_count=train_data["dishes_name"].value_counts()
emp={}
dishes={}
for i in range(len(emp_id)):
    emp[emp_id[i]]=i
for i in range(len(dishes_name)):
    dishes[dishes_name[i]]=i
data=np.mat(np.zeros((len(emp),len(dishes))))
for i in train_data.index:
    data[emp[train_data["emp_id"].loc[i]],dishes[train_data["dishes_name"].loc[i]]]=1

#构建同现矩阵occurrence
occurrence=np.mat(np.zeros((len(dishes),len(dishes))))
for i in range(len(dishes)):
    for j in range(len(dishes)):
        if i!=j:
            for e in range(len(emp)):
                if data[e,i]==1 and data[e,j]==1:
                    occurrence[i,j]+=1

#构建所有菜品的相似度矩阵sim
sim=np.mat(np.zeros((len(dishes),len(dishes))))
for i in range(len(dishes)):
    for j in range(len(dishes)):
        if i==j:
            sim[i,j]=1
        else:
            sim[i,j]=occurrence[i,j]/(dishes_count[dishes_name[i]]+dishes_count[dishes_name[j]]-occurrence[i,j])

#模型评估

#构建基于测试集的菜品的二元矩阵R
test_emp_id=test_data["emp_id"].value_counts().index
test_emp={}
test_dishes={}
for i in range(len(test_emp_id)):
    test_emp[test_emp_id[i]]=i
R=np.mat(np.zeros((len(test_emp),len(dishes))))
for i in test_data.index:
    R[test_emp[test_data["emp_id"].loc[i]],dishes[test_data["dishes_name"].loc[i]]]=1

#计算测试集用户对每个菜品的感兴趣程度的矩阵P
P=(R*sim).tolist()

#定义推荐列表长度
num=5

#生成推荐列表，并计算正确推荐的菜品数
correct=0
for i in range(len(P)):
    max_index=list(map(P[i].index,heapq.nlargest(num,P[i])))
    for j in range(num):
        if R[i,max_index[j]]==1:
            correct+=1

#计算准确率并输出
print(correct/(num*len(P)))














