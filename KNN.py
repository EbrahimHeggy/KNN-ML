

# In[134]:


import pandas as pd
import numpy as np
import math
import random

data = pd.read_csv('BankNote_Authentication.csv')
data = np.array(data)
np.random.shuffle(data)

m = 960


# In[135]:


mean_list = []
for i in range(4):
    mean=0.0
    for j in range(m):
        mean = mean + data[j][i]
    mean=mean/m
    mean_list.append(mean)


# In[136]:


std_list = []
for i in range(4):
    std=0.0
    for j in range(m):
        std= std+ ((data[j][i]-mean_list[i])*(data[j][i]-mean_list[i]))
    std=std/m
    std= math.sqrt(std)
    std_list.append(std)


# In[137]:


size = len(data)
for i in range(4):
    for j in range(size):
        data[j][i] = ((data[j][i]-mean_list[i])/std_list[i])
data_train = data[:960]
data_test = data[960:]        


# In[138]:


def knn (x_list,k):
    min_dist = []
    for i in range (m):
        distance = 0.0
        for j in range (4):
            distance += ((x_list[j]-data_train[i][j]) * (x_list[j]-data_train[i][j]))
        distance = math.sqrt(distance)
        list_index =[distance,i]
        min_dist.append(list_index)   
    min_dist.sort()
    #print(min_dist)

    zeroes=0
    ones =0
    i = min_dist[0]
    val = data_train[i[1]][4]
    for i in range (k):
        ind = min_dist[i]
        last_index = ind[1]
        if data_train[last_index][4] == 0.0:
            zeroes+=1
        else:
            ones +=1 
    if zeroes>ones :
        return 0.0
    elif ones>zeroes:
        return 1.0 
    else :
        return val


# In[139]:


test_size= len(data_test)
def run (k):
    acc=0.0
    for i in range(test_size):
        row = [data_test[i][0],data_test[i][1],data_test[i][2],data_test[i][3]]
        r= knn(row,k)
       # print(r)
        if  r == data_test[i][4]:
            acc +=1
    acc = acc
    return acc


# In[140]:


for i in range(1,10):
    print("k value:",i)
    acc = run(i)
    print("Number of correctly classified instances :", acc, " Total number of instances :" ,test_size)
    print("Accuracy : ",(acc/test_size))


# In[ ]:




