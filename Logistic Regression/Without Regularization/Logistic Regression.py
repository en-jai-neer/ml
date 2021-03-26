
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[2]:


data = pd.read_csv('data_banknote_authentication.txt', sep=",", header=None)


# In[3]:


data.rename(columns={0: "Variance", 1: "Skewness", 2: "Curtosis", 3: "Entropy", 4: "Class"}, inplace=True)


# In[4]:


Y = data.loc[:, 'Class']
X = data.loc[:, :'Entropy']


# In[5]:


#The features are scaled using the equation -  z=(x-μ)/σ

for colName in X.columns:
        X[colName] = (X[colName] - X[colName].mean())/X[colName].std()


# In[6]:


X.insert(0, 'X0', 1)


# In[7]:


X.head()


# In[8]:


X.describe()


# In[9]:


#Splitting the data into training set and testing set in the ratio of 80:20

X_train=X.sample(frac=0.8,random_state=3) #random state is a seed value
X_test=X.drop(X_train.index)
Y_train=Y.sample(frac=0.8,random_state=3) 
Y_test=Y.drop(Y_train.index)
X_train.head()


# In[10]:


def sigmoid(z):
    return 1/ (1 + np.exp(-z))


# In[11]:


#Converting Pandas Series to Numpy arrays

X_train = X_train.values
X_test = X_test.values
Y_train = Y_train.values
Y_test = Y_test.values


# In[12]:


X_train.shape


# In[13]:


def costofgradient(it, i):
    sum = 0
    for j, row in enumerate(X_train):
        sum += (sigmoid(np.dot(w[it], row)) - Y_train[j] )*row[i]
    return sum


# In[14]:


def updateW(it):
    locW = []
    for i in range(5):
        locW.append(w[it][i] - alpha*costofgradient(it, i))
    return locW


# In[28]:


w=[]
w.append(np.random.rand(5))
alpha = 0.075     #learning Rate
iterations = 50  #Epochs
x_axis=[]
y_axis = []

for it in range(iterations):
    x_axis.append(it)
    y_pred = sigmoid(np.dot(X_test, w[it]))
    y_axis.append(np.sum((y_pred-Y_test)**2))
    x = updateW(it)
    w.append(x)

plt.plot(x_axis, y_axis, color="#001a66", linewidth=2)
plt.xlabel('Iterations')  
plt.ylabel('Loss')  
plt.title('Progression of Loss Function with Gaussian Weight Initialization')
plt.show
plt.savefig('Gaussian.jpg', bbox_inches='tight', dpi=300, quality=80, optimize=True, progressive=True)


# In[31]:


y_pred = sigmoid(np.matmul(X_test, w[iterations]))
for i, val in enumerate(y_pred):
    if(val>=0.5):
        y_pred[i]=1
    else:
        y_pred[i]=0

round_off_values = np.around(w[iterations], decimals = 5)
print(round_off_values)
print(y_pred.shape)


# In[30]:


#Computing the Confusion Matrix
K = len(np.unique(Y_test)) # Number of classes 
confusion_matrix = np.zeros((K, K))

for i in range(len(Y_test)):
    x = Y_test[i]
    y = y_pred[i].astype(int)
    confusion_matrix[x][y] += 1
print(confusion_matrix)

precision = confusion_matrix[1][1]/(confusion_matrix[0][1] + confusion_matrix[1][1])
recall = confusion_matrix[1][1]/(confusion_matrix[1][0] + confusion_matrix[1][1])
print('The Precision is - ', precision)
print('The Recall is - ', recall)
f_score = 2*precision*recall/(precision+recall)
print('The F1-score is - ',f_score)

sse = np.sum((y_pred-Y_test)**2)
print('The Squared Sum of Errors is - ',sse)


# In[27]:


print('Accuracy of logistic regression classifier on test set: {:.5f}'.format((Y_test.shape[0]-sse)/Y_test.shape[0]*100))

