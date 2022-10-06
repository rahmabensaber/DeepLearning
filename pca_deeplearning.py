#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_breast_cancer


# In[2]:


breast = load_breast_cancer()


# In[3]:


breast_data = breast.data


# In[4]:


breast_data.shape


# In[5]:


breast_labels = breast.target
breast_labels.shape


# In[6]:


import numpy as np
labels = np.reshape(breast_labels,(569,1))


# In[7]:


final_breast_data = np.concatenate([breast_data,labels],axis=1)
final_breast_data.shape


# In[8]:


import pandas as pd
breast_dataset = pd.DataFrame(final_breast_data)


# In[9]:


features = breast.feature_names
features


# In[10]:


features_labels = np.append(features,'label')


# In[11]:


breast_dataset.columns = features_labels


# In[12]:


breast_dataset.head()


# In[13]:


breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)


# In[14]:


breast_dataset.tail()


# In[15]:


from keras.datasets import cifar10


# In[17]:


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# In[18]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[19]:


print('Traning data shape:', x_train.shape)
print('Testing data shape:', x_test.shape)


# In[20]:


y_train.shape,y_test.shape


# In[21]:


# Find the unique numbers from the train labels
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)


# In[24]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


label_dict = {
 0: 'airplane',
 1: 'automobile',
 2: 'bird',
 3: 'cat',
 4: 'deer',
 5: 'dog',
 6: 'frog',
 7: 'horse',
 8: 'ship',
 9: 'truck',
}
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(x_train[0], (32,32,3))
plt.imshow(curr_img)
print(plt.title("(Label: " + str(label_dict[y_train[0][0]]) + ")"))

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(x_test[0],(32,32,3))
plt.imshow(curr_img)
print(plt.title("(Label: " + str(label_dict[y_test[0][0]]) + ")"))


# In[26]:


from sklearn.preprocessing import StandardScaler
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features
x.shape


# In[27]:


np.mean(x),np.std(x)


# In[28]:


feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x,columns=feat_cols)
normalised_breast.tail()


# In[29]:


from sklearn.decomposition import PCA
pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)


# In[30]:


principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2'])
principal_breast_Df.tail()


# In[31]:


print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))


# In[32]:


plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
               , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})


# In[33]:


np.min(x_train),np.max(x_train)


# In[34]:


x_train = x_train/255.0


# In[35]:


np.min(x_train),np.max(x_train)


# In[36]:


x_train.shape


# In[37]:


x_train_flat = x_train.reshape(-1,3072)
feat_cols = ['pixel'+str(i) for i in range(x_train_flat.shape[1])]
df_cifar = pd.DataFrame(x_train_flat,columns=feat_cols)
df_cifar['label'] = y_train
print('Size of the dataframe: {}'.format(df_cifar.shape))


# In[38]:


df_cifar.head()


# In[39]:


pca_cifar = PCA(n_components=2)
principalComponents_cifar = pca_cifar.fit_transform(df_cifar.iloc[:,:-1])


# In[40]:


principal_cifar_Df = pd.DataFrame(data = principalComponents_cifar
             , columns = ['principal component 1', 'principal component 2'])
principal_cifar_Df['y'] = y_train
principal_cifar_Df.head()


# In[41]:


print('Explained variation per principal component: {}'.format(pca_cifar.explained_variance_ratio_))


# In[42]:


import seaborn as sns
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=principal_cifar_Df,
    legend="full",
    alpha=0.3
)


# In[43]:


x_test = x_test/255.0
x_test = x_test.reshape(-1,32,32,3)


# In[44]:


x_test_flat = x_test.reshape(-1,3072)


# In[45]:


pca = PCA(0.9)


# In[46]:


pca.fit(x_train_flat)


# In[47]:


pca.n_components_


# In[48]:


train_img_pca = pca.transform(x_train_flat)
test_img_pca = pca.transform(x_test_flat)


# In[50]:


from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from tensorflow.keras.optimizers import RMSprop


# In[51]:


batch_size = 128
num_classes = 10
epochs = 20


# In[52]:


model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(99,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# In[53]:


model.summary()


# In[61]:


model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(3072,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train_flat, y_train,batch_size=batch_size,epochs=epochs,verbose=1,
                    validation_data=(x_test_flat, y_test))


# In[ ]:




