
# coding: utf-8

# In[1]:

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

import pandas as pd
import numpy as np


# In[4]:

data = pd.read_csv('E:/data_files/train.csv',skiprows=[0],header=None)


# In[257]:

data.head()


# In[3]:

from sklearn.model_selection import train_test_split


# In[272]:

train,test=train_test_split(data, test_size=0.2, random_state=42)


# In[273]:

labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (test.ix[:,1:].values).astype('float32')
test_labels = test.ix[:,0].values.astype('int32')
test_labels = np_utils.to_categorical(test_labels, 10)



# In[274]:

y_train = np_utils.to_categorical(labels) 


# In[275]:

# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]


# In[276]:

#for tensorflow as backend use the below code
X_train1 = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test1 = X_test.reshape(X_test.shape[0], 28, 28, 1)


# In[286]:

model = Sequential()


# In[117]:

#input == the output whatever the stride is set when the valid_mode = same


# In[287]:

model.add(Convolution2D(32,5,5,activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))




# In[288]:

#2ND ADDITION
model.add(Convolution2D(32,5,5,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# In[289]:

#dense_layer_addition
model.add(Flatten())
model.add(Dense(128, activation='relu',input_shape=(28,28,1)))

model.add(Dense(10, activation='softmax'))


# In[ ]:




# In[290]:

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[291]:


model.fit(X_train1, y_train, 
          batch_size=32, epochs=1, verbose=1) 


# In[292]:

score = model.evaluate(X_test1, test_labels, verbose=0)


# In[293]:

score


# In[256]:

model.summary()


# In[1]:




# In[4]:




# In[ ]:




# In[ ]:



