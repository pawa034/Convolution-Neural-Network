

```python
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

import pandas as pd
import numpy as np
```

    Using TensorFlow backend.
    


```python
data = pd.read_csv('E:/data_files/train.csv',skiprows=[0],header=None)
```


```python
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>775</th>
      <th>776</th>
      <th>777</th>
      <th>778</th>
      <th>779</th>
      <th>780</th>
      <th>781</th>
      <th>782</th>
      <th>783</th>
      <th>784</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split
```


```python
train,test=train_test_split(data, test_size=0.2, random_state=42)
```


```python
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (test.ix[:,1:].values).astype('float32')
test_labels = test.ix[:,0].values.astype('int32')
test_labels = np_utils.to_categorical(test_labels, 10)


```


```python
y_train = np_utils.to_categorical(labels) 
```


```python
# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]
```


```python
#for tensorflow as backend use the below code
X_train1 = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test1 = X_test.reshape(X_test.shape[0], 28, 28, 1)
```


```python
model = Sequential()
```


```python
#input == the output whatever the stride is set when the valid_mode = same
```


```python
model.add(Convolution2D(32,5,5,activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))



```

    C:\Users\pm00450108\AppData\Local\Continuum\Anaconda3\lib\site-packages\ipykernel\__main__.py:1: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (5, 5), input_shape=(28, 28, 1..., activation="relu")`
      if __name__ == '__main__':
    


```python
#2ND ADDITION
model.add(Convolution2D(32,5,5,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

```

    C:\Users\pm00450108\AppData\Local\Continuum\Anaconda3\lib\site-packages\ipykernel\__main__.py:2: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (5, 5), activation="relu")`
      from ipykernel import kernelapp as app
    


```python
#dense_layer_addition
model.add(Flatten())
model.add(Dense(128, activation='relu',input_shape=(28,28,1)))

model.add(Dense(10, activation='softmax'))

```


```python

```


```python
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```


```python
              
model.fit(X_train1, y_train, 
          batch_size=32, epochs=1, verbose=1) 
```

    Epoch 1/1
    33600/33600 [==============================] - 23s - loss: 0.1866 - acc: 0.9415    
    




    <keras.callbacks.History at 0x1ed1b15d278>




```python
score = model.evaluate(X_test1, test_labels, verbose=0)
```


```python
score
```




    [0.068537806312864019, 0.97726190476190478]




```python
    model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_49 (Conv2D)           (None, 24, 24, 32)        832       
    _________________________________________________________________
    max_pooling2d_42 (MaxPooling (None, 12, 12, 32)        0         
    _________________________________________________________________
    dropout_62 (Dropout)         (None, 12, 12, 32)        0         
    _________________________________________________________________
    conv2d_50 (Conv2D)           (None, 8, 8, 64)          51264     
    _________________________________________________________________
    max_pooling2d_43 (MaxPooling (None, 4, 4, 64)          0         
    _________________________________________________________________
    dropout_63 (Dropout)         (None, 4, 4, 64)          0         
    _________________________________________________________________
    flatten_26 (Flatten)         (None, 1024)              0         
    _________________________________________________________________
    dense_51 (Dense)             (None, 128)               131200    
    _________________________________________________________________
    dropout_64 (Dropout)         (None, 128)               0         
    _________________________________________________________________
    dense_52 (Dense)             (None, 10)                1290      
    =================================================================
    Total params: 184,586
    Trainable params: 184,586
    Non-trainable params: 0
    _________________________________________________________________
    


```python

```


```python

```

    1.0.1
    


```python

```


```python

```
