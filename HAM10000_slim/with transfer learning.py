#!/usr/bin/env python
# coding: utf-8

# with https://www.kaggle.com/sarques/aiwitbor  and https://www.tensorflow.org/beta/tutorials/images/transfer_learning

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools


# In[2]:


import keras
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Flatten
from keras import backend as K
import itertools
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.metrics import categorical_accuracy


# In[3]:


base_add = os.path.join('data')
print(base_add)
print(os.listdir(base_add))


# # Preprocess imtems like in previous file

# In[4]:


images_paths = list(Path(base_add).glob(os.path.join('resized_images', '*.jpg')))


# In[5]:


image_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in images_paths}
print(list(image_path_dict.keys())[0])
print(list(image_path_dict.values())[0])
lesion_type_dict = {'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'}
df = pd.read_csv(os.path.join(base_add, 'HAM10000_metadata.csv'))


# In[6]:


df[df['dx'] == 'mel'].head()


# In[7]:


df['cell_type'] = df['dx'].map(lesion_type_dict.get)
df['path'] = df['image_id'].map(image_path_dict.get)
df['dx_code'] = pd.Categorical(df['dx']).codes
df['age'].fillna(df['age'].mean(), inplace = True)
df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x)))
df['image'].map(lambda x: x.shape).value_counts()


# In[8]:


df.head()


# In[9]:


feats = df.drop(['dx_code'], axis = 1)
target = df['dx_code']


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2)


# In[ ]:





# In[11]:



x_train = np.asarray(x_train['image'].tolist())
x_test = np.asarray(x_test['image'].tolist())
x_valid = np.asarray(x_valid['image'].tolist())


# In[12]:


example_image = x_train[0]
example_image.shape


# In[13]:


imgplot = plt.imshow(example_image)


# In[ ]:





# In[14]:


# Normalization
x_train_mean = np.mean(x_train)
x_test_mean = np.mean(x_test)
x_valid_mean = np.mean(x_valid)

x_train_std = np.std(x_train)
x_test_std = np.std(x_test)
x_valid_std = np.std(x_valid)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std
x_valid = (x_valid - x_valid_mean)/x_valid_std


# In[15]:


# Label Encodeing
y_train = to_categorical(y_train, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)
y_valid = to_categorical(y_valid, num_classes = 7)


# In[16]:


x_train = x_train.reshape(x_train.shape[0],75, 100, 3)
x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))
x_valid = x_valid.reshape(x_valid.shape[0], *(75, 100, 3))


# In[17]:


input_shape = (75, 100, 3)
num_classes = 7


# In[18]:


example_image = x_train[5012]
example_image.shape


# In[19]:


imgplot = plt.imshow(example_image)


# # Get the pretrained ConvNet

# In[20]:


base_model = keras.applications.MobileNetV2(input_shape=input_shape,
                                               include_top=False,
                                               weights='imagenet')


# In[21]:


base_model.trainable = False
base_model.summary()


# In[22]:


x = base_model.layers[-1].output
print(x)
print(base_model.input)


# In[23]:



x = Dropout(0.25)(x)
flattened = Flatten()(x)
predictions = Dense(num_classes, activation='softmax')(flattened)

# Create a new model with the new outputs
model = Model(inputs=base_model.input, outputs=predictions)


# In[24]:


for layer in model.layers[:-2]:
    layer.trainable = False


# In[25]:


optimizer = Adam(lr = .001, beta_1 = .9, beta_2 = .999, epsilon = None, decay = .0, amsgrad = False)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = [categorical_accuracy])


# In[26]:


model.summary()


# In[ ]:





# In[27]:


datagen = ImageDataGenerator(featurewise_center = False, samplewise_center = False,
                            featurewise_std_normalization = False, samplewise_std_normalization = False, 
                            zca_whitening = False, rotation_range = 10, zoom_range = 0.1, 
                            width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True, 
                            vertical_flip = True)
datagen.fit(x_train)


# In[28]:


datagen.flow(x_train, y_train, batch_size = 1)[0][1]


# In[29]:


class_weights={
    0: 1.0,  
    1: 1.0,  
    2: 1.0,  
    3: 1.0,  
    4: 1.0,  # mel
    5: 1.0,  
    6: 1.0,  
}


# In[30]:


learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_categorical_accuracy', patience = 3, verbose = 1, 
                                           factor = 0.2, min_lr = 0.0001)


# In[31]:


checkpoint = ModelCheckpoint("best_model.h5", monitor='val_categorical_accuracy', verbose=1,
                             save_best_only=True, mode='max')


# In[32]:


epochs = 2
batch_size = 10
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), 
                             epochs = epochs,
                              verbose = 1,
                              class_weight=class_weights,
                              validation_data=datagen.flow(x_valid, y_valid, batch_size = batch_size),                              
                              callbacks=[checkpoint, learning_rate_reduction]
                             )


# In[33]:


test_batches = datagen.flow(x_test, y_test, batch_size = batch_size)


# In[36]:


test_loss, val_acc  = model.evaluate_generator(test_batches)
print(test_loss)
print(val_acc)


# In[ ]:


#model.load_weights('best_model.h5')

