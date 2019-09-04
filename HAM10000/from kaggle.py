#!/usr/bin/env python
# coding: utf-8

# HAM1000 from https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000 and https://www.kaggle.com/sarques/aiwitbor

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools


# In[2]:


import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# In[3]:


base_add = os.path.join('data')
print(base_add)
print(os.listdir(base_add))


# In[4]:


images_paths = list(Path(base_add).glob(os.path.join('**', '*.jpg')))


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


df.head()


# In[7]:


df['dx'].unique()


# In[8]:


df['cell_type'] = df['dx'].map(lesion_type_dict.get)
df['path'] = df['image_id'].map(image_path_dict.get)
df['dx_code'] = pd.Categorical(df['dx']).codes
df.head()


# In[9]:


df['age'].fillna(df['age'].mean(), inplace = True)
df.info()


# In[10]:


df['cell_type'].value_counts().plot(kind = 'bar')


# In[11]:


df['age'].hist(bins=20)


# In[12]:


df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((100, 75))))
df['image'].map(lambda x: x.shape).value_counts()


# In[13]:


feats = df.drop(['dx_code'], axis = 1)
target = df['dx_code']


# In[14]:


feats.info()


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2)


# In[16]:


# Normalization
x_train = np.asarray(x_train['image'].tolist())
x_test = np.asarray(x_test['image'].tolist())

x_train_mean = np.mean(x_train)
x_test_mean = np.mean(x_test)

x_train_std = np.std(x_train)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std


# In[17]:


# Label Encodeing
y_train = to_categorical(y_train, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)


# In[18]:


y_train[:5]


# In[27]:


x_train = x_train.reshape(x_train.shape[0],75, 100, 3)
x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))


# In[28]:


input_shape = (75, 100, 3)
num_classes = 7

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'Same', input_shape = input_shape))
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'Same'))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'Same'))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.40))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# In[29]:


optimizer = Adam(lr = .001, beta_1 = .9, beta_2 = .999, epsilon = None, decay = .0, amsgrad = False)


# In[30]:


model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[31]:


learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 1, 
                                           factor = 0.5, min_lr = 0.00001)


# In[32]:


datagen = ImageDataGenerator(featurewise_center = False, samplewise_center = False,
                            featurewise_std_normalization = False, samplewise_std_normalization = False, 
                            zca_whitening = False, rotation_range = 10, zoom_range = 0.1, 
                            width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = False, 
                            vertical_flip = False)
datagen.fit(x_train)


# In[33]:


epochs = 25
batch_size = 10
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), 
                             epochs = epochs, verbose = 1, steps_per_epoch = x_train.shape[0] // batch_size,
                             callbacks = [learning_rate_reduction])


# In[ ]:


from matplotlib import figure


# In[ ]:


def plot_weight_image(layer, x, y):
    weights = model.layers[layer].get_weights()
    fig = plt.figure()
    for j in range(len(weights[0])):
        ax = fig.add_subplot(y, x, j+1)
        ax.matshow(weights[0][j][0], cmap = plt.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return plt


# In[ ]:


a = model.layers[0].get_weights()
print(a[0][:, :, :, 0])
print("*"*40)
print(a[0][:, :, :, 1])
a[0].shape


# In[ ]:


np.random.seed(12345)
def save_image(layer):
    a = model.layers[layer].get_weights()
    for i in range(100):
        for j in range(100):
            try:
                grid = a[0][:, :, i, j]
                img = plt.imshow(grid, interpolation = 'spline16', cmap = 'plasma')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig("test{} {}-{}.jpg".format(layer, i , j))

            except:
                break


# In[ ]:


# We are creating these images for weights of only 1st covolutional layer now, 
# all of them would take a lot more time. 
layers = [0]
# , 1, 4, 5]
for layer in layers:
    save_image(layer)


# In[ ]:




