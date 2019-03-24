#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # which gpu to use

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as sk_mae
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dont allocate entire vram initially
set_session(tf.Session(config=config))
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Flatten, Concatenate
from keras.models import Sequential,Model
from keras.metrics import mean_absolute_error
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pdb
import numpy as np
import sys


# In[2]:


#Reading dat
print("Reading data...")
#pdb.set_trace()
img_dir = '/media/samba_share/data/BoneAge/Image/rsna/boneage-training-dataset/'
csv_path = '/media/samba_share/data/BoneAge/Image/rsna/boneage-training-dataset.csv'
age_df = pd.read_csv(csv_path)

print age_df.head(10)


# In[3]:


age_df['path'] = age_df['id'].map(lambda x: img_dir+"{}.png".format(x))
age_df['exists'] = age_df['path'].map(os.path.exists)
age_df['gender'] = age_df['male'].map(lambda x: "male" if x else "female")
print age_df['gender'].head(10)
print age_df['exists'].head(10)
print age_df['path'].head(10)


# In[4]:


# age_df['path'] = age_df['id'].map(lambda x: img_dir+"{}.png".format(x))
# age_df['exists'] = age_df['path'].map(os.path.exists)
# age_df['gender'] = age_df['male'].map(lambda x: "male" if x else "female")
mu = age_df['boneage'].mean()
sigma = age_df['boneage'].std()
print mu
print sigma
age_df['zscore'] = age_df['boneage'].map(lambda x: (x-mu)/sigma)
print age_df['zscore'].head(10)
age_df.dropna(inplace=True)


# In[5]:


#Examine the distribution of age and gender
print("{} images found out of total {} images".format(age_df['exists'].sum(),age_df.shape[0]))
print(age_df.sample(5))
age_df[['boneage','gender','zscore']].hist()
plt.show()
print("Reading complete !!!\n")


# In[6]:


# #Split into training testing and validation datasets
# print("Preparing training, testing and validation datasets ...")
# age_df['boneage_category'] = pd.cut(age_df['boneage'], 10)
# raw_train_df, test_df = train_test_split(age_df, 
#                                    test_size = 0.2, 
#                                    random_state = 2018,
#                                    stratify = age_df['boneage_category'])
# raw_train_df, valid_df = train_test_split(raw_train_df, 
#                                    test_size = 0.1,
#                                    random_state = 2018,
#                                    stratify = raw_train_df['boneage_category'])


# In[ ]:


#Split into training testing and validation datasets
print("Preparing training, testing and validation datasets ...")
age_df['boneage_category'] = pd.cut(age_df['boneage'], 10)
train_df, test_df = train_test_split(age_df, 
                                   test_size = 0.2, 
                                   random_state = 2018,
                                   stratify = age_df['boneage_category'])
train_df, valid_df = train_test_split(raw_train_df, 
                                   test_size = 0.1,
                                   random_state = 2018,
                                   stratify = raw_train_df['boneage_category'])


# In[7]:


#Balance the distribution in the training set
# train_df = raw_train_df.groupby(['boneage_category','gender']).apply(lambda x: x.sample(500, replace = True)).reset_index(drop=True)


# In[8]:


print(train_df.sample(5))
train_df[['boneage','gender']].hist(figsize = (10, 5))
plt.show()


# In[9]:


train_size = train_df.shape[0]
valid_size = valid_df.shape[0]
test_size = test_df.shape[0]
print("# Training images:   {}".format(train_size))
print("# Validation images: {}".format(valid_size))
print("# Testing images:    {}".format(test_size))


# In[10]:


from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
IMG_SIZE = (224, 224) # slightly smaller than vgg16 normally expects
core_idg = ImageDataGenerator(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range = 0.15, 
                              width_shift_range = 0.15, 
                              rotation_range = 5, 
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range=0.25,
                             preprocessing_function = preprocess_input)


# In[11]:


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


# In[12]:


train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'zscore', 
                               
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 32)

valid_gen = flow_from_dataframe(core_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'zscore', 
                              
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 256) # we can use much larger batches for evaluation
# used a fixed dataset for evaluating the algorithm
test_X, test_Y = next(flow_from_dataframe(core_idg, 
                              test_df, 
                             path_col = 'path',
                            y_col = 'zscore', 
                                
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 1024))


# In[13]:


t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -127, vmax = 127)
    c_ax.set_title('%2.0f months' % (c_y*sigma+mu))
    c_ax.axis('off')


# In[14]:


print t_x.shape[1:]


# In[15]:


print("Compiling deep model ...")
img = Input(t_x.shape[1:])

cnn_vec = InceptionV3(input_shape = t_x.shape[1:], include_top = False, weights = 'imagenet')(img)
cnn_vec = GlobalAveragePooling2D()(cnn_vec)
cnn_vec = Dropout(0.2)(cnn_vec)


dense_layer = Dense(1024, activation = 'relu')(cnn_vec)
dense_layer = Dropout(0.2)(dense_layer)
dense_layer = Dense(1024,activation='relu')(dense_layer)
dense_layer = Dropout(0.2)(dense_layer)
output_layer = Dense(1, activation = 'linear')(dense_layer) # linear is what 16bit did
bone_age_model = Model(inputs=img,outputs=output_layer)


# In[16]:


def mae_months(in_gt, in_pred):
    return mean_absolute_error(mu+sigma*in_gt, mu+sigma*in_pred)
bone_age_model.compile(optimizer = 'adam', loss = 'mse', metrics = [mae_months])
bone_age_model.summary()
print("Model compiled !!!\n")


# In[17]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('bone_age')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[19]:


bone_age_model.fit_generator(train_gen,
                                  steps_per_epoch = train_size/10,
                                  validation_data = (test_X,test_Y),
                                  epochs = 15, 
                                  callbacks = callbacks_list)


# In[20]:


bone_age_model.load_weights(weight_path)
print("Training complete !!!\n")


# In[21]:


#Evaluate model on test dataset
print("Evaluating model on test data ...\n")
print("Preparing testing dataset...")
test_X, test_Y = next(flow_from_dataframe(core_idg,test_df, 
                             path_col = 'path',
                            y_col = 'zscore',
                          
                            batch_size = 1024,
                            target_size = IMG_SIZE,
                             color_mode = 'rgb'))
                             # one big batch
print("Data prepared !!!")


# In[27]:


pred_Y = mu+sigma*bone_age_model.predict(x=test_X,batch_size=25,verbose=1)
test_Y_months = mu+sigma*test_Y
print("Mean absolute error on test data: "+str(sk_mae(test_Y_months,pred_Y)))

fig, ax1 = plt.subplots(1,1, figsize = (6,6))
ax1.plot(test_Y_months, pred_Y, 'r.', label = 'predictions')
ax1.plot(test_Y_months, test_Y_months, 'b-', label = 'actual')
ax1.legend()
ax1.set_xlabel('Actual Age (Months)')
ax1.set_ylabel('Predicted Age (Months)')

ord_idx = np.argsort(test_Y)
ord_idx = ord_idx[np.linspace(0, len(ord_idx)-1, num=8).astype(int)] # take 8 evenly spaced ones
fig, m_axs = plt.subplots(2, 4, figsize = (16, 32))
for (idx, c_ax) in zip(ord_idx, m_axs.flatten()):
    c_ax.imshow(test_X[idx, :,:,0], cmap = 'bone')
    title = 'Age: %2.1f/nPredicted Age: %2.1f/nGender: ' % (test_Y_months[idx], pred_Y[idx])
#     if test_X[1][idx]==0:
#       title+="Female/n"
#     else:
#       title+="Male/n"
    c_ax.set_title(title)
    c_ax.axis('off')
plt.show()


# In[26]:


ord_idx = np.argsort(test_Y)
ord_idx = ord_idx[np.linspace(0, len(ord_idx)-1, num=8).astype(int)] # take 8 evenly spaced ones
fig, m_axs = plt.subplots(2, 4, figsize = (16, 32))
for (idx, c_ax) in zip(ord_idx, m_axs.flatten()):
    c_ax.imshow(test_X[idx, :,:,0], cmap = 'bone')
    title = 'Age: %2.1fY Predicted Age: %2.1fY: ' % (test_Y_months[idx]/12.0, pred_Y[idx]/12.0)
#     if test_X[1][idx]==0:
#       title+="Female/n"
#     else:
#       title+="Male/n"
    c_ax.set_title(title)
    c_ax.axis('off')
plt.show()


# In[ ]:


ord_idx = np.argsort(test_Y)
ord_idx = ord_idx[np.linspace(0, len(ord_idx)-1, 8).astype(int)] # take 8 evenly spaced ones
fig, m_axs = plt.subplots(4, 2, figsize = (16, 32))
for (idx, c_ax) in zip(ord_idx, m_axs.flatten()):
    c_ax.imshow(test_X[idx, :,:,0], cmap = 'bone')
    
    c_ax.set_title('Age: %2.1fY\nPredicted Age: %2.1fY' % (test_Y_months[idx]/12.0, 
                                                           pred_Y[idx]/12.0))
    c_ax.axis('off')
fig.savefig('trained_img_predictions.png', dpi = 300)

