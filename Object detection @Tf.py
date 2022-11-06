#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import pandas as pd
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
X_train.shape


# In[4]:


X_test.shape


# In[5]:


y_train.shape


# In[6]:


y_test.shape


# In[7]:


y_train[:5]


# In[8]:


y_train = y_train.reshape(-1,)
y_train[:5]


# In[9]:


y_test = y_test.reshape(-1,)


# In[10]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[11]:


def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[12]:


plot_sample(X_train, y_train, 0)


# In[13]:


plot_sample(X_train, y_train, 49089)


# In[14]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[15]:


fig, ax = plt.subplots(7, 7)
k = 0
 
for i in range(7):
    for j in range(7):
        ax[i][j].imshow(X_train[k], aspect='auto')
        k += 1
 
plt.show()


# In[16]:


ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=10)


# In[17]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# In[18]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[19]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[20]:


cnn.fit(X_train, y_train, epochs=20)


# In[21]:


cnn.evaluate(X_test,y_test)


# In[22]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[23]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[24]:


y_test[:5]


# In[25]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = cnn.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# In[26]:


plot_sample(X_test, y_test,121)


# In[27]:


plot_sample(X_test, y_test,1311)


# In[28]:


classes[y_classes[1311]]


# In[29]:


classes[y_classes[500]]


# In[30]:


classes[y_classes[2]]


# In[31]:


labels = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()
 
# select the image from our test dataset
#img = tf.keras.utils.load_img('Z:\conda\envs\image classification tf\images\d6.jpg', target_size = (100,100,3))
#image_number = tf.keras.utils.img_to_array(img)

image_number = 500

#5722
# display the image
plt.imshow(X_test[image_number])
 
# load the image in an array
n = np.array(X_test[image_number])
 
# reshape it
p = n.reshape(1, 32, 32, 3)
 
# pass in the network for prediction and
# save the predicted label
predicted_label = labels[cnn.predict(p).argmax()]
 
# load the original label
original_label = labels[y_test[image_number]]
 
# display the result
print("Original label is {} and predicted label is {}".format(original_label, predicted_label))


# In[32]:


from matplotlib.image import imread

img = imread('Z:\\conda\\envs\\image classification tf\\images\\d7.jpg')
print(type(img))
img = img/255
plt.imshow(img)
plt.xlabel(classes[8])


# In[33]:


import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


# In[34]:


import numpy as np
from keras.applications.vgg16 import preprocess_input

img = tf.keras.utils.load_img('Z:\conda\envs\image classification tf\images\download3.jpg', target_size = (100,100,3))
img = tf.keras.utils.img_to_array(img)
img = img/255
plt.imshow(img)
plt.xlabel(classes[y_classes[0]])
plt.show()


# In[35]:


img = tf.keras.utils.load_img('Z:\conda\envs\image classification tf\images\d6.jpg', target_size = (100,100,3))
img = tf.keras.utils.img_to_array(img)
img = img/255
plt.imshow(img)
plt.xlabel(classes[y_classes[7]])
plt.show()


# In[36]:


#numpydata = tf.keras.utils.img_to_array(img)
 
#print(type(numpydata))
 
#print(numpydata.shape)


# In[37]:


converter = tf.lite.TFLiteConverter.from_saved_model("Z:/conda/envs/image classification tf/saved_model")
tflite_model = converter.convert()


# In[38]:


len(tflite_model)


# In[39]:


converter = tf.lite.TFLiteConverter.from_saved_model("Z:/conda/envs/image classification tf/saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()


# In[40]:


len(tflite_quant_model)


# In[41]:


with open("tflite_model.tflite", "wb") as f:
    f.write(tflite_model)
    
with open("tflite_quant_model.tflite", "wb") as f:
    f.write(tflite_quant_model)


# In[ ]:




