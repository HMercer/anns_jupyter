#!/usr/bin/env python
# coding: utf-8

# Recurent version of "sinus-full" to predict values outside of range as well
# 

# In[1]:


import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import shuffle_batch, variable_summaries
import os

get_ipython().run_line_magic('matplotlib', 'notebook')

dir_path = os.getcwd()

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
print(dir_path)


# In[2]:


tf.VERSION


# In[3]:


import platform
print(platform.python_version())


# In[4]:


t_min, t_max = -5, 5
section_start = (t_max + t_min) / 2
resolution = 0.1
n_steps = 20

def time_series(t):
    return np.sin(t)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)


# In[5]:


t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))

t_instance = np.linspace(start = section_start, stop = section_start + resolution * (n_steps + 1),num = n_steps + 1)

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.title("A time series (generated)", fontsize=14)
plt.plot(t, time_series(t), label=r"original")
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
plt.legend(loc="lower left", fontsize=14)
#plt.axis([-10, 10, -17, 13])
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.title("A training instance", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "c*", markersize=10, label="target")
plt.legend(loc="upper left")
plt.xlabel("Time")


# In[6]:


n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])


# In[7]:


cell =  tf.keras.layers.SimpleRNNCell(units=n_neurons, activation=tf.nn.relu)                        


# In[8]:


rnn_outputs = tf.keras.layers.RNN(cell,dtype=tf.float32, name="hidden1")(X)
print(rnn_outputs.get_shape())


# In[9]:


stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons], name='reshape1')
stacked_outputs = tf.keras.layers.Dense(n_outputs,name="hidden2")(stacked_rnn_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs], name='reshape2')


# In[ ]:





# In[10]:


learning_rate = 0.001

loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# In[11]:


n_iterations = 1500
batch_size = 50
save_path =os.path.join(dir_path,"model","recurrent_sinus_model")

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
    
    saver.save(sess, save_path)


# In[ ]:


with tf.Session() as sess:                      
    saver.restore(sess, save_path)  

    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})


# In[ ]:


plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")

plt.show()


# In[ ]:


with tf.Session() as sess:                      
    saver.restore(sess, save_path)  

    X_new = time_series(np.array(t.reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})


# In[ ]:


plt.title("A time series (generated)", fontsize=14)
plt.plot(t, time_series(t), label=r"original",linewidth=5,c='r')
plt.plot(t[:-1], time_series(t[:-1]), "b-", linewidth=3, label="A training instance")
plt.legend(loc="lower left", fontsize=14)

plt.xlabel("Time")
plt.ylabel("Value")


# In[ ]:




