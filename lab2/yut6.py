import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import random as rd
from datetime import datetime

with np.load("notMNIST.npz") as data:
    images, labels  =data["images"], data["labels"]

# flatten the input images
image_size = 28
num_labels = 10

images = images/255.0
images = images.reshape((image_size * image_size,18720)).T
labels = (np.arange(num_labels) == labels[:,None])
labels = labels.reshape(-1,10).astype(np.float32)

image_train = images[0:15000,:]
image_val = images[15000:16000,:]
image_test = images[16000:,:]
label_train = labels[0:15000,:]
label_val = labels[15000:16000,:]
label_test = labels[16000:,:]

def neural_network(images_train, labels_train, images_val,labels_val, images_test,labels_test,learning_rate,hidden_num,layer_num, dropout):

    # Placeholders
    X = tf.placeholder("float32", shape=(None, 784))
    Y = tf.placeholder("float32", shape=(None, 10))
    keep_prob = tf.placeholder("float32")

    W = []
    b = []
    Z = []

    # Initiate the neural network
    for i in range(layer_num+1):
        if i == 0 and layer_num==1:
            W.append(tf.Variable(tf.random_normal(([784, hidden_num[i][1]]),0,1), name="weight%d"%i))
            b.append(tf.Variable(tf.random_normal(([10]),0,1), name="bias%d"%i))
            Z.append(tf.add(tf.matmul(X, W[i]), b[i]))
            Z[i] = tf.nn.relu(Z[i])
            if dropout:
                Z[i]=tf.nn.dropout(Z[i],keep_prob)
            break
        elif i == 0:
            W.append(tf.Variable(tf.random_normal(([784, hidden_num[i][1]]),0,1), name="weight%d"%i))
            b.append(tf.Variable(tf.random_normal(([hidden_num[i][1]]),0,1), name="bias%d"%i))
            Z.append(tf.add(tf.matmul(X, W[i]), b[i]))
            Z[i] = tf.nn.relu(Z[i])
            if dropout:
                Z[i]=tf.nn.dropout(Z[i],keep_prob)
        elif i == layer_num:
            W.append(tf.Variable(tf.random_normal(([hidden_num[i-1][1], 10]),0,1), name="weight%d"%i))
            b.append(tf.Variable(tf.random_normal(([10]),0,1), name="bias%d"%i))
        else:
            W.append(tf.Variable(tf.random_normal(([hidden_num[i-1][1],hidden_num[i][1]]),0,1), name="weight%d"%i))
            b.append(tf.Variable(tf.random_normal(([hidden_num[i-1][1]]),0,1), name="bias%d"%i))
            Z.append(tf.add(tf.matmul(Z[i-1], W[i]), b[i]))
            Z[i] = tf.nn.relu(Z[i])
            if dropout:
                Z[i]=tf.nn.dropout(Z[i],keep_prob)

    logits = tf.add(tf.matmul(Z[-1], W[-1]), b[-1])
    output = tf.nn.softmax(logits)

    # cost = -tf.reduce_sum(Y*tf.log(output))
    cost_batch = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y)
    cost = tf.reduce_mean(cost_batch)

    lr_copy = learning_rate
    learning_rate =tf.cast(learning_rate,"float32")
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    pred = tf.argmax(output,1)
    pred_float = tf.cast(pred, "float32")

    #accuracy
    errors = tf.reduce_sum(tf.cast(tf.not_equal(pred_float, tf.cast(tf.argmax(Y,1),"float32")),"float32"))

    sess = tf.InteractiveSession()

    init = tf.initialize_all_variables()
    sess.run(init)

    errors_train = []
    costs_train = []
    errors_eval = []
    costs_eval = []
    epoch_num = 2
    errors_trend = []
    weight_save = []
    weight_copy = []

    def modify_weights():
        lis = []
        for i in range(len(W)):
            lis.append(W[i])
        for i in range(len(b)):
            lis.append(b[i])
        return lis


    def save_weights():
        # weight = []
        # # for i in range(len(W)+len(b)):
        # #     weight.append([])
        weight = sess.run(modify_weights(), feed_dict={X: x_batch, Y: y_batch, keep_prob: 0.5})
        weight_save.append(weight)


    print ("Training Start: learning rate = %0.6f, hidden unit number = %04d"%(lr_copy,hidden_num[0][1]))
    for epoch in range(epoch_num):
      for i in xrange(150):
        x_batch = images_train[i * 100: (i + 1) * 100]
        y_batch = labels_train[i * 100: (i + 1) * 100]
        sess.run(train_op, feed_dict={X: x_batch, Y: y_batch, keep_prob: 0.5})
        #Display logs per epoch step
      if epoch % 1 == 0:
        save_weights()
        weight_copy = weight_save[-1]
        #print len(weight_copy)
        cost_train, error_train = sess.run([cost, errors],feed_dict={X: images_train, Y: labels_train,keep_prob: 1})
        errors_train.append(error_train)
        costs_train.append(-cost_train)
        cost_eval, error_eval = sess.run([cost, errors],feed_dict={X: images_val, Y: labels_val,keep_prob: 1})
        errors_eval.append(error_eval)
        costs_eval.append(-cost_eval)
        print ("Epoch:%04d, log-likelihood = -%0.9f, # of training errors = %0.4f, # of evaluation errors = %0.4f" %
               (epoch+1, cost_train, error_train, error_eval))
        if epoch >=1:
            if (errors_eval[-2] - errors_eval[-1]) <= 0:
                errors_trend.append(-1)
            else:
                errors_trend.append(1)
        if epoch >= 10 and np.sum(errors_trend[-5:])==-5:
            weight_copy = weight_save[-1]
            print "Early stopping triggered: last 6 evaluation errors are non-decreasing ",errors_eval[-6:]
            break

    epoches = range(0,epoch_num)

    #plot_neural(epoches,costs_train,epoches,costs_eval,epoches,errors_train,epoches,errors_eval,lr_copy,lr)
    print len(weight_copy), len(W), len(b)

    def modify_dic():
        dic = {X: images_test, Y: labels_test,keep_prob: 1}
        for i in range(len(W)):
            dic[W[i]] = weight_copy[i]
        for i in range(len(b)):
            dic[b[i]] = weight_copy[len(W)+i]
        return dic

    error_test = sess.run(errors,feed_dict=modify_dic())

    return error_test


# Task 6
for i in range(10):
    print i
    rd.seed(datetime.now())
    learning_r=10**rd.uniform(-2,-4)
    layer_num=rd.randint(1,3)
    hidden_num=rd.randint(1,5)*100
    if layer_num==1:
        hidden_matrix=[[0,hidden_num]]
    elif layer_num==2:
        hidden_matrix=[[0,hidden_num],[1,hidden_num]]
    else:
        hidden_matrix=[[0,hidden_num],[1,hidden_num],[2,hidden_num]]
    dropout=rd.randint(0,1)
    if dropout==0:
        dropout=False
    else:
        drop=True
    ers= neural_network(image_train,label_train,image_val,label_val,image_test,label_test,learning_r,hidden_matrix,layer_num,True)
    print ("Case with learning rate %.8f, %1d layers and hidden units %4d at each layer has validating errors: %.1f" %(learning_r,
                layer_num,hidden_num,ers))