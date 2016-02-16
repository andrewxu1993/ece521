import numpy as np
import tensorflow as tf
import random as rd
from datetime import datetime

def accuracy(predictions, labels):
  predictions=tf.argmax(predictions, 1)
  labels=tf.argmax(labels, 1)
  return tf.reduce_sum(tf.cast(tf.not_equal(predictions,labels),"float32"))

def l2(batch_size,learning_rate,hidden_num):
  with np.load("notMNIST.npz") as data:
      images, labels = data["images"], data["labels"]

  image_size=28
  num_channels=1
  num_labels=10


  images=images.T.astype("float32")
  labels=np.eye(10)[labels[:,0]].astype("float32")

  s=images.shape
  images=images.reshape(s[0],image_size*image_size)


  train_dataset=images[0:15000]
  train_labels=labels[0:15000]
  valid_dataset=images[15000:16000]
  valid_labels=labels[15000:16000]
  test_dataset=images[16000:18720]
  test_labels=labels[16000:18720]

  #print train_dataset.shape
  #print valid_dataset.shape
  #print test_dataset.shape


  graph = tf.Graph()



  with graph.as_default():
    x_train=tf.placeholder(tf.float32,shape=(None,image_size*image_size))
    y_train=tf.placeholder(tf.float32,shape=(None,num_labels))


    # layer 1
    w1=tf.Variable(tf.truncated_normal([image_size*image_size,hidden_num]))
    b1=tf.Variable(tf.truncated_normal([hidden_num]))

    # layer2
    w2=tf.Variable(tf.truncated_normal([hidden_num,hidden_num]))
    b2=tf.Variable(tf.truncated_normal([hidden_num]))

    # layer 3
    w3=tf.Variable(tf.truncated_normal([hidden_num,num_labels]))
    b3=tf.Variable(tf.truncated_normal([num_labels]))



    logits=tf.matmul(x_train,w1)
    logits=tf.add(logits,b1)
    logits=tf.nn.relu(logits)
    logits=tf.matmul(logits,w2)
    logits=tf.add(logits,b2)
    logits=tf.nn.relu(logits)
    logits=tf.matmul(logits,w3)
    logits=tf.add(logits,b3)


    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,y_train))

    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    train_prediction=tf.nn.softmax(logits)

    er=accuracy(train_prediction,y_train)

  epoch_num=100

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    #print ("initialized")
    training_likelihood=[]
    training_error=[]
    validate_likelihood=[]
    validate_error=[]

    for step in range (epoch_num):
      for j in range (150):
        x_batch=train_dataset[j*batch_size:(j+1)*batch_size]
        y_batch=train_labels[j*batch_size:(j+1)*batch_size]


        feed_dict={x_train:x_batch,y_train:y_batch}
        _,l,tp=session.run([optimizer,cost,er], feed_dict=feed_dict)

      training_error.append(tp)
      training_likelihood.append(-1*l)

      feed_dict={x_train:valid_dataset,y_train:valid_labels}
      _,l,vp=session.run([optimizer,cost,er], feed_dict=feed_dict)
      validate_error.append(vp)
      validate_likelihood.append(-1*l)

      print("Epoch %03d, Validation Likelihood: %.5f, Validation Error Number: %.1f" % (step,-1*l,vp))

    feed_dict={x_train:test_dataset,y_train:test_labels}
    _,l,tp=session.run([optimizer,cost,er], feed_dict=feed_dict)
    print("Test Error Number: %.1f" % tp)
    return validate_error

def l3(batch_size,learning_rate,hidden_num):
  with np.load("notMNIST.npz") as data:
      images, labels = data["images"], data["labels"]

  image_size=28
  num_channels=1
  num_labels=10


  images=images.T.astype("float32")
  labels=np.eye(10)[labels[:,0]].astype("float32")

  s=images.shape
  images=images.reshape(s[0],image_size*image_size)


  train_dataset=images[0:15000]
  train_labels=labels[0:15000]
  valid_dataset=images[15000:16000]
  valid_labels=labels[15000:16000]
  test_dataset=images[16000:18720]
  test_labels=labels[16000:18720]

  #print train_dataset.shape
  #print valid_dataset.shape
  #print test_dataset.shape


  graph = tf.Graph()



  with graph.as_default():
    x_train=tf.placeholder(tf.float32,shape=(None,image_size*image_size))
    y_train=tf.placeholder(tf.float32,shape=(None,num_labels))


    # layer 1
    w1=tf.Variable(tf.truncated_normal([image_size*image_size,hidden_num]))
    b1=tf.Variable(tf.truncated_normal([hidden_num]))

    # layer2
    w2=tf.Variable(tf.truncated_normal([hidden_num,hidden_num]))
    b2=tf.Variable(tf.truncated_normal([hidden_num]))

    # layer3
    w3=tf.Variable(tf.truncated_normal([hidden_num,hidden_num]))
    b3=tf.Variable(tf.truncated_normal([hidden_num]))

    # layer 4
    w4=tf.Variable(tf.truncated_normal([hidden_num,num_labels]))
    b4=tf.Variable(tf.truncated_normal([num_labels]))



    logits=tf.matmul(x_train,w1)
    logits=tf.add(logits,b1)
    logits=tf.nn.relu(logits)
    logits=tf.matmul(logits,w2)
    logits=tf.add(logits,b2)
    logits=tf.nn.relu(logits)
    logits=tf.matmul(logits,w3)
    logits=tf.add(logits,b3)
    logits=tf.nn.relu(logits)
    logits=tf.matmul(logits,w4)
    logits=tf.add(logits,b4)


    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,y_train))

    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    train_prediction=tf.nn.softmax(logits)

    er=accuracy(train_prediction,y_train)

  epoch_num=100

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    #print ("initialized")
    training_likelihood=[]
    training_error=[]
    validate_likelihood=[]
    validate_error=[]

    for step in range (epoch_num):
      for j in range (150):
        x_batch=train_dataset[j*batch_size:(j+1)*batch_size]
        y_batch=train_labels[j*batch_size:(j+1)*batch_size]


        feed_dict={x_train:x_batch,y_train:y_batch}
        _,l,tp=session.run([optimizer,cost,er], feed_dict=feed_dict)

      training_error.append(tp)
      training_likelihood.append(-1*l)

      feed_dict={x_train:valid_dataset,y_train:valid_labels}
      _,l,vp=session.run([optimizer,cost,er], feed_dict=feed_dict)
      validate_error.append(vp)
      validate_likelihood.append(-1*l)

      print("Epoch %03d, Validation Likelihood: %.5f, Validation Error Number: %.1f" % (step,-1*l,vp))

    feed_dict={x_train:test_dataset,y_train:test_labels}
    _,l,tp=session.run([optimizer,cost,er], feed_dict=feed_dict)
    print("Test Error Number: %.1f" % tp)
    return validate_error

def l1(batch_size,learning_rate,hidden_num):
  with np.load("notMNIST.npz") as data:
      images, labels = data["images"], data["labels"]

  image_size=28
  num_channels=1
  num_labels=10


  images=images.T.astype("float32")
  labels=np.eye(10)[labels[:,0]].astype("float32")

  s=images.shape
  images=images.reshape(s[0],image_size*image_size)


  train_dataset=images[0:15000]
  train_labels=labels[0:15000]
  valid_dataset=images[15000:16000]
  valid_labels=labels[15000:16000]
  test_dataset=images[16000:18720]
  test_labels=labels[16000:18720]

  #print train_dataset.shape
  #print valid_dataset.shape
  #print test_dataset.shape


  graph = tf.Graph()



  with graph.as_default():
    x_train=tf.placeholder(tf.float32,shape=(None,image_size*image_size))
    y_train=tf.placeholder(tf.float32,shape=(None,num_labels))


    # layer 1
    w1=tf.Variable(tf.truncated_normal([image_size*image_size,hidden_num]))
    b1=tf.Variable(tf.truncated_normal([hidden_num]))

    # layer 4
    w4=tf.Variable(tf.truncated_normal([hidden_num,num_labels]))
    b4=tf.Variable(tf.truncated_normal([num_labels]))



    logits=tf.matmul(x_train,w1)
    logits=tf.add(logits,b1)
    logits=tf.nn.relu(logits)
    logits=tf.nn.relu(logits)
    logits=tf.matmul(logits,w4)
    logits=tf.add(logits,b4)


    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,y_train))

    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    train_prediction=tf.nn.softmax(logits)

    er=accuracy(train_prediction,y_train)

  epoch_num=100

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    #print ("initialized")
    training_likelihood=[]
    training_error=[]
    validate_likelihood=[]
    validate_error=[]

    for step in range (epoch_num):
      for j in range (150):
        x_batch=train_dataset[j*batch_size:(j+1)*batch_size]
        y_batch=train_labels[j*batch_size:(j+1)*batch_size]


        feed_dict={x_train:x_batch,y_train:y_batch}
        _,l,tp=session.run([optimizer,cost,er], feed_dict=feed_dict)

      training_error.append(tp)
      training_likelihood.append(-1*l)

      feed_dict={x_train:valid_dataset,y_train:valid_labels}
      _,l,vp=session.run([optimizer,cost,er], feed_dict=feed_dict)
      validate_error.append(vp)
      validate_likelihood.append(-1*l)

      print("Epoch %03d, Validation Likelihood: %.5f, Validation Error Number: %.1f" % (step,-1*l,vp))

    feed_dict={x_train:test_dataset,y_train:test_labels}
    _,l,tp=session.run([optimizer,cost,er], feed_dict=feed_dict)
    print("Test Error Number: %.1f" % tp)
    return validate_error

if __name__=="__main__":

  total_run=10

  learning_r=[]
  layer_n=[]
  hidden_n=[]
  vas=[]

  rd.seed(datetime.now())

  for i in range(0,total_run):
    lr=10**rd.uniform(-1,-3)
    ln=rd.randint(1,3)
    hn=rd.randint(1,5)*100
    learning_r.append(lr)
    layer_n.append(ln)
    hidden_n.append(hn)
    if ln==2:
      vas.append(l2(100,lr,hn)) # the best
    elif ln==3:
      vas.append(l3(100,lr,hn))
    else:
      vas.append(l1(100,lr,hn))

  for i in range(0,total_run):
    print ("The case with learning rate "+ str(learning_r[i])+", layer number "+str(layer_n[i])
            +", hidden number "+str(hidden_n[i])+", has validating accuracy: "+str(vas[i]))
