import numpy as np
import tensorflow as tf
import random as rd
from datetime import datetime

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

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
    x_train=tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
    y_train=tf.placeholder(tf.float32,shape=(batch_size,num_labels))


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

    optimizer=tf.train.MomentumOptimizer(learning_rate,learning_rate).minimize(cost)

    train_prediction=tf.nn.softmax(logits)
    valid_prediction=tf.nn.relu(tf.add(tf.matmul(valid_dataset,w1),b1))
    valid_prediction=tf.nn.relu(tf.add(tf.matmul(valid_prediction,w2),b2))
    valid_prediction=tf.nn.softmax(tf.add(tf.matmul(valid_prediction,w3),b3))
    test_prediction=tf.add(tf.matmul(test_dataset,w1),b1)
    test_prediction=tf.add(tf.matmul(test_prediction,w2),b2)
    test_prediction=tf.nn.softmax(tf.add(tf.matmul(test_prediction,w3),b3))


  step_num=1000

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print ("initialized")
    va=[]
    ta=[]
    for step in range (step_num):
      offset=(step*batch_size)%(train_labels.shape[0]-batch_size)
      x_batch=train_dataset[offset:(offset+batch_size),:]
      y_batch=train_labels[offset:(offset+batch_size),:]


      feed_dict={x_train:x_batch,y_train:y_batch}
      _,l,tp,vp,tp=session.run([optimizer,cost,train_prediction,valid_prediction,test_prediction],
                               feed_dict=feed_dict)

      if (step%100==0):
        #print ("Minibatch loss at step %d: %f" %(step,l))
        #print("Minibatch accuracy: %.1f%%" % accuracy(tp,y_batch))
        va.append(accuracy(vp,valid_labels))
        ta.append(accuracy(tp,test_labels))
        #if len(va)>5 and va[-1]<va[-2] and va[-1]<va[-3]:
        #  va.pop(1)
        #  ta.pop(1)
        #  break

        #print("Validation accuracy: %.1f%%" % accuracy(vp,valid_labels))
    print("Test accuracy: %.1f%%" % ta[-1])
    print va
    return va

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
    x_train=tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
    y_train=tf.placeholder(tf.float32,shape=(batch_size,num_labels))


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

    optimizer=tf.train.MomentumOptimizer(learning_rate,learning_rate).minimize(cost)

    train_prediction=tf.nn.softmax(logits)
    valid_prediction=tf.nn.relu(tf.add(tf.matmul(valid_dataset,w1),b1))
    valid_prediction=tf.nn.relu(tf.add(tf.matmul(valid_prediction,w2),b2))
    valid_prediction=tf.nn.relu(tf.add(tf.matmul(valid_prediction,w3),b3))
    valid_prediction=tf.nn.softmax(tf.add(tf.matmul(valid_prediction,w4),b4))
    test_prediction=tf.nn.relu(tf.add(tf.matmul(test_dataset,w1),b1))
    test_prediction=tf.nn.relu(tf.add(tf.matmul(test_prediction,w2),b2))
    test_prediction=tf.nn.relu(tf.add(tf.matmul(test_prediction,w3),b3))
    test_prediction=tf.nn.softmax(tf.add(tf.matmul(test_prediction,w4),b4))


  step_num=1000

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print ("initialized")
    va=[]
    ta=[]
    for step in range (step_num):
      offset=(step*batch_size)%(train_labels.shape[0]-batch_size)
      x_batch=train_dataset[offset:(offset+batch_size),:]
      y_batch=train_labels[offset:(offset+batch_size),:]


      feed_dict={x_train:x_batch,y_train:y_batch}
      _,l,tp,vp,tp=session.run([optimizer,cost,train_prediction,valid_prediction,test_prediction],
                               feed_dict=feed_dict)

      if (step%100==0):
        #print ("Minibatch loss at step %d: %f" %(step,l))
        #print("Minibatch accuracy: %.1f%%" % accuracy(tp,y_batch))
        va.append(accuracy(vp,valid_labels))
        ta.append(accuracy(tp,test_labels))
        #if len(va)>5 and va[-1]<va[-2] and va[-1]<va[-3]:
        #  va.pop(1)
        #  ta.pop(1)
        #  break

        #print("Validation accuracy: %.1f%%" % accuracy(vp,valid_labels))
    print("Test accuracy: %.1f%%" % ta[-1])
    print va
    return va

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
    x_train=tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
    y_train=tf.placeholder(tf.float32,shape=(batch_size,num_labels))


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

    optimizer=tf.train.MomentumOptimizer(learning_rate,learning_rate).minimize(cost)

    train_prediction=tf.nn.softmax(logits)
    valid_prediction=tf.nn.relu(tf.add(tf.matmul(valid_dataset,w1),b1))
    valid_prediction=tf.nn.softmax(tf.add(tf.matmul(valid_prediction,w4),b4))
    test_prediction=tf.nn.relu(tf.add(tf.matmul(test_dataset,w1),b1))
    test_prediction=tf.nn.softmax(tf.add(tf.matmul(test_prediction,w4),b4))


  step_num=1000

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print ("initialized")
    va=[]
    ta=[]
    for step in range (step_num):
      offset=(step*batch_size)%(train_labels.shape[0]-batch_size)
      x_batch=train_dataset[offset:(offset+batch_size),:]
      y_batch=train_labels[offset:(offset+batch_size),:]


      feed_dict={x_train:x_batch,y_train:y_batch}
      _,l,tp,vp,tp=session.run([optimizer,cost,train_prediction,valid_prediction,test_prediction],
                               feed_dict=feed_dict)

      if (step%100==0):
        #print ("Minibatch loss at step %d: %f" %(step,l))
        #print("Minibatch accuracy: %.1f%%" % accuracy(tp,y_batch))
        va.append(accuracy(vp,valid_labels))
        ta.append(accuracy(tp,test_labels))
        #if len(va)>5 and va[-1]<va[-2] and va[-1]<va[-3]:
        #  va.pop(1)
        #  ta.pop(1)
        #  break

        #print("Validation accuracy: %.1f%%" % accuracy(vp,valid_labels))
    print("Test accuracy: %.1f%%" % ta[-1])
    print va
    return va

if __name__=="__main__":

  total_run=2

  learning_r=[]
  layer_n=[]
  hidden_n=[]
  vas=[]

  rd.seed(datetime.now())

  for i in range(0,total_run):
    lr=10**rd.uniform(-2,-4)
    ln=rd.randint(1,3)
    hn=rd.randint(1,5)*100
    learning_r.append(lr)
    layer_n.append(ln)
    hidden_n.append(hn)
    ln=1
    if ln==2:
      vas.append(l2(100,lr,hn)) # the best
    elif ln==3:
      vas.append(l3(100,lr,hn))
    else:
      vas.append(l1(100,lr,hn))

  for i in range(0,total_run):
    print ("The case with learning rate "+ str(learning_r[i])+", layer number "+str(layer_n[i])
            +", hidden number "+str(hidden_n[i])+", has validating accuracy: "+str(vas[i]))
