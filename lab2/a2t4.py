import numpy as np
import tensorflow as tf

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

def a2t2(batch_size,learning_rate,hidden_num):
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
    w1=tf.Variable(tf.truncated_normal([image_size*image_size,num_labels*hidden_num]))
    b1=tf.Variable(tf.truncated_normal([num_labels*hidden_num]))

    # layer2
    w2=tf.Variable(tf.truncated_normal([num_labels*hidden_num,num_labels*hidden_num]))
    b2=tf.Variable(tf.truncated_normal([num_labels*hidden_num]))

    # layer 3
    w3=tf.Variable(tf.truncated_normal([num_labels*hidden_num,num_labels]))
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

    optimizer=tf.train.MomentumOptimizer(learning_rate,learning_rate/10).minimize(cost)

    train_prediction=tf.nn.softmax(logits)
    valid_prediction=tf.add(tf.matmul(valid_dataset,w1),b1)
    valid_prediction=tf.add(tf.matmul(valid_prediction,w2),b2)
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

      if (step%10==0):
        print ("Minibatch loss at step %d: %f" %(step,l))
        #print("Minibatch accuracy: %.1f%%" % accuracy(tp,y_batch))
        va.append(accuracy(vp,valid_labels))
        ta.append(accuracy(tp,test_labels))
        if len(va)>5 and va[-1]<va[-2] and va[-1]<va[-3]:
          va.pop(1)
          ta.pop(1)
          break

        #print("Validation accuracy: %.1f%%" % accuracy(vp,valid_labels))
    print("Test accuracy: %.1f%%" % ta[-1])
    return va

if __name__=="__main__":
  print ("Stamp 1")
  vas=[]



  vas.append(a2t2(100,0.000001,500)) # the best


  print vas