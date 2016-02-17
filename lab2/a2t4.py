import numpy as np
import tensorflow as tf

def accuracy(predictions, labels):
  predictions=tf.argmax(predictions, 1)
  labels=tf.argmax(labels, 1)
  return tf.reduce_sum(tf.cast(tf.not_equal(predictions,labels),"float32"))

def a2t4(batch_size,learning_rate,hidden_num):
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

      print("Epoch %03d, Training Error Number: %.1f, Validation Error Number: %.1f" % (step,tp,vp))

    feed_dict={x_train:test_dataset,y_train:test_labels}
    _,l,tp=session.run([optimizer,cost,er], feed_dict=feed_dict)
    print("Test Error Number: %.1f" % tp)
    return validate_error

if __name__=="__main__":
  vas=[]



  vas.append(a2t4(100,0.01,500)) # the best



  print vas