import numpy as np
import tensorflow as tf

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

with np.load("notMNIST.npz") as data:
    images, labels = data["images"], data["labels"]

image_size=28
num_channels=1
num_labels=10

batch_size=50

print ("Stamp 6")

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


hidden_num=1000

graph = tf.Graph()



with graph.as_default():
  x_train=tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
  y_train=tf.placeholder(tf.float32,shape=(batch_size,num_labels))


  # layer 1
  w=tf.Variable(tf.truncated_normal([image_size*image_size,num_labels*hidden_num]))
  b=tf.Variable(tf.truncated_normal([num_labels*hidden_num]))

  # layer 2
  w2=tf.Variable(tf.truncated_normal([num_labels*hidden_num,10]))
  b2=tf.Variable(tf.truncated_normal([num_labels]))



  #print x_train.get_shape()
  logits=tf.matmul(x_train,w)
  logits=tf.add(logits,b)
  #print logits.get_shape()
  #logits=tf.nn.relu(tf.reshape(logits,[batch_size,num_labels,hidden_num])+b)
  logits=tf.nn.relu(logits)
  #print logits.get_shape()
  #print w2.get_shape()
  #print b2.get_shape()
  #print logits.get_shape()
  logits=tf.matmul(logits,w2)
  #print logits.get_shape()
  logits=tf.add(logits,b2)

  #print logits.get_shape()
  #print y_train.get_shape()

  cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,y_train))

  optimizer=tf.train.MomentumOptimizer(0.0001,0.0001).minimize(cost)

  train_prediction=tf.nn.softmax(logits)
  valid_prediction=tf.matmul(valid_dataset,w)+b
  valid_prediction=tf.nn.softmax(tf.matmul(valid_prediction,w2)+b2)
  test_prediction=tf.matmul(test_dataset,w)+b
  test_prediction=tf.nn.softmax(tf.matmul(test_prediction,w2)+b2)

step_num=3000

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print ("initialized")
  for step in range (step_num):
    offset=(step*batch_size)%(train_labels.shape[0]-batch_size)
    x_batch=train_dataset[offset:(offset+batch_size),:]
    y_batch=train_labels[offset:(offset+batch_size),:]


    feed_dict={x_train:x_batch,y_train:y_batch}
    _,l,tp,vp=session.run([optimizer,cost,train_prediction,valid_prediction],feed_dict=feed_dict)

    if (step%100==0):
      print ("Minibatch loss at step %d: %f" %(step,l))
      print("Minibatch accuracy: %.1f%%" % accuracy(tp,y_batch))
      print("Validation accuracy: %.1f%%" % accuracy(vp,valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(),test_labels))