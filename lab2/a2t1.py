import numpy as np
import tensorflow as tf

with np.load("notMNIST.npz") as data:
    images, labels = data["images"], data["labels"]

al=28*28
samplen=18720
labeln=10
images=images.T
labels=np.eye(10)[labels[:,0]]
#print np.shape(images)

images=images.reshape(samplen,al)

print np.shape(images)
print np.shape(labels)

x_train=images[0:15000]
t_train=labels[0:15000]
x_eval=images[15001:16000]
t_eval=labels[15001:16000]

#print np.shape(x_train)

X=tf.placeholder("float",shape=(None,al))
Y=tf.placeholder("float",shape=(None,10))

W=tf.Variable(np.random.randn(al,labeln).astype("float32"),name="weight")
b=tf.Variable(np.random.randn(1,labeln).astype("float32"),name="bias")

logits=tf.add(tf.matmul(X,W),b)
output=tf.nn.softmax(logits)

cost_batch=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y)
cost=tf.reduce_mean(cost_batch)

optimizer=tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.01)
train_op=optimizer.minimize(cost)

pred=tf.argmax(output,1)
pred_float=tf.cast(pred,"float")

correct_prediction=tf.equal(pred_float,tf.cast(tf.argmax(Y,1),"float"))
accuracy=tf.reduce_sum(tf.cast(correct_prediction,"float"))
# accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))


sess=tf.InteractiveSession()

init=tf.initialize_all_variables()
sess.run(init)

for epoch in range(1000):
  for i in xrange(75):
    x_batch = x_train[i * 200: (i + 1) * 200]
    y_batch = t_train[i * 200: (i + 1) * 200]
    cost_np, _ = sess.run([cost, train_op],
                          feed_dict={X: x_batch, Y: y_batch})
    #Display logs per epoch step
  if epoch % 50 == 0:
    cost_train, accuracy_train = sess.run([cost, accuracy],
                                          feed_dict={X: x_train, Y: t_train})
    cost_eval, accuracy_eval = sess.run([cost, accuracy],
                                                   feed_dict={X: x_eval, Y: t_eval})
    print ("Epoch:%04d, cost=%0.9f, Train Accuracy=%0.4f, Eval Accuracy=%0.4f" %
           (epoch+1, cost_train, accuracy_train, accuracy_eval))

#print entropy(1000,1)
#print entropy(-1000,0)
#print entropy(1000,0)
#print entropy(-1000,1)

# tf.Session().run()
