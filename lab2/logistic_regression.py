import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

#def entropy(x,t):
#    return np.maximum(x,0)-x*t+np.log(1+np.exp(-np.abs(x)))

with np.load("TINY_MNIST.npz") as data:
    x_train,t_train=data["x"],data["t"]
    x_eval,t_eval=data["x_eval"],data["t_eval"]

print np.shape(x_train)

X=tf.placeholder("float",shape=(None,64))
Y=tf.placeholder("float",shape=(None,1))

W=tf.Variable(np.random.randn(64,1).astype("float32"),name="weight")
b=tf.Variable(np.random.randn(1).astype("float32"),name="bias")

logits=tf.add(tf.matmul(X,W),b)
output=tf.nn.sigmoid(logits)

cost_batch=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,targets=Y)
cost=tf.reduce_mean(cost_batch)

norm_w=tf.nn.l2_loss(W)

optimizer=tf.train.MomentumOptimizer(learning_rate=1.0,momentum=0.99)
train_op=optimizer.minimize(cost)

pred=tf.greater(output,0.5)
pred_float=tf.cast(pred,"float")

correct_prediction=tf.equal(pred_float,Y)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))


sess=tf.InteractiveSession()

init=tf.initialize_all_variables()
sess.run(init)

for epoch in range(2000):
  for i in xrange(8):
    x_batch = x_train[i * 100: (i + 1) * 100]
    y_batch = t_train[i * 100: (i + 1) * 100]
    cost_np, _ = sess.run([cost, train_op],
                          feed_dict={X: x_batch, Y: y_batch})
    #Display logs per epoch step
  if epoch % 50 == 0:
    cost_train, accuracy_train = sess.run([cost, accuracy],
                                          feed_dict={X: x_train, Y: t_train})
    cost_eval, accuracy_eval, norm_w_np = sess.run([cost, accuracy, norm_w],
                                                   feed_dict={X: x_eval, Y: t_eval})
    print ("Epoch:%04d, cost=%0.9f, Train Accuracy=%0.4f, Eval Accuracy=%0.4f,    Norm of Weights=%0.4f" %
           (epoch+1, cost_train, accuracy_train, accuracy_eval, norm_w_np))

#print entropy(1000,1)
#print entropy(-1000,0)
#print entropy(1000,0)
#print entropy(-1000,1)

# tf.Session().run()
