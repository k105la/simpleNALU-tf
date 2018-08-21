# ## Neural Arithmatic Logic Units
# Google DeepMind's research paper: https://arxiv.org/abs/1808.00508


import tensorflow as tf
import numpy as np



# The Neural Arithmetic Logic Unit
def NALU(in_dim, out_dim):

    shape = (int(in_dim.shape[-1]), out_dim)
    epsilon = 1e-7 
    
    # NAC
    W_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    M_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    G = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
        
    W = tf.tanh(W_hat) * tf.sigmoid(M_hat)
    # Forward propogation
    a = tf.matmul(in_dim, W)
    
    # NALU  
    m = tf.exp(tf.matmul(tf.log(tf.abs(in_dim) + epsilon), W))
    g = tf.sigmoid(tf.matmul(in_dim, G))
    y = g * a + (1 - g) * m
    
    return y


### Helper Function

def generate_dataset(size=10000):
    # input data
    X = np.random.randint(9, size=(size,2))
    # output data (labels)   
    Y = np.prod(X, axis=1, keepdims=True)

        
    return X, Y


### Train NALU on generated data

# Hyperparameters
EPOCHS = 200
LEARNING_RATE = 1e-3
BATCH_SIZE = 10



# create dataset
X_data, Y_data = generate_dataset()


# define placeholders and network
X = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 2])

Y_true = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1])

Y_pred = NALU(X, 1)

loss = tf.nn.l2_loss(Y_pred - Y_true) 
tf.summary.histogram('loss', loss) # Loss summary

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


# create session
sess = tf.Session()
# create writer to store tensorboard graph   
writer = tf.summary.FileWriter('/tmp', sess.graph)

summaries = tf.summary.merge_all()
saver = tf.train.Saver() # Add ops to save and restore all the variables.

init = tf.global_variables_initializer()
    
sess.run(init)

# Run training loop
for i in range(EPOCHS):
    j = 0
    g = 0
        
    while j < len(X_data):
        xs, ys = X_data[j:j + BATCH_SIZE], Y_data[j:j + BATCH_SIZE]

        _, summary, ys_pred, l = sess.run([optimizer, summaries, Y_pred, loss], 
                    feed_dict={X: xs, Y_true: ys})
        
        writer.add_summary(summary, i)        
        # calculate number of correct predictions from batch
        g += np.sum(np.isclose(ys, ys_pred, atol=1e-4, rtol=1e-4)) 

        j += BATCH_SIZE

    acc = g / len(Y_data)
        
    print(f'epoch {i}, loss: {l}, accuracy: {acc}')
    # Save model checkpoints.
    save_path = saver.save(sess, 'tmp/model.ckpt') 

print(f'Model saved in path: {save_path}')
