import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import time #重要的评价包
import os
import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from confidentlearning.classification import RankPruning
from sklearn.linear_model import LogisticRegression as logreg


class cnn():

    def __init__(self,epochs=None):
        self.graph = self.createModel()
        self.load = False
        if epochs == None:
            self.epochs = 10
        else:
            self.epochs = epochs

    def dataLoaderConcise(self, path=None):
        if path is None:
            path = '/home/zhaok14/example/PycharmProjects/cnn_scratch/MNIST_data'
        else:
            pass
        # load mnist data
        return input_data.read_data_sets(path, one_hot=True)

    #2. define the cnn micro-architecture; 2+2 structure
    def createModel(self):
        graph1 = tf.Graph()
        with graph1.as_default():

            self.input_data = tf.placeholder(dtype=tf.float32,shape=[None, 784])
            self.label = tf.placeholder(dtype=tf.float32, shape=[None, 10])
            input_data= tf.reshape(self.input_data, [-1, 28, 28, 1])

            conv1 = tf.layers.conv2d(inputs=input_data,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            self.pred = tf.layers.dense(inputs=dense1, units=10)

            self.loss = tf.losses.softmax_cross_entropy(self.label, self.pred)
            self.optimize = tf.train.AdadeltaOptimizer(learning_rate=0.01, rho=0.95, epsilon=1e-06).minimize(self.loss)

            self.saver = tf.train.Saver(max_to_keep=1)

        return graph1

    def dataPackaging(self):
        while True:
            yield 1

    def trainFit(self,mnist,noisyLabels=False):

        if noisyLabels:
            print('training with nosiy labels!')
        epochs = self.epochs
        minibatch_size = 128
        print('training data size:',len(mnist.train._images))
        iterations_per_epoch = len(mnist.train._images)//minibatch_size + 1
        with tf.Session(graph=self.graph) as self.sess:
            self.sess.run(tf.global_variables_initializer()) #or load weight!!!
            if self.load == True:
                self.load = False

            genor = self.dataPackaging()
            pbar = tqdm.tqdm(genor)
            for i in range(epochs):
                iteration = 0
                for _ in pbar:
                    batch_x, batch_y = mnist.train.next_batch(minibatch_size)
                    feed = {self.input_data: batch_x, self.label: batch_y}
                    _, loss= self.sess.run([self.optimize, self.loss], feed_dict=feed)
                    pr = 'epoch:%d/%d,iteration: %d/%d ,loss: %s' % (epochs, i, iterations_per_epoch, iteration, loss)
                    pbar.set_description(pr)
                    if iteration == iterations_per_epoch:
                        break
                    else:
                        iteration += 1 #in fact train more than the one iteration
            pbar.close()
            if noisyLabels:
                self.saver.save(self.sess, os.path.join(os.getcwd(),  'mnistclf1_mini', 'speechNoisy.module'), global_step=epochs)
            else:
                self.saver.save(self.sess, os.path.join(os.getcwd(), 'mnistclf1_mini', 'speech.module'), global_step=epochs)

    def compute_accuracy(self, v_x, v_y, str=None):
        if str == None:
            ckpt = tf.train.get_checkpoint_state('./mnistclf1_mini/')  # 通过检查点文件锁定最新的模型
            print('checking if the model is correctly used:',ckpt.model_checkpoint_path)
            new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
        else:
            new_saver = tf.train.import_meta_graph(str + '.meta')  # 载入图结构，保存在.meta文件中
        with tf.Session(graph=self.graph) as self.sess:
            new_saver.restore(self.sess, ckpt.model_checkpoint_path if (str==None) else str) # 载入参数，参数保存在两个文件中，不过restore会自己寻找
            feed = {self.input_data: v_x, self.label: v_y}
            y_pre = self.sess.run(self.pred, feed_dict=feed)
        correct_prediction = np.equal(np.argmax(y_pre, 1), np.argmax(v_y, 1))
        accuracy = np.mean(correct_prediction)
        accuracy = round(accuracy,4)
        return accuracy

    def test(self,mnist):
        print(self.compute_accuracy(mnist.test.images, mnist.test.labels,str='./mnistclf1_mini/speechNoisy.module-10' )*100,'% ')

    def nosiyLabels(self,mnist,ratio=None):
        # print(mnist.train._labels)
        # print(type(mnist.train._labels))
        prop = 1
        if ratio is not None:
            prop = ratio
        temp = mnist.train._labels
        t = int(temp.shape[0]*prop-1)
        for i in range(t):
            # print('original:',temp[t,:])
            # print('ing:',np.roll(temp[t,:],2))
            temp[i,:] = np.roll(temp[i,:],2)
            # print('finals:',temp[i,:])
        mnist.train._labels = temp












