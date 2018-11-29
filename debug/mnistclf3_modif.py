import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from assist import DataSet
import matplotlib.pyplot as plt
import time #重要的评价包
import os
import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from sklearn.base import BaseEstimator
from confidentlearning.classification import RankPruning
import gc

#keynotes(from the rp module->classification:description of the clf para.)
# clf : sklearn.classifier or equivalent class
#       The clf object must have the following three functions defined:
#       1. clf.predict_proba(X) # Predicted probabilities
#       2. clf.predict(X) # Predict labels
#       3. clf.fit(X, y, sample_weight) # Train classifier
#       Stores the classifier used in Rank Pruning.
#       Default classifier used is logistic regression.


class cnn(BaseEstimator):

    def __init__(self,epochs=None, datapath=None,noisy=False,ratio=None):
        self.graph = self.createModel()
        self.mnistdata = self.dataLoaderConcise(datapath)

        if epochs == None:
            self.epochs = 10
        else:
            self.epochs = epochs
        self.noisymark = noisy
        if self.noisymark:
            self.nosiyLabels(ratio)
        else:
            pass

    def dataLoaderConcise(self, datapath=None):
        if datapath is None:
            datapath = '/home/zhaok14/example/PycharmProjects/cnn_scratch/MNIST_data'
        else:
            pass
        # load mnist data
        return input_data.read_data_sets(datapath, one_hot=True)

    #2. define the cnn micro-architecture; 2+2 structure
    def createModel(self):
        graph1 = tf.Graph()
        with graph1.as_default():

            self.input_data = tf.placeholder(dtype=tf.float32,shape=[None, 784])
            self.label = tf.placeholder(dtype=tf.float32, shape=[None, 10])
            self.weights = tf.placeholder(dtype=tf.float32, shape=[None,])
            input_data= tf.reshape(self.input_data, [-1, 28, 28, 1])

            conv1 = tf.layers.conv2d(inputs=input_data,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            self.pred = tf.layers.dense(inputs=dense1, units=10)

            self.loss = tf.losses.softmax_cross_entropy(self.label, self.pred,self.weights)
            self.optimize = tf.train.AdadeltaOptimizer(learning_rate=0.01, rho=0.95, epsilon=1e-06).minimize(self.loss)

            self.prob = tf.nn.softmax(self.pred, name="softmax_tensor")

            self.saver = tf.train.Saver(max_to_keep=1)

        return graph1

    def dataPackaging(self):
        while True:
            yield 1

    def fit(self, X, y, sample_weight=None):
        #1. choose the training data according to x
        #2. check if y is right
        #3. attach sample_weight to the training data
        assert(len(X) == len(y))
        indices = X.sum(axis=1)
        images = self.mnistdata.train._images[indices]
        labels = self.mnistdata.train._labels[indices]
        option = dict(dtype=np.float32, reshape=True)
        subset = DataSet(images,labels,**option)
        #2. weights matching
        subset.plugin(sample_weight)
        #3. training
        epochs = self.epochs
        minibatch_size = 128
        print('training data size:', subset.num_examples)
        iterations_per_epoch = len(self.mnistdata.train._images) // minibatch_size + 1
        with tf.Session(graph=self.graph) as self.sess:
            self.sess.run(tf.global_variables_initializer()) #or load weight!!!
            genor = self.dataPackaging()
            pbar = tqdm.tqdm(genor)
            for i in range(epochs):
                iteration = 0
                for _ in pbar:
                    batch_x, batch_y, weights = subset.next_batch_weighting(minibatch_size)
                    feed = {self.input_data: batch_x, self.label: batch_y, self.weights:weights}
                    _, loss= self.sess.run([self.optimize, self.loss], feed_dict=feed)
                    pr = 'epoch:%d/%d,iteration: %d/%d ,loss: %s' % (epochs, i, iterations_per_epoch, iteration, loss)
                    pbar.set_description(pr)
                    if iteration == iterations_per_epoch:
                        break
                    else:
                        iteration += 1 #in fact train more than the one iteration
            pbar.close()
            if self.noisymark:
                self.saver.save(self.sess, os.path.join(os.getcwd(),  'mnistclf3_modif', 'speechNoisy.module'), global_step=epochs)
            else:
                self.saver.save(self.sess, os.path.join(os.getcwd(), 'mnistclf3_modif', 'speech.module'), global_step=epochs)

    def predict(self, X):
        # get the index of the max probability
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)

    def predict_proba(self, X, ):
        # form the subset
        indices = X.sum(axis=1)
        images = self.mnistdata.train._images[indices]
        # load the graph and net-weights
        if self.noisymark:
            strss = './mnistclf3_modif/speechNoisy.module-'+str(self.epochs)
        else:
            strss = './mnistclf3_modif/speech.module-'++str(self.epochs)
        resaver = tf.train.import_meta_graph(strss + '.meta')  # 载入图结构，保存在.meta文件中
        with tf.Session(graph=self.graph) as self.sess:
            resaver.restore(self.sess, strss)  # 载入参数，参数保存在两个文件中，不过restore会自己寻找
            y_prob = self.sess.run(self.prob, feed_dict={self.input_data:images})
            return y_prob

    def compute_accuracy(self, v_x, v_y, str=None):
        if str == None:
            ckpt = tf.train.get_checkpoint_state('./mnistclf1_mini/')  # 通过检查点文件锁定最新的模型
            print('checking if the model is correctly used:',ckpt.model_checkpoint_path)
            new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
        else:
            new_saver = tf.train.import_meta_graph(str + '.meta')  # 载入图结构，保存在.meta文件中
        weights = np.ones((len(v_x),), dtype=np.float32)
        with tf.Session(graph=self.graph) as self.sess:
            new_saver.restore(self.sess, ckpt.model_checkpoint_path if (str==None) else str) # 载入参数，参数保存在两个文件中，不过restore会自己寻找
            feed = {self.input_data: v_x, self.label: v_y, self.weights: weights}
            y_pre = self.sess.run(self.pred, feed_dict=feed)
        correct_prediction = np.equal(np.argmax(y_pre, 1), np.argmax(v_y, 1))
        accuracy = np.mean(correct_prediction)
        accuracy = round(accuracy,4)
        return accuracy

    def test(self):
        if self.noisymark:
            value = self.compute_accuracy(self.mnistdata.test.images, self.mnistdata.test.labels,str='./mnistclf3_modif/speechNoisy.module-'+str(self.epochs)) * 100
        else:
            value = self.compute_accuracy(self.mnistdata.test.images, self.mnistdata.test.labels,str='./mnistclf3_modif/speech.module-'+str(self.epochs)) * 100
        print("accuracy:", value, '% ')
        return value

    def nosiyLabels(self,ratio=None):
        # print(mnist.train._labels)
        # print(type(mnist.train._labels))
        prop = 1
        if ratio is not None:
            prop = ratio
        temp = self.mnistdata.train._labels
        print('noisy proption:',ratio)
        t = int(temp.shape[0]*prop-1)
        for i in range(t):
            # print('original:',temp[t,:])
            # print('ing:',np.roll(temp[t,:],2))
            temp[i,:] = np.roll(temp[i,:],2)
            # print('finals:',temp[i,:])
        self.mnistdata.train._labels = temp


if (__name__ == '__main__'):

    #just to get the size of training data.
    trial=cnn(epochs = 10,datapath=None,noisy=True)
    counts = trial.mnistdata.train.num_examples

    clfaccu = []
    for i in range(11):
        clf = cnn(epochs=10, datapath=None, noisy=True,ratio=i/10)
        idx = np.arange(counts).reshape((counts, 1))
        indices = idx.sum(axis=1)
        chus = clf.mnistdata.train._labels[indices]
        chus = np.argmax(chus, axis=1)

        clf.fit(idx,chus)
        clfaccu.append(clf.test())
        os.system('rm -rf ./mnistclf3_modif/* ')


    gc.collect()
    rpaccu = []
    for i in range(11):
        rp = RankPruning(clf=cnn(epochs = 10,datapath=None,noisy=True,ratio=i/10))
        idx = np.arange(counts).reshape((counts, 1))
        indices = idx.sum(axis=1)
        chus = rp.clf.mnistdata.train._labels[indices]
        chus = np.argmax(chus, axis=1)

        rp.fit(idx,chus)
        rpaccu.append(rp.clf.test())
        os.system('rm -rf ./mnistclf3_modif/* ')

    x = np.linspace(0, 1, 11)
    plt.plot(x, clfaccu,label='original classifier')
    plt.plot(x, rpaccu,label='rank pruning classifier')

    plt.xlabel('nosiy ratio')
    plt.ylabel('accuracy/%')

    plt.legend()

    plt.show()

    print('clfaccu:',clfaccu)
    print('rpaccu',rpaccu)