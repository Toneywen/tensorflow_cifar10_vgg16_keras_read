import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import tensorflow as tf

class CNN:
    def __init__(self):              
        # Network parameters
        self.Num_colorChannel = 3
        self.learning_rate = 0.001
        self.img_size = 32
        self.output_graph = True
        
        # start session
        self.sess = tf.Session()
        
        # set placeholders and build network
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, 3], name="image")
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name="classes")
        self._build_net(self.x)

        # Set Loss
        cross_entropy = tf.losses.softmax_cross_entropy(self.y_, self.predict, reduction=tf.losses.Reduction.MEAN)
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.loss = cross_entropy + l2 * 0.0001
        self.pred = tf.argmax(self.predict, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, tf.argmax(self.y_, axis=1)), tf.float32))
     
        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, momentum=.9).minimize(self.loss)

        if self.output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self, batch_images):
        with tf.name_scope("CNN"):
            with tf.name_scope("conv1"):
                self.conv1_1 = tf.layers.conv2d(batch_images, filters=32, kernel_size=3, activation=tf.nn.relu, padding='same', name="conv1_1")
                self.conv1_2 = tf.layers.conv2d(self.conv1_1, filters=32, kernel_size=3, activation=tf.nn.relu, padding='same', name="conv1_2")
                self.pool1 = tf.layers.max_pooling2d(self.conv1_2, 2, 2, name="pool_1")

            # conv2
            with tf.name_scope("conv2"):
                self.conv2_1 = tf.layers.conv2d(self.pool1, filters=64, kernel_size=3, activation=tf.nn.relu, padding="same", name="conv2_1")
                self.conv2_2 = tf.layers.conv2d(self.conv2_1, filters=64, kernel_size=3, activation=tf.nn.relu, padding="same", name="conv2_2")
                self.pool2 = tf.layers.max_pooling2d(self.conv2_2, 2, 2, name="pool_2")

            # conv3
            with tf.name_scope("conv3"):
                self.conv3_1 = tf.layers.conv2d(self.pool2, filters=128, kernel_size=3, activation=tf.nn.relu,
                                                padding="same", name="conv3_1")
                self.conv3_2 = tf.layers.conv2d(self.conv3_1, filters=128, kernel_size=3,
                                                activation=tf.nn.relu, padding="same", name="conv3_2")
                self.pool3 = tf.layers.max_pooling2d(self.conv3_2, 2, 2, name="pool_3")

            # conv4
            with tf.name_scope("conv4"):
                self.conv4_1 = tf.layers.conv2d(self.pool3, filters=256, kernel_size=3, activation=tf.nn.relu,
                                                padding="same", name="conv4_1")
                self.conv4_2 = tf.layers.conv2d(self.conv4_1, filters=256, kernel_size=3,
                                                activation=tf.nn.relu, padding="same", name="conv4_2")
                self.pool4 = tf.layers.max_pooling2d(self.conv4_2, 2, 2, name="pool_4")

            # FC
            with tf.name_scope("FC"):
                self.flatten = tf.layers.flatten(self.pool4, name="flatten")
                self.FC2 = tf.layers.dense(self.flatten, 1024, activation=tf.nn.relu, name="FC2")
                self.predict = tf.layers.dense(inputs=self.FC2, units=10)

    # if you want to save the params to your pc, you can call this function
    def save_net(self):
        saver = tf.train.Saver()
        saver.save(self.sess, "./params", write_meta_graph=False)
    # you can use it after initializaing the net to test new images
    def restore_net(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './params')
    # train the net and return loss and accuracy
    def learn(self, x_batch, y_batch):
        _, loss, accuracy = self.sess.run([self.train_op, self.loss, self.accuracy], feed_dict={self.x: x_batch, self.y_: y_batch})
        return loss, accuracy


def main():
    # Datasets
    train_acc = []
    train_loss = []
    # the param about normalization from Imagenet
    mean = np.array([123.68, 116.779, 103.979]).reshape((1, 1, 1, 3))

    # load data. may you need download it
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # call this func to make labels to onehot
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    # data generate
    train_gen = ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
    ).flow(x_train, y_train, batch_size=32)
    
    test_gen = ImageDataGenerator().flow(x_test, y_test, batch_size=32)
    # Instantiated the object of CNN
    testnet = CNN()
    for epoch in range(200):
        print('epochs:', epoch)
        # train_one_batch also can accept your own session
        for iter in range(50000 // 32):
            images, labels = train_gen.next()
            images = images - mean
            loss, acc = testnet.learn(images, labels)
            train_acc.append(acc)
            train_loss.append(loss)
        meanacc = np.mean(train_acc)
        meanloss = np.mean(train_loss)
        print('mean acc', meanacc, 'mean loss', meanloss)


if __name__ == "__main__":
        main()

