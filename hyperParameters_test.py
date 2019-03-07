import tensorflow as tf
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator

# hyper-parameters：
Momentum = 0.9
Start_learning_rate = 0.01
Augmentation = False
num_classes = 10
frequency = 20
divisor = 2

if num_classes==10:
    from keras.datasets import cifar10
else:
    from keras.datasets import cifar100


class CNN:
    def __init__(self):
        self.learning_rate = Start_learning_rate
        self.num_channels = 3
        self.num_classes = num_classes
        self.image_size = 32
        self.output_graph = True

        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, self.num_channels], name="batch_images")
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="labels")
        self._build_net(self.x)
        cross_entropy = tf.losses.softmax_cross_entropy(self.y_, self.predict, reduction=tf.losses.Reduction.MEAN)
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        # 如果l2不加系数，那么正确率就一直停留在0.1或者0.01(cifar100)
        self.loss = cross_entropy + l2*0.0001
        self.pred = tf.argmax(self.predict, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, tf.argmax(self.y_, axis=1)), tf.float32))
        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, momentum=Momentum).minimize(self.loss)

        if self.output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())


    def _build_net(self, batch_images):
        with tf.name_scope("CNN"):
            with tf.name_scope("conv1"):
                self.conv1_1 = tf.layers.conv2d(batch_images, filters=32, kernel_size=3, activation=tf.nn.relu, name="conv1_1", padding="Same")
                self.conv1_2 = tf.layers.conv2d(self.conv1_1, filters=32, kernel_size=3, activation=tf.nn.relu, name="conv1_2", padding="Same")
                self.pool_1 = tf.layers.max_pooling2d(self.conv1_2, 2, 2, name="pool_1")

            with tf.name_scope("conv2"):
                self.conv2_1 = tf.layers.conv2d(self.pool_1, filters=64, kernel_size=3, activation=tf.nn.relu, name="conv2_1", padding="Same")
                self.conv2_2 = tf.layers.conv2d(self.conv2_1, filters=64, kernel_size=3, activation=tf.nn.relu, name="conv2_2", padding="Same")
                self.pool_2 = tf.layers.max_pooling2d(self.conv2_2, 2, 2, name="pool_2")

            with tf.name_scope("conv3"):
                self.conv3_1 = tf.layers.conv2d(self.pool_2, filters=128, kernel_size=3, activation=tf.nn.relu, name="conv3_1", padding="Same")
                self.conv3_2 = tf.layers.conv2d(self.conv3_1, filters=128, kernel_size=3, activation=tf.nn.relu, name="conv3_2", padding="Same")
                self.pool_3 = tf.layers.max_pooling2d(self.conv3_2, 2, 2, name="pool_3")

            with tf.name_scope("conv4"):
                self.conv4_1 = tf.layers.conv2d(self.pool_3, filters=256, kernel_size=3, activation=tf.nn.relu, name="conv4_1", padding="Same")
                self.conv4_2 = tf.layers.conv2d(self.conv4_1, filters=256, kernel_size=3, activation=tf.nn.relu, name="conv4_2", padding="Same")
                self.pool_4 = tf.layers.max_pooling2d(self.conv4_2, 2, 2, name="pool_4")

            with tf.name_scope("FC"):
                self.flatten = tf.layers.flatten(self.pool_4, "flatten")
                self.fc = tf.layers.dense(self.flatten, 256, activation=tf.nn.relu, name="fc")
                self.predict = tf.layers.dense(self.fc, num_classes, name="predict")


    def learn(self, x_batch, y_batch, learning_rate):

        self.learning_rate = learning_rate
        _, loss, accuracy = self.sess.run([self.train_op, self.loss, self.accuracy], feed_dict={self.x:x_batch, self.y_:y_batch})
        return loss, accuracy


def main():
    normalization_mean = np.array([123.68, 116.779, 103.979]).reshape((1, 1, 1, 3))
    train_acc = []
    train_loss = []
    if num_classes == 100:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train - normalization_mean
    if Augmentation:
        train_gen = ImageDataGenerator( horizontal_flip=True,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        shear_range=0.1,
                                        zoom_range=0.1,).flow(x_train, y_train, batch_size=32)
    else:
        train_gen = ImageDataGenerator().flow(x_train, y_train, batch_size=32)

    test_gen = ImageDataGenerator().flow(x_test, y_test, batch_size=32)
    net = CNN()
    epochs = 200
    learning_rate = Start_learning_rate
    print("y_train.shape[0]:", y_train.shape[0])
    for i in range(epochs):
        print("epochs:", i)
        if (i+1) % frequency == 0:
            learning_rate /= divisor 
            print("**×*×*×*×*×*×*×*×*×*×*×*×*×*learning_rate:", learning_rate)
        for t in range(y_train.shape[0]//32):
            x_batch, y_batch = train_gen.next()
            loss, accuracy = net.learn(x_batch, y_batch, learning_rate)
            train_acc.append(accuracy)
            train_loss.append(loss)
        meanacc = np.mean(train_acc)
        meanloss = np.mean(train_loss)
        print('mean acc', meanacc, 'mean loss', meanloss)

if __name__=="__main__":
    main()
