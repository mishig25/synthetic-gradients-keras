import random
import tensorflow as tf

from tqdm import tqdm # Used to display training progress bar

from keras import backend as K
from keras.layers import Dense, BatchNormalization, Activation

sess = tf.Session()
K.set_session(sess)

class Layer():
    
    def __init__(self, units, inputs, name, out=False, sg=False):
        self.name = name
        with tf.variable_scope(self.name):
            if sg:
                inputs_c = K.concatenate(inputs, 1)
                self.output = Dense(units, kernel_initializer=tf.zeros_initializer())(inputs_c)
            else:
                self.output = Dense(units)(inputs)
                if out: self.output = Activation('relu')(BatchNormalization()(self.output))
        self.layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.name)


class ModelSG():
    
    def __init__(self, sess):
        self.sess = sess
        # self.dataset = input_data.read_data_sets("data/", one_hot=True)
        self.dataset = tf.contrib.learn.datasets.mnist.read_data_sets("data/", one_hot=True)

        self.lr_div = 10
        self.lr_div_steps = set([300000, 400000])

        self.create_layers()

    def create_layers(self):
        # Inputs
        X = tf.placeholder(tf.float32, shape=(None, 784), name="data") # Input
        Y = tf.placeholder(tf.float32, shape=(None, 10), name="labels") # Target
        self.inputs = [X,Y]

        # inference
        layer1 = Layer(256,X,'layer1')
        layer2 = Layer(256,layer1.output,'layer2')
        layer3 = Layer(256,layer2.output,'layer3')
        logits = Layer(10,layer3.output,'layer4',out=True)
        self.layers = [layer1,layer2,layer3,logits]

        # sg layers
        synth_layer1 = Layer(256, [layer1.output,Y], 'sg2', sg=True)
        synth_layer2 = Layer(256, [layer2.output,Y], 'sg3', sg=True)
        synth_layer3 = Layer(256, [layer3.output,Y], 'sg4', sg=True)
        self.synth_layers = [synth_layer1,synth_layer2,synth_layer3]
    
    def train_layer_n(self, h_m, h_n, d_hat_m, class_loss, next_l, d_n=None, p=True):
        if d_n is not None: d_n = self.synth_layers[d_n].output
        if p: h_n = self.layers[h_n].output
        with tf.variable_scope(self.layers[h_m].name):
            layer_grads = tf.gradients(h_n, [self.layers[h_m].output]+self.layers[next_l].layer_vars, d_n)
            layer_gv = list(zip(layer_grads[1:],self.layers[next_l].layer_vars))
            layer_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).apply_gradients(layer_gv)
        with tf.variable_scope(self.synth_layers[d_hat_m].name):
            d_m = layer_grads[0]
            sg_loss = tf.divide(tf.losses.mean_squared_error(self.synth_layers[d_hat_m].output, d_m), class_loss)
            sg_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(sg_loss, var_list=self.synth_layers[d_hat_m].layer_vars)
        return layer_opt, sg_opt
    
    def prepare_training(self, learning_rate):
        self.learning_rate = tf.Variable(learning_rate, dtype=tf.float32, name="lr")
        self.reduce_lr = tf.assign(self.learning_rate, self.learning_rate/self.lr_div, name="lr_decrease")

        self.pred_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.inputs[1], logits=self.layers[3].output, scope="prediction_loss")

        layer4_opt, sg4_opt = self.train_layer_n(2, self.pred_loss, 2, self.pred_loss, 3, p=False)
        layer3_opt, sg3_opt = self.train_layer_n(1, 2, 1, self.pred_loss, 2, 2)
        layer2_opt, sg2_opt = self.train_layer_n(0, 1, 0, self.pred_loss, 1, 1)

        layer1_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.layers[0].output, var_list=self.layers[0].layer_vars, 
                                            grad_loss=self.synth_layers[0].output)

        self.decoupled_training = [[layer1_opt],[layer2_opt, sg2_opt],
                    [layer3_opt, sg3_opt],[layer4_opt, sg4_opt]]

    def train(self, iterations, batch_size, update_prob, learning_rate):
        self.prepare_training(learning_rate)
        with self.sess.as_default():
            init = tf.global_variables_initializer()
            self.sess.run(init)
            for i in tqdm(range(1,iterations+1)):
                if i in self.lr_div_steps: self.sess.run(self.reduce_lr) # Decrease learning rate
                
                data, target = self.dataset.train.next_batch(batch_size)
                X,Y = self.inputs[0], self.inputs[1]
                
                for i in self.decoupled_training: 
                    if random.random() <= update_prob: self.sess.run(i, feed_dict={X:data,Y:target})
    
    def test(self, batch_size):
        X,Y = self.inputs[0], self.inputs[1]
        preds = tf.nn.softmax(self.layers[3].output, name="predictions")
        correct_preds = tf.equal(tf.argmax(preds,1), tf.argmax(Y,1), name="correct_predictions")
        accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.float32), name="correct_prediction_count") / batch_size
        with self.sess.as_default():
            n_batches = int(self.dataset.test.num_examples/batch_size)
            test_accuracy = 0
            test_loss = 0
            for _ in range(n_batches):
                Xb, Yb = self.dataset.test.next_batch(batch_size)
                batch_accuracy, batch_loss = self.sess.run([accuracy, self.pred_loss], feed_dict={X:Xb,Y:Yb})
                test_loss += batch_loss
                test_accuracy += batch_accuracy
            print ('Validtion on test sample |','Loss:',test_loss/n_batches,'Accuracy:',test_accuracy/n_batches)

'''
1. shrink them even more
2. add validity and tensorboard
'''

model = ModelSG(sess)
model.create_layers()
model.train(500000, 250, 0.2, 3e-5)
model.test(250)