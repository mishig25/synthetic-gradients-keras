import argparse
import tensorflow as tf
import keras.backend as K

from model import ModelSG

if __name__ == "__main__":

    sess = tf.Session()
    K.set_session(sess)

    ''' Args for hyperparameters '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--iterations", type=int,
                        default=500000, help="Number of Iterations: int")
    parser.add_argument("-B", "--batch", type=int,
                        default=250, help="Batch Size: int")
    parser.add_argument("-P", "--update_prob", type=float, default=0.2,
                        help="Synthetic Grad Update Probability: float [0,1]")
    parser.add_argument("-L", "--l_rate", type=float,
                        default=3e-5, help="Learning Rate: float")
    args = parser.parse_args()

    ''' Training and Testing the Model '''
    model = ModelSG(sess)
    model.create_layers()
    model.train(args.iterations, args.batch,
                args.update_prob, args.l_rate)
    model.test(args.batch)

