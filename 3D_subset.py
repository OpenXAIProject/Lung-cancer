#Copyright 2018 (Institution) under XAI Project supported by Ministry of Science and ICT, Korea

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#   https://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append(".")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from modules.sequential import Sequential
from modules.linear import Linear
from modules.softmax import Softmax
from modules.relu import Relu
from modules.tanh import Tanh

from modules.convolution3D import Convolution3D
from modules.maxpool3D import MaxPool3D
from modules.convolution import Convolution
from modules.avgpool import AvgPool
from modules.maxpool import MaxPool
from modules.utils import Utils, Summaries, plot_relevances
import modules.render as render
import tensorflow as tf
import numpy as np
import pdb
from scipy.ndimage import rotate
import tflearn
import h5py
import itertools
from preprocessing import ImageDataGenerator
flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 20, 'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 100, 'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.0001, 'Initial learning rate')
flags.DEFINE_string("summaries_dir", 'convolutional_logs', 'Summaries directory')
flags.DEFINE_boolean("relevance", False, 'Compute relevances')
flags.DEFINE_string("relevance_method", 'alphabeta', 'relevance methods: simple/eps/w^2/alphabeta')
flags.DEFINE_boolean("save_model", True, 'Save the trained model')
flags.DEFINE_boolean("reload_model", True, 'Restore the trained model')
flags.DEFINE_integer("Class", 2, 'Number of class.')

FLAGS = flags.FLAGS
def nn(phase):
    return Sequential(
        [Convolution3D(kernel_size=3, output_depth=32, input_depth=1, batch_size=FLAGS.batch_size, input_dim=32,
                     act='lrelu', phase = phase, stride_size=1, pad='SAME'),
         Convolution3D(kernel_size=3, output_depth=32, input_depth=32, batch_size=FLAGS.batch_size,
                     act='lrelu', phase = phase, stride_size=1, pad='SAME'),
         MaxPool3D(),

         Convolution3D(kernel_size=3, output_depth=64, input_depth=32, batch_size=FLAGS.batch_size,
                     act='lrelu', phase = phase, stride_size=1, pad='SAME'),
         Convolution3D(kernel_size=3, output_depth=64, input_depth=64, batch_size=FLAGS.batch_size,
                     act='lrelu', phase = phase, stride_size=1, pad='SAME'),
         MaxPool3D(),

         Convolution3D(kernel_size=3, output_depth=128, input_depth=64, batch_size=FLAGS.batch_size,
                     act='lrelu', phase = phase, stride_size=1, pad='SAME'),
         Convolution3D(kernel_size=3, output_depth=128, input_depth=64, batch_size=FLAGS.batch_size,
                     act='lrelu', phase = phase, stride_size=1, pad='SAME'),
         MaxPool3D(),
         Convolution3D(kernel_size=4, output_depth=128, stride_size=1, act='lrelu', phase = phase, pad='VALID'),
         Convolution3D(kernel_size=1, output_depth=2, stride_size=1, phase = phase, final = True, pad='VALID')
         ])


def visualize(relevances, images_tensor):
    n = FLAGS.batch_size
    heatmap = relevances.reshape([n, 50, 50, 1])
    input_images = images_tensor.reshape([n, 50, 50, 1])
    heatmaps = []
    for h, heat in enumerate(heatmap):
        input_image = input_images[h]
        maps = render.hm_to_rgb(heat, input_image, scaling=3, sigma=2, cmap='PuOr')
        heatmaps.append(maps)
    R = np.array(heatmaps)
    with tf.name_scope('input_reshape'):
        img = tf.summary.image('input', tf.cast(R, tf.float32), n)
    return img.eval()


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def format_image(image, num_images):
    """
    Formats images
    """
    idxs = np.random.choice(image.shape[0], num_images)
    M = image.shape[1]
    N = image.shape[2]
    imagex = np.squeeze(image[idxs, :, :, :])
    print(imagex.shape)

    return imagex

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plots ROC curve

    Args:
    -----
    FPR, TPR and AUC
    """
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='(AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.axis('equal')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('roc1.png', bbox_inches='tight')


def create_mosaic(image, nrows, ncols):
    """
    Tiles all the layers in nrows x ncols
    Args:
    ------
    image = 3d numpy array of M * N * number of filters dimensions
    nrows = integer representing number of images in a row
    ncol = integer representing number of images in a column

    returns formatted image
    """

    M = image.shape[1]
    N = image.shape[2]

    npad = ((0, 0), (1, 1), (1, 1))
    image = np.pad(image, pad_width=npad, mode='constant', \
                   constant_values=0)
    M += 2
    N += 2
    image = image.reshape(nrows, ncols, M, N)
    image = np.transpose(image, (0, 2, 1, 3))
    image = image.reshape(M * nrows, N * ncols)
    return image


def plot_predictions(images, filename):
    """
    Plots the predictions mosaic
    """
    imagex = format_image(images, 4)
    mosaic = create_mosaic(imagex, 2, 2)
    plt.figure(figsize=(12, 12))
    plt.imshow(mosaic, cmap='gray')
    plt.axis('off')
    plt.savefig(filename + '.png', bbox_inches='tight')


def plot_relevances(rel, img, writer):
    img_summary = visualize(rel, img)
    writer.add_summary(img_summary)
    writer.flush()


def train(tag):
    # Import data
    tag = tag
    sub = 'subset' + str(tag)

    x_train_whole = []
    y_train_whole = []
    if tag==0 or tag==1 or tag==2:
        tot = 8
    elif tag==6:
        tot = 14
    elif tag==8:
        tot = 15
    else:
        tot = 16
    x_test_pos = []
    x_test_neg = []
    for num in range(tot):
        h5f = h5py.File('./src/data/3D_data/' + sub + '_' + str(num) + '.h5', 'r')
        y_tmp = np.asarray(h5f['Y'])
        x_tmp = np.asarray(h5f['X'])
        if max(y_tmp) != 0:
            x_tmp_pos = x_tmp[np.where(y_tmp == 1)[0],:,:,:,:]
            if x_test_pos == []:
                x_test_pos = x_tmp_pos
            else:
                x_test_pos = np.concatenate([x_test_pos, x_tmp_pos])
            negIndex = np.random.choice(np.where(y_tmp == 0)[0], len(x_tmp_pos) * 3, replace=False)
            x_tmp_neg = x_tmp[negIndex, :, :, :, :]
            if x_test_neg == []:
                x_test_neg = x_tmp_neg
            else:
                x_test_neg = np.concatenate([x_test_neg, x_tmp_neg])

            del x_tmp_pos
            del x_tmp_neg
            del negIndex
        del x_tmp
        del y_tmp
    y_test_pos = np.ones(len(x_test_pos))
    y_test_neg = np.zeros(len(x_test_neg))

    x_test_tmp = np.concatenate([x_test_pos, x_test_neg])
    y_test_tmp = np.concatenate([y_test_pos, y_test_neg])


    idx = np.arange(0, len(y_test_tmp))
    np.random.shuffle(idx)
    x_test = np.asarray([x_test_tmp[i] for i in idx])
    y_test = np.asarray([y_test_tmp[i] for i in idx])
    del x_test_tmp
    del y_test_tmp
    del y_test_neg
    del x_test_neg
    del x_test_pos
    del y_test_pos
    print (len(x_test))
    print (len(y_test))
    sub = 'subset'
    for i in range(10):
    #for i in range(2):
        subset = sub+str(i)
        if i != tag:
            if i == 0 or i == 1 or i == 2:
                tot = 8
            elif i == 6:
                tot = 14
            elif i == 8:
                tot = 15
            else:
                tot = 16
            x_train_pos = []
            x_train_neg = []
            for num in range(tot):
            #for num in range(1):
                h5f2 = h5py.File('./src/data/3D_data/' + subset + '_' + str(num) + '.h5', 'r')
                x_tmp = np.asarray(h5f2['X'])
                y_tmp = np.asarray(h5f2['Y'])
                if max(y_tmp)!=0:
                    x_tmp_pos = x_tmp[np.where(y_tmp == 1)[0], :, :, :, :]
                    inp90 = np.zeros_like(x_tmp_pos)
                    inp180 = np.zeros_like(x_tmp_pos)
                    inp270 = np.zeros_like(x_tmp_pos)
                    inp45 = np.zeros_like(x_tmp_pos)
                    inp135 = np.zeros_like(x_tmp_pos)
                    inp225 = np.zeros_like(x_tmp_pos)
                    inp315 = np.zeros_like(x_tmp_pos)

                    for aug in range(len(x_tmp_pos)):
                        inp90[aug,:,:,:,:] = rotate(x_tmp_pos[aug,:,:,:,:], 90, reshape=False)
                        inp180[aug,:,:,:,:] = rotate(x_tmp_pos[aug,:,:,:,:], 180, reshape=False)
                        inp270[aug, :, :, :, :] = rotate(x_tmp_pos[aug, :, :, :, :], 270, reshape=False)
                        inp45[aug, :, :, :, :] = rotate(x_tmp_pos[aug, :, :, :, :], 45, reshape=False)
                        inp135[aug, :, :, :, :] = rotate(x_tmp_pos[aug, :, :, :, :], 135, reshape=False)
                        inp225[aug, :, :, :, :] = rotate(x_tmp_pos[aug, :, :, :, :], 225, reshape=False)
                        inp315[aug, :, :, :, :] = rotate(x_tmp_pos[aug, :, :, :, :], 315, reshape=False)

                    tmp = np.concatenate([np.concatenate([np.concatenate([np.concatenate([np.concatenate([np.concatenate([np.concatenate([x_tmp_pos, inp90]), inp180]), inp270]), inp45]), inp135]), inp225]), inp315])
                    idx2 = np.arange(0, len(tmp))
                    np.random.shuffle(idx2)
                    tmp2 = np.asarray([tmp[a] for a in idx2])
                    del inp90
                    del inp180
                    del inp270
                    del inp45
                    del inp135
                    del inp225
                    del inp315
                    if x_train_pos == []:
                        x_train_pos = tmp2[0:int(len(tmp)/4),:,:,:,:]
                    else:
                        x_train_pos = np.concatenate([x_train_pos, tmp2[0:int(len(tmp)/5),:,:,:,:]])

                    del tmp
                    negIndex = np.random.choice(np.where(y_tmp == 0)[0], len(x_tmp_pos) * 5, replace=False)
                    x_tmp_neg = x_tmp[negIndex, :, :, :, :]
                    if x_train_neg == []:
                        x_train_neg = x_tmp_neg
                    else:
                        x_train_neg = np.concatenate([x_train_neg, x_tmp_neg])

                    del tmp2
                    del x_tmp_neg
                    del x_tmp_pos
                    del negIndex
                del x_tmp
                del y_tmp
            y_train_pos = np.ones(len(x_train_pos))
            y_train_neg = np.zeros(len(x_train_neg))
            x_train_tmp = np.concatenate([x_train_pos, x_train_neg])
            y_train_tmp = np.concatenate([y_train_pos, y_train_neg])
            del x_train_pos
            del x_train_neg
            del y_train_neg
            del y_train_pos
            idx = np.arange(0, len(y_train_tmp))
            np.random.shuffle(idx)
            x_train = np.asarray([x_train_tmp[a] for a in idx])
            y_train = np.asarray([y_train_tmp[a] for a in idx])
            del x_train_tmp
            del y_train_tmp
            if x_train_whole==[]:
                x_train_whole = x_train
                y_train_whole = y_train
            else:
                x_train_whole = np.concatenate([x_train_whole, x_train])
                y_train_whole = np.concatenate([y_train_whole, y_train])
            print (len(x_train_whole))

            del x_train
            del y_train
    x_train = x_train_whole
    y_train = y_train_whole
    del x_train_whole
    del y_train_whole
    print (len(x_train))
    print (len(y_train))
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # with tf.Session() as sess:
        # Input placeholders
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 32, 32, 32, 1], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
            phase = tf.placeholder(tf.bool, name='phase')
        with tf.variable_scope('model'):
            net = nn(phase)
            # x_prep = prep_data_augment(x)
            # x_input = data_augment(x_prep)
            inp = tf.reshape(x, [FLAGS.batch_size, 32, 32, 32, 1])
            op = net.forward(inp)
            y = tf.reshape(op, [FLAGS.batch_size, 2])
            soft = tf.nn.softmax(y)
            trainer = net.fit(output=y, ground_truth=y_, loss='focal loss', optimizer='adam',
                              opt_params=[FLAGS.learning_rate])
        with tf.variable_scope('relevance'):
            if FLAGS.relevance:

                LRP = net.lrp(y, FLAGS.relevance_method, 1)
                # LRP layerwise
                relevance_layerwise = []
                # R = input_rel2
                # for layer in net.modules[::-1]:
                #     R = net.lrp_layerwise(layer, R, FLAGS.relevance_method, 1e-8)
                #     relevance_layerwise.append(R)

            else:
                LRP = []
                relevance_layerwise = []

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
            # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.where(tf.greater(y,0),tf.ones_like(y, dtype=tf.float32), tf.zeros_like(y, dtype=tf.float32)), 2), tf.argmax(y_, 2)), tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./conv_log/'+str(tag)+'_train', sess.graph)
        test_writer = tf.summary.FileWriter('./conv_log/'+str(tag)+'_test')

        tf.global_variables_initializer().run()

        utils = Utils(sess, './3D_model/subset'+str(tag))
        if FLAGS.reload_model:
            utils.reload_model()
        train_acc = []
        test_acc = []
        for i in range(FLAGS.max_steps):

            if i % FLAGS.test_every == 0:  # test-set accuracy
                x_test_batch, y_test_batch = next_batch(FLAGS.batch_size, x_test, y_test)
                tmp_y_batch = np.zeros([FLAGS.batch_size,2])
                tmp_y_batch[:, 0] = np.ones([FLAGS.batch_size]) - y_test_batch
                tmp_y_batch[:, 1] = np.zeros([FLAGS.batch_size]) + y_test_batch
                y_test_batch = tmp_y_batch
                test_inp = {x: x_test_batch, y_: y_test_batch, phase: False}

                # pdb.set_trace()
                summary, acc, relevance_test, op2, soft_val, rel_layer = sess.run([merged, accuracy, LRP, y, soft, relevance_layerwise],
                                                                        feed_dict=test_inp)
                test_writer.add_summary(summary, i)
                test_acc.append(acc)
                print('-----------')
                for m in range(FLAGS.batch_size):
                    print(np.argmax(y_test_batch[m, :]),y_test_batch[m, :], end=" ")
                    print(np.argmax(op2[m, :]),op2[m,:], end=" ")
                    print(soft_val[m,:])
                    print("|")
                print('Accuracy at step %s: %f' % (i, acc))
                print(tag)
                # print([np.sum(rel) for rel in rel_layer])
                # print(np.sum(relevance_test))

                # save model if required
                if FLAGS.save_model:
                    utils.save_model()

            else:
                x_train_batch, y_train_batch = next_batch(FLAGS.batch_size, x_train, y_train)
                tmp_y_batch = np.zeros([FLAGS.batch_size, 2])
                tmp_y_batch[:, 0] = np.ones([FLAGS.batch_size]) - y_train_batch
                tmp_y_batch[:, 1] = np.zeros([FLAGS.batch_size]) + y_train_batch
                y_train_batch = tmp_y_batch
                inp = {x: x_train_batch, y_: y_train_batch, phase: True}
                summary, acc2, _, relevance_train, op2, soft_val, rel_layer = sess.run(
                    [merged, accuracy, trainer.train, LRP, y, soft, relevance_layerwise], feed_dict=inp)
                train_writer.add_summary(summary, i)
                #print(soft_val[0,:])
                train_acc.append(acc2)
        print(np.mean(train_acc), np.mean(test_acc))

        # relevances plotted with visually pleasing color schemes
        if FLAGS.relevance:
            # plot test images with relevances overlaid
            images = test_inp[test_inp.keys()[0]].reshape([FLAGS.batch_size, 32, 32, 32, 1])
            # images = (images + 1)/2.0
            plot_relevances(relevance_test.reshape([FLAGS.batch_size, 32, 32, 32, 1]),
                            images, test_writer)

        train_writer.close()
        test_writer.close()


def main(_):
    tag = int(sys.argv[1])
    #tag = 0
    train(tag)

if __name__ == '__main__':
    tf.app.run()
