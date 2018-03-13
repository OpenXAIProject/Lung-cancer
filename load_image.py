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
from modules.sequential import Sequential
from modules.convolution3D import Convolution3D
from modules.maxpool3D import MaxPool3D
from modules.utils import Utils, Summaries, plot_relevances
import modules.render as render
import tensorflow as tf
import numpy as np
import h5py
import imageio
flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 300, 'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 20, 'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 100, 'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.001, 'Initial learning rate')
flags.DEFINE_string("summaries_dir", 'convolutional_logs', 'Summaries directory')
flags.DEFINE_boolean("relevance", True, 'Compute relevances')
flags.DEFINE_string("relevance_method", 'alphabeta', 'relevance methods: simple/eps/w^2/alphabeta')
flags.DEFINE_boolean("save_model", False, 'Save the trained model')
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
    n = len(relevances)

    for j in range(n):
        heatmaps = []
        for i in range(32):
            heatmap = relevances[j, i, :, :,  :]
            input_images = images_tensor[j, i, :, :, :]

            maps = render.hm_to_rgb(np.squeeze(heatmap), scaling=8, sigma=2, cmap='bwr')
            img = input_images
            if heatmaps == []:
                heatmaps = np.expand_dims(maps,0)
                imgs = np.expand_dims(img,0)
            else:
                heatmaps = np.concatenate([heatmaps,np.expand_dims(maps,0)],0)
                imgs = np.concatenate([imgs,np.expand_dims(img,0)],0)
        imageio.mimsave('./Demo_img/heatmap'+str(j)+'.gif', heatmaps)

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



def plot_relevances(rel, img, writer):
    img_summary = visualize(rel, img)
    writer.add_summary(img_summary)
    writer.flush()

def load_data(tag,num):
    tag = str(tag)
    num = int(num)
    sub = 'subset' + tag

    tag = int(tag)
    h5f = h5py.File('./src/data/3D_data/' + sub + '_' + str(num) + '.h5', 'r')
    y_test = np.asarray(h5f['Y'])
    x_test = np.asarray(h5f['X'])
    cnt = 0;
    cnt2 = 0;
    x_test_batch_gt = np.zeros([100, 32, 32, 32, 1])
    y_test_batch_gt = np.zeros([100, 2])

    x_test_batch_ngt = np.zeros([100, 32, 32, 32, 1])
    y_test_batch_ngt = np.zeros([100, 2])
    for i in range(FLAGS.max_steps):

        x_test_batch, y_test_batch = np.asarray(x_test)[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :,
                                     :], np.asarray(y_test)[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
        tmp_y_batch = np.zeros([FLAGS.batch_size, 2])
        tmp_y_batch[:, 0] = np.ones([FLAGS.batch_size]) - y_test_batch
        tmp_y_batch[:, 1] = np.zeros([FLAGS.batch_size]) + y_test_batch
        y_test_batch = tmp_y_batch
        for m in range(FLAGS.batch_size):

            if np.argmax(y_test_batch[m, :]) == 1:
                x_test_batch_gt[cnt, :, :, :, :] = x_test_batch[m, :, :, :, :]
                y_test_batch_gt[cnt, :] = y_test_batch[m, :]
                cnt = cnt + 1
            else:
                if cnt2 < 20:
                    x_test_batch_ngt[cnt2, :, :, :, :] = x_test_batch[m, :, :, :, :]
                    y_test_batch_ngt[cnt2, :] = y_test_batch[m, :]
                    cnt2 = cnt2 + 1
    y_demo_test = np.concatenate([y_test_batch_gt[0:cnt, :], y_test_batch_ngt[0:cnt2, :]])
    x_demo_test = np.concatenate([x_test_batch_gt[0:cnt, :, :, :, :], x_test_batch_ngt[0:cnt2, :, :, :, :]])
    for j in range(cnt):
        imgs = []
        for i in range(32):
            x_tmp = x_demo_test[j, i, :, :, :]

            img = render.enlarge_image(x_tmp, 8)
            if imgs == []:
                imgs = np.expand_dims(img,0)
            else:
                imgs = np.concatenate([imgs,np.expand_dims(img,0)],0)
        imageio.mimsave('./Demo_img/img'+str(j)+'.gif', imgs)
    np.save('./Demo_img/test_demo_x.npy', x_demo_test[0:20,:,:,:,:])
    np.save('./Demo_img/test_demo_y.npy', y_demo_test[0:20, :])
def train(tag):
    tag = str(tag)

    tag = int(tag)
    x_test_batch = np.load('./Demo_img/test_demo_x.npy')
    y_test_batch = np.load('./Demo_img/test_demo_y.npy')
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
        with tf.variable_scope('relevance'):
            if FLAGS.relevance:

                LRP = net.lrp(soft, FLAGS.relevance_method, 2)

                # LRP layerwise
                relevance_layerwise = []
                #R = tf.expand_dims(soft[0, :], 0)
                R = soft
                for layer in net.modules[::-1]:
                    R = net.lrp_layerwise(layer, R, FLAGS.relevance_method, 2)
                    relevance_layerwise.append(R)

            else:
                LRP = []
                relevance_layerwise = []

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
            # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.where(tf.greater(y,0),tf.ones_like(y, dtype=tf.float32), tf.zeros_like(y, dtype=tf.float32)), 2), tf.argmax(y_, 2)), tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter('./conv_log/LRP')

        tf.global_variables_initializer().run()

        utils = Utils(sess, './3D_model/subset'+str(tag))
        if FLAGS.reload_model:
            utils.reload_model()

        test_inp = {x: x_test_batch, y_: y_test_batch, phase: False}

        # pdb.set_trace()
        relevance_test, op, soft_val, rel_layer = sess.run([LRP, y, soft, relevance_layerwise],
                                                            feed_dict=test_inp)
        for m in range(FLAGS.batch_size):
            print(soft_val[m, :])
        np.save('./Demo_img/soft.npy',soft_val)
        if FLAGS.relevance:
            # plot test images with relevances overlaid
            images = test_inp[test_inp.keys()[0]].reshape([FLAGS.batch_size, 32, 32, 32, 1])
            # images = (images + 1)/2.0
            plot_relevances(relevance_test.reshape([FLAGS.batch_size, 32, 32, 32, 1]),
                            images, test_writer)
        test_writer.close()


def main(_):
    bol = int(sys.argv[1])
    # num = sys.argv[2]
    tag = 0
    num = 0
    if bol == 0:
        load_data(tag, num)
    else:
        train(tag)
if __name__ == '__main__':
    tf.app.run()
