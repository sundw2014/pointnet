import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='point_regressiong_range', help='Model name: point_regressiong_range [default: point_regressiong_range]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--h5_filename', default='kitti_velo_patches.h5', help='dataset h5 file name [default: kitti_velo_patches.h5]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp ' + __file__ + ' %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, query_points, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, query_points, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'query_points': query_points,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        batches = batch_generator(FLAGS.h5_filename)
        train_batch_generator = batches.get_train_batch_generator()
        eval_batch_generator = batches.get_eval_batch_generator()

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer, train_batch_generator)
            eval_one_epoch(sess, ops, test_writer, eval_batch_generator)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

class batch_generator(object):
    """batch_generator"""
    def __init__(self, filename):
        self.filename = filename
        self.train_count = 0
        self.eval_count = 0
        self.h5 = h5py.File(self.filename)
        self.frames = self.h5.keys()
        n_frame = len(self.frames)
        self.train_frames = self.frames[0:int(n_frame*0.8)]
        self.eval_frames = self.frames[int(n_frame*0.8):-1]

    def get_train_batch_generator(self):
        while True:
            frame_idxs = np.random.permutation(len(self.train_frames))
            print("train random permutation, len(frame_idxs) = %d"%len(frame_idxs))
            for frame_idx in frame_idxs:
                frame = self.h5[self.train_frames[frame_idx]]
                patches = frame.keys()
                patch_idxs = np.random.permutation(len(patches))
                for patch_idx in patch_idxs:
                    patch = frame[patches[patch_idx]]
                    pl = patch['pl']
                    query = patch['query']
                    query_points = query[:,0:2]
                    range_gt = query[:,2]
                    yield pl, query_points, range_gt

    def get_eval_batch_generator(self):
        while True:
            frame_idxs = np.random.permutation(len(self.eval_frames))
            print("eval random permutation, len(frame_idxs) = %d"%len(frame_idxs))
            for frame_idx in frame_idxs:
                frame = self.h5[self.eval_frames[frame_idx]]
                patches = frame.keys()
                patch_idxs = np.random.permutation(len(patches))
                for patch_idx in patch_idxs:
                    patch = frame[patches[patch_idx]]
                    pl = patch['pl']
                    query = patch['query']
                    query_points = query[:,0:2]
                    range_gt = query[:,2]
                    yield pl, query_points, range_gt

def train_one_epoch(sess, ops, train_writer, train_batch_generator):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    loss_sum = 0
    n_batches = 8000

    for i in range(n_batches):
		#log_string('----' + str(i) + '-----')
        pl, query_points, range_gt = train_batch_generator.next()

        feed_dict = {ops['pointclouds_pl']: np.tile(np.expand_dims(pl, axis=0), [BATCH_SIZE, 1, 1]),
            ops['query_points']: query_points,
            ops['labels_pl']: range_gt,
            ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)

        print('train pred_val: ', pred_val[0:3])
        print('train range_gt: ', range_gt[0:3])

        train_writer.add_summary(summary, step)
        loss_sum += loss_val

    log_string('train mean loss: %f' % (loss_sum / float(n_batches)))

def eval_one_epoch(sess, ops, test_writer, eval_batch_generator):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    loss_sum = 0
    n_batches = 2000

    for i in range(n_batches):
		#log_string('----' + str(i) + '-----')
        pl, query_points, range_gt = eval_batch_generator.next()

        feed_dict = {ops['pointclouds_pl']: np.tile(np.expand_dims(pl, axis=0), [BATCH_SIZE, 1, 1]),
            ops['query_points']: query_points,
            ops['labels_pl']: range_gt,
            ops['is_training_pl']: is_training,}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)

        print('eval pred_val: ', pred_val[0:3])
        print('eval range_gt: ', range_gt[0:3])

        test_writer.add_summary(summary, step)
        loss_sum += loss_val

    log_string('eval mean loss: %f' % (loss_sum / float(n_batches)))

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
