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
parser.add_argument('--model', default='point_regressiong_pose', help='Model name[default: point_regressiong_pose]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--h5_filename', default='kitti_velo_patches.h5', help='dataset h5 file name [default: kitti_velo_patches.h5]')
parser.add_argument('--num_point', type=int, default=120000, help='Point Number [120000] [default: 120000]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=40000, help='Decay step for lr decay [default: 40000]')
parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate for lr decay [default: 0.1]')
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

MAX_NUM_POINT = 120000

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(step):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        step,
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(step):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      step,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            point_cloud_moving, point_cloud_fixed, pose_gt = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points = MODEL.get_model(point_cloud_moving, point_cloud_fixed, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, pose_gt, end_points)
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

        ops = {'point_cloud_moving': point_cloud_moving,
               'point_cloud_fixed': point_cloud_fixed,
               'pose_gt': pose_gt,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        batches = batch_generator(FLAGS.h5_filename, BATCH_SIZE, NUM_POINT)
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
    def __init__(self, filename, batch_size, num_point):
        self.filename = filename
        self.h5 = h5py.File(self.filename)
        self.batches = self.h5.keys()
        n_batches = len(self.batches)
        self.train_batches = self.batches[0:int(n_batches*0.8)]
        self.eval_batches = self.batches[int(n_batches*0.8):-1]
        self.batch_size = batch_size
        self.num_point = num_point

    def get_train_batch_generator(self):
        while True:
            batch_idxs = np.random.permutation(len(self.train_batches))
            print("train random permutation, len(batch_idxs) = %d"%len(batch_idxs))
            for i in range(0, len(batch_idxs), self.batch_size):
                point_cloud_moving = np.zeros((self.batch_size, self.num_point, 3))
                point_cloud_fixed = np.zeros((self.batch_size, self.num_point, 3))
                pose_gt = np.zeros((self.batch_size, 6))
                for j in range(self.batch_size):
                    batch = self.h5[self.train_batches[batch_idxs[i+j]]]
                    point_cloud_moving[j,:,:] = batch['point_cloud_moving']
                    point_cloud_fixed[j,:,:] = batch['point_cloud_fixed']
                    pose_gt[j, :] = batch['pose']
                yield point_cloud_moving, point_cloud_fixed, pose_gt

    def get_eval_batch_generator(self):
        while True:
            batch_idxs = np.random.permutation(len(self.eval_batches))
            print("eval random permutation, len(batch_idxs) = %d"%len(batch_idxs))
            for i in range(0, len(batch_idxs), self.batch_size):
                point_cloud_moving = np.zeros((self.batch_size, self.num_point, 3))
                point_cloud_fixed = np.zeros((self.batch_size, self.num_point, 3))
                pose_gt = np.zeros((self.batch_size, 6))
                for j in range(self.batch_size):
                    batch = self.h5[self.eval_batches[batch_idxs[i+j]]]
                    point_cloud_moving[j,:,:] = batch['point_cloud_moving']
                    point_cloud_fixed[j,:,:] = batch['point_cloud_fixed']
                    pose_gt[j, :] = batch['pose']
                yield point_cloud_moving, point_cloud_fixed, pose_gt

def train_one_epoch(sess, ops, train_writer, train_batch_generator):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    loss_sum = 0
    n_batches = 100

    for i in range(n_batches):
		#log_string('----' + str(i) + '-----')
        point_cloud_moving, point_cloud_fixed, pose_gt = train_batch_generator.next()

        feed_dict = {ops['point_cloud_moving']: point_cloud_moving,
            ops['point_cloud_fixed']: point_cloud_fixed,
            ops['pose_gt']: pose_gt,
            ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)

        print('train pred_val: ', pred_val[0,:].squeeze())
        print('train pose_gt: ', pose_gt[0,:].squeeze())

        train_writer.add_summary(summary, step)
        loss_sum += loss_val

    log_string('train mean loss: %f' % (loss_sum / float(n_batches)))

def eval_one_epoch(sess, ops, test_writer, eval_batch_generator):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    loss_sum = 0
    n_batches = 25

    for i in range(n_batches):
		#log_string('----' + str(i) + '-----')
        point_cloud_moving, point_cloud_fixed, pose_gt = eval_batch_generator.next()

        feed_dict = {ops['point_cloud_moving']: point_cloud_moving,
            ops['point_cloud_fixed']: point_cloud_fixed,
            ops['pose_gt']: pose_gt,
            ops['is_training_pl']: is_training,}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)

        print('eval pred_val: ', pred_val[0,:].squeeze())
        print('eval pose_gt: ', pose_gt[0,:].squeeze())

        test_writer.add_summary(summary, step)
        loss_sum += loss_val

    log_string('eval mean loss: %f' % (loss_sum / float(n_batches)))

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
