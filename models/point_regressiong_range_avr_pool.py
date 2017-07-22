import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    query_points = tf.placeholder(tf.float32, shape=(batch_size, 2))
    range_pl = tf.placeholder(tf.float32, shape=(batch_size))
    return pointclouds_pl, query_points, range_pl

def get_model(point_cloud, query_points, is_training, bn_decay=None):
    """ range regression PointNet, input is BxNx3(point_cloud) and Bx2(query_points), output Bx1 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    #
    query_points = tf.tile(tf.expand_dims(query_points, [1]), [1, num_point, 1]) # Now, query_points is with shape BxNx2
    query_points = tf.concat([query_points, tf.zeros([batch_size, num_point, 1])], 2) # Now, query_points is with shape BxNx3
    # point_cloud_ab = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 2]) # alpha and beta, shape:BxNx2
    # point_cloud_range = tf.slice(point_cloud, [0, 0, 2], [-1, -1, 1]) # range, shape:BxNx1
    # shift_ab = point_cloud_ab - query_points
    shifted_pl = point_cloud - query_points
    #

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: average pooling
    net = tf_util.avg_pool2d(net, [num_point,1],
                             padding='VALID', scope='avgpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=False, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: Bx1,
        label: Bx1, """
    regression_loss = tf.reduce_mean(tf.squared_difference(pred, label))
    tf.summary.scalar('regression_loss', regression_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('mat loss', mat_diff_loss)

    return regression_loss + mat_diff_loss * reg_weight
