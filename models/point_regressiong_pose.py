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
    point_cloud_moving = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    point_cloud_fixed = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pose_gt = tf.placeholder(tf.float32, shape=(batch_size, 6))
    return point_cloud_moving, point_cloud_fixed, pose_gt

def feature_extractor(point_cloud, feature_length, is_training, bn_decay=None):
    """ extract fix length feature from a point cloud, input is BxNx3, output B x feature_length """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

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
    net_transformed = tf.matmul(tf.squeeze(net, [2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, feature_length, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    # output feature size = B x feature_length
    feature = tf.reshape(net, [batch_size, -1])

    return feature, end_points

def get_model(point_cloud_moving, point_cloud_fixed, is_training, bn_decay=None):
    """ pose regression PointNet, input is BxNx3(point_cloud_moving) and BxNx3(point_cloud_fixed), output Bx6(pose 6 DOF) """
    batch_size = point_cloud_moving.get_shape()[0].value
    num_point = point_cloud_moving.get_shape()[1].value
    end_points = {}
    feature_length = 1024
    with tf.variable_scope('shared_feature_extractor') as sc:
        feature_moving, ep = feature_extractor(point_cloud_moving, feature_length, is_training, bn_decay)
        end_points['transform'] = ep['transform']
        tf.get_variable_scope().reuse_variables()
        feature_fixed, transform = feature_extractor(point_cloud_fixed, feature_length, is_training, bn_decay)
    feature_concat = tf.concat([feature_moving, feature_fixed], 1)

    net = tf_util.fully_connected(feature_concat, 512, bn=False, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                       scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                       scope='dp2')
    pose = tf_util.fully_connected(net, 6, activation_fn=None, scope='fc3')
    return pose, end_points

def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: Bx6,
        label: Bx6, """
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
