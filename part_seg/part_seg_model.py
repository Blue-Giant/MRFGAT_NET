import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../'))
import tf_util
from transform_nets import input_transform_net


def multiscale_view_filedGAP(point_cloud, nn_idx1, nn_idx2, nn_idx3, nn_idx4, out_dims2neighbors=None,
                                  k_neighbors=None, multi_heads=3, scope_flag=0, is_training=None, bn_decay=False):

    """Construct edge feature for each point with laplacian structure
    Args:
    point_cloud: (batch_size, num_points, 1, num_dims)
    nn_idx: (batch_size, num_points, k)
    k: int

    Returns:
    edge features: (batch_size, num_points, k, num_dims)
    """
    if scope_flag == 0:
        flag_scope = 'transform'
    else:
        flag_scope = 'laplace'

    update_local_list = []
    update_neighbors_edge_list = []
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    nnidx = [nn_idx1, nn_idx2, nn_idx3, nn_idx4]
    point_cloud_neighbors = []
    point_cloud_central_tilde = []
    edge_features = []
    for i in range(multi_heads):
        point_centroid_tilde = tf.tile(point_cloud_central, [1, 1, k_neighbors[i], 1])
        point_cloud_central_tilde.append(point_centroid_tilde)

        point_neighbors = tf.gather(point_cloud_flat, nnidx[i] + idx_)
        point_cloud_neighbors.append(point_neighbors)

        edges = point_centroid_tilde - point_neighbors
        edge_features.append(edges)

    # 多通道注意力机制
    for multi_i in range(multi_heads):
        output_dim = out_dims2neighbors[multi_i]
        with tf.variable_scope('multi_attention', reuse=tf.AUTO_REUSE):
            new_edge_feature = tf_util.conv2d_nobias(edge_features[multi_i], output_dim, [1, 1], padding='VALID',
                                                     stride=[1, 1], activation_fn=tf.nn.relu, bn=True,
                                                     is_training=is_training, scope='new_edge' + str(multi_i)+flag_scope,
                                                     bn_decay=bn_decay)

            attention2new_edge = tf_util.conv2d(new_edge_feature, 1, [1, 1],
                                                padding='VALID', stride=[1, 1], activation_fn=tf.nn.leaky_relu,
                                                bn=True, is_training=is_training,
                                                scope='attention_edge_features' + str(multi_i)+flag_scope,
                                                bn_decay=bn_decay)

            attention2new_edge = tf.nn.softmax(attention2new_edge)
            attention_weight2new_edge = tf.transpose(attention2new_edge, [0, 1, 3, 2])
            weight_new_edge_feature = tf.matmul(attention_weight2new_edge, new_edge_feature)

            new_neighbors = tf_util.conv2d_nobias(point_cloud_neighbors[multi_i], output_dim, [1, 1], padding='VALID', stride=[1, 1],
                                                  activation_fn=tf.nn.relu, bn=True, is_training=is_training,
                                                  scope='new_neighbors_features' + str(multi_i) + flag_scope,
                                                  bn_decay=bn_decay)

            attention2neighbors = tf_util.conv2d(edge_features[multi_i], 1, [1, 1],
                                                 padding='VALID', stride=[1, 1], activation_fn=tf.nn.leaky_relu,
                                                 bn=True, is_training=is_training,
                                                 scope='attention_neighbors_features' + str(multi_i) + flag_scope,
                                                 bn_decay=bn_decay)

            attention2neighbors = tf.nn.softmax(attention2neighbors)
            attention2neighbors = tf.transpose(attention2neighbors, perm=[0, 1, 3, 2])
            weight_new_neighbors_feature = tf.matmul(attention2neighbors, new_neighbors)

            new_points_feature = tf.concat([weight_new_neighbors_feature, weight_new_edge_feature], axis=-1)

            update_neighbors_edge_list.append(new_points_feature)

            reduce_max_new_edge_feature = tf.reduce_max(new_edge_feature, axis=-2, keep_dims=True)
            update_local_list.append(reduce_max_new_edge_feature)

    return update_neighbors_edge_list, update_local_list


def get_model(point_cloud, input_label, is_training, cat_num, part_num, \
    batch_size, num_point, weight_decay, bn_decay=None):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    k_neighbors = (10, 20, 30, 40)
    out_dims2neighbors = (8, 16, 16, 24)
    n_heads = 4
    origin_adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx1 = tf_util.knn(origin_adj_matrix, k=k_neighbors[0])
    nn_idx2 = tf_util.knn(origin_adj_matrix, k=k_neighbors[1])
    nn_idx3 = tf_util.knn(origin_adj_matrix, k=k_neighbors[2])
    nn_idx4 = tf_util.knn(origin_adj_matrix, k=k_neighbors[3])

    # 对齐后的点云进入laplace—smooth过程
    flag1 = 1
    neighbors_edge_feature_list, local_feature_list = multiscale_view_filedGAP(
        point_cloud, nn_idx1=nn_idx1, nn_idx2=nn_idx2, nn_idx3=nn_idx3, nn_idx4=nn_idx4,
        out_dims2neighbors=out_dims2neighbors, k_neighbors=k_neighbors, multi_heads=n_heads,
        scope_flag=flag1, is_training=is_training, bn_decay=bn_decay)

    # 类似多通道注意力机制最后的输出，将得到的多通道数据串联起来 32*4。由于每个通道都最后做了 relu 激活，所以都为正值
    neighbors_edge_feature = tf.concat(neighbors_edge_feature_list, axis=-1)

    conv_net = tf_util.conv2d(neighbors_edge_feature, 128, [1, 1], padding='VALID', stride=[1, 1],
                              bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net1 = conv_net  # 128

    conv_net = tf_util.conv2d(conv_net, 64, [1, 1], padding='VALID', stride=[1, 1],
                              bn=True, is_training=is_training, scope='conv2',
                              bn_decay=bn_decay)
    net2 = conv_net  # 64

    conv_net = tf_util.conv2d(conv_net, 64, [1, 1], padding='VALID', stride=[1, 1],
                              bn=True, is_training=is_training, scope='conv3',
                              bn_decay=bn_decay)
    net3 = conv_net  # 64

    local_feature = tf.concat(local_feature_list, axis=-1)

    out7 = tf_util.conv2d(tf.concat([net1, net2, net3, local_feature], axis=-1), 1024, [1, 1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv7', bn_decay=bn_decay, is_dist=True)
    out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')

    one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
    one_hot_label_expand = tf_util.conv2d(one_hot_label_expand, 64, [1, 1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='one_hot_label_expand', bn_decay=bn_decay, is_dist=True)
    out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])
    expand = tf.tile(out_max, [1, num_point, 1, 1])

    concat = tf.concat(axis=3, values=[expand,
                                     net1,
                                     net2,
                                     net3,
                                    local_feature])

    net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay, is_dist=True)
    net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay, is_dist=True)
    net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp2')
    net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay, is_dist=True)
    net2 = tf_util.conv2d(net2, part_num, [1,1], padding='VALID', stride=[1,1], activation_fn=None,
            bn=False, scope='seg/conv4', weight_decay=weight_decay, is_dist=True)

    net2 = tf.reshape(net2, [batch_size, num_point, part_num])

    return net2


def get_loss(seg_pred, seg):
  per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg), axis=1)
  seg_loss = tf.reduce_mean(per_instance_seg_loss)
  per_instance_seg_pred_res = tf.argmax(seg_pred, 2)
  
  return seg_loss, per_instance_seg_loss, per_instance_seg_pred_res

