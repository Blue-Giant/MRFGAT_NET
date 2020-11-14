import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))

import tf_util
# from transform_nets import input_transform_net15


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


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
            attention2new_neighbors = tf.transpose(attention2neighbors, perm=[0, 1, 3, 2])
            weight_new_neighbors_feature = tf.matmul(attention2new_neighbors, new_neighbors)

            new_points_feature = tf.concat([weight_new_neighbors_feature, weight_new_edge_feature], axis=-1)

            update_neighbors_edge_list.append(new_points_feature)

            reduce_max_new_edge_feature = tf.reduce_max(new_edge_feature, axis=-2, keep_dims=True)
            update_local_list.append(reduce_max_new_edge_feature)

    return update_neighbors_edge_list, update_local_list


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
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
                              bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    net2 = conv_net  # 64

    conv_net = tf_util.conv2d(conv_net, 64, [1, 1], padding='VALID', stride=[1, 1],
                              bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    net3 = conv_net  # 64

    conv_net = tf_util.conv2d(conv_net, 64, [1, 1], padding='VALID', stride=[1, 1],
                              bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    net4 = conv_net  # 64

    local_feature = tf.concat(local_feature_list, axis=-1)

    net_out_concat = tf.concat([net1, net2, net3, net4, local_feature], axis=-1)
    net_out = tf_util.conv2d(net_out_concat, 1024, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, scope='agg', bn_decay=bn_decay)
 
    net_out = tf.reduce_max(net_out, axis=1, keep_dims=True)

    # MLP on global point cloud vector
    net_out = tf.reshape(net_out, [batch_size, -1])
    net_out = tf_util.fully_connected(net_out, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)

    net_out = tf_util.dropout(net_out, keep_prob=0.5, is_training=is_training, scope='dp1')

    net_out = tf_util.fully_connected(net_out, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)

    net_out = tf_util.dropout(net_out, keep_prob=0.5, is_training=is_training, scope='dp2')

    net_out = tf_util.fully_connected(net_out, 40, activation_fn=None, scope='fc3')

    return net_out, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
       label: B, """
    labels = tf.one_hot(indices=label, depth=40)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    classify_loss = tf.reduce_mean(loss)
    return classify_loss


if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)

  # # np.save('./debug/input_feed.npy', input_feed)
  # input_feed = np.load('./debug/input_feed.npy')
  # print input_feed

  with tf.Graph().as_default():
    input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
    pos, ftr = get_model(input_pl, tf.constant(True))
    # loss = get_loss(logits, label_pl, None)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {input_pl: input_feed, label_pl: label_feed}
      res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
      print (res1.shape)
      print (res1)

      # print (res2.shape)
      # print (res2)