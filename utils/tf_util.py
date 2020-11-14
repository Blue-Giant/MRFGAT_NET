import numpy as np
import tensorflow as tf

def _variable_on_cpu(name, shape, initializer, use_fp16=False, trainable=True):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_dist=False):
    """ 2D convolution with non-linear operation.

      Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

      Returns:
        Variable tensor
      """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn', is_dist=is_dist)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv2d_nobias(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_dist=False):
    """ 2D convolution with non-linear operation.

      Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

      Returns:
        Variable tensor
      """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn', is_dist=is_dist)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs



def batch_norm_for_fc(inputs, is_training, bn_decay, scope, is_dist=False):
    """ Batch normalization on FC data.

    Args:
        inputs:      Tensor, 2D BxC input
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
        is_dist:     true indicating distributed training scheme
    Return:
        normed:      batch-normalized maps
    """
    if is_dist:
        return batch_norm_dist_template(inputs, is_training, scope, [0, ], bn_decay)
    else:
        return batch_norm_template(inputs, is_training, scope, [0, ], bn_decay)

def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None,
                    is_dist=False):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[num_input_units, num_outputs],
                                              use_xavier=use_xavier,
                                              stddev=stddev,
                                              wd=weight_decay)
        outputs = tf.matmul(inputs, weights)
        biases = _variable_on_cpu('biases', [num_outputs],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn', is_dist=is_dist)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D max pooling.

    Args:
      inputs: 4-D tensor BxHxWxC
      kernel_size: a list of 2 ints
      stride: a list of 2 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs


def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


def batch_norm_dist_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ The batch normalization for distributed training.
    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = _variable_on_cpu('beta', [num_channels], initializer=tf.zeros_initializer())
        gamma = _variable_on_cpu('gamma', [num_channels], initializer=tf.ones_initializer())

        pop_mean = _variable_on_cpu('pop_mean', [num_channels], initializer=tf.zeros_initializer(), trainable=False)
        pop_var = _variable_on_cpu('pop_var', [num_channels], initializer=tf.ones_initializer(), trainable=False)

        def train_bn_op():
            batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
            decay = bn_decay if bn_decay is not None else 0.9
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, 1e-3)

        def test_bn_op():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, 1e-3)

        normed = tf.cond(is_training,
                         train_bn_op,
                         test_bn_op)
        return normed


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, is_dist=False):
    """ Batch normalization on 2D convolutional maps.

    Args:
        inputs:      Tensor, 4D BHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
        is_dist:     true indicating distributed training scheme
    Return:
        normed:      batch-normalized maps
    """
    if is_dist:
        return batch_norm_dist_template(inputs, is_training, scope, [0, 1, 2], bn_decay)
    else:
        return batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
  """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs


def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.

    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)

    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
      Args:
        pairwise distance: (batch_size, num_points, num_points)
        k: int

      Returns:
        nearest neighbors: (batch_size, num_points, k)
      """
    neg_adj = -adj_matrix
    _, nn_idx = tf.nn.top_k(neg_adj, k=k)
    return nn_idx


def get_neighbors(point_cloud, nn_idx, k=20):
    """Construct neighbors feature for each point
      Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k)
        k: int

      Returns:
        neighbors features: (batch_size, num_points, k, num_dims)
      """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    og_num_dims = point_cloud.get_shape().as_list()[-1]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)
    if og_num_dims == 1:
        point_cloud = tf.expand_dims(point_cloud, -1)

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)

    return point_cloud_neighbors


# laplace 光滑+多头注意力机制
def get_feature_laplace_smooth_multi_attention15(point_cloud, nn_idx, out_dims=16, k_neighbors=20, heads=3,
                                                scope_flag=0, is_training=None, bn_decay=False):

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

    update_feature2_edge_list = []
    update_centroid_feature_list = []
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

    # 把点云 point_cloud 变为batch_size*num_points行 num_dim 列的矩阵
    # 第一个 num_points 行代表第一个点云，第二个 num_points 行代表第二个点云......
    # 第batch_size个 num_points 行代表第batch_size个点云
    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])

    # 将point_cloud变成point_cloud_flat后(batch_size*num_points行)，每隔num_points行取出一行，
    # 要取的行号根据为nn_idx里面的索引元素(k个)+batch_size*num_points
    # 所以每个点云(batch_size个)都会得到自己的k个邻居
    # 最后得到 point_cloud_neighbors：(batch_size, num_points, k, num_dims)
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k_neighbors, 1])

    edge_feature = point_cloud_central - point_cloud_neighbors

    # 多通道注意力机制+ laplace光滑过程
    multi_head = heads
    output_dim = out_dims
    attention_out_dim = k_neighbors
    for multi_i in range(multi_head):
        with tf.variable_scope('v_attention', reuse=tf.AUTO_REUSE):
            # 对初始输入做一个非线性变换(不加bias)
            new_centroid_feature = conv2d_nobias(point_cloud_central, output_dim, [1, 1], padding='VALID',
                                                 stride=[1, 1], bn=True, is_training=is_training,
                                                 scope='smooth_new_centroid_feature' + str(multi_i)+flag_scope,
                                                 bn_decay=bn_decay)

            # 对初始输入形成的边做一个非线性变换(不加bias)
            new_edge_feature = conv2d_nobias(edge_feature, output_dim, [1, 1], padding='VALID', stride=[1, 1],
                                             bn=True, is_training=is_training,
                                             scope='smooth_new_edge_feature' + str(multi_i)+flag_scope,
                                             bn_decay=bn_decay)

            # 对新点集的点与点之间，求一个自注意权值矩阵
            self_new_centroid = tf.concat([new_centroid_feature, new_centroid_feature], axis=-1)
            attention_weight_matrix = conv2d(self_new_centroid, attention_out_dim, [1, 1], padding='VALID',
                                             stride=[1, 1], activation_fn=tf.nn.leaky_relu,
                                             bn=True, is_training=is_training,
                                             scope='smooth_centroid_attention' + str(multi_i)+flag_scope,
                                             bn_decay=bn_decay)
            # 权值矩阵行归一化
            attention_weight_matrix = tf.nn.softmax(attention_weight_matrix)

            # 对非线性变换后的边特征乘上权值,点集也乘上权值矩阵
            weight_new_centroid_feature = tf.matmul(attention_weight_matrix, new_centroid_feature)

            # 中心点进行laplace
            factor = tf.constant(0.5)
            smooth_centroid_mixture = tf.add(tf.multiply(1 - factor, new_centroid_feature), tf.multiply(
                factor, weight_new_centroid_feature))

            # 对新点集的点与新边集的边之间，求一个自注意权值矩阵
            new_centroid_edge_concat = tf.concat([new_centroid_feature, new_edge_feature], axis=-1)
            attention_weight_centroid_edge = conv2d(new_centroid_edge_concat, attention_out_dim, [1, 1], padding='VALID',
                                                    stride=[1, 1], activation_fn=tf.nn.leaky_relu,
                                                    bn=True, is_training=is_training,
                                                    scope='centroid_edge_attention' + str(multi_i)+flag_scope,
                                                    bn_decay=bn_decay)

            weight_new_edge_feature = tf.matmul(attention_weight_centroid_edge, new_edge_feature)

            edge_centroid_mixture = tf.concat([new_centroid_feature, weight_new_edge_feature], axis=-1)

            # 光滑后的特征存储到一个列表中
            update_feature2_edge_list.append(edge_centroid_mixture)
            update_centroid_feature_list.append(smooth_centroid_mixture)

    return update_feature2_edge_list, update_centroid_feature_list


# laplace 锐化+多头注意力机制
def get_feature_laplace_sharp_multi_attention15(point_cloud, nn_idx, out_dims=16, k_neighbors=20, heads=3,
                                               scope_flag=0, is_training=None, bn_decay=False):
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

    update_feature2_edge_list = []
    update_centroid_feature_list = []
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

    # 把点云 point_cloud 变为batch_size*num_points行 num_dim 列的矩阵
    # 第一个 num_points 行代表第一个点云，第二个 num_points 行代表第二个点云......
    # 第batch_size个 num_points 行代表第batch_size个点云
    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])

    # 将point_cloud变成point_cloud_flat后(batch_size*num_points行)，每隔num_points行取出一行，
    # 要取的行号根据为nn_idx里面的索引元素(k个)+batch_size*num_points
    # 所以每个点云(batch_size个)都会得到自己的k个邻居
    # 最后得到 point_cloud_neighbors：(batch_size, num_points, k, num_dims)
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k_neighbors, 1])

    edge_feature = point_cloud_central - point_cloud_neighbors

    # 多通道注意力机制+ laplace锐化过程
    multi_head = heads
    output_dim = out_dims
    attention_out_dim = k_neighbors
    for multi_i in range(multi_head):
        with tf.variable_scope('v_attention', reuse=tf.AUTO_REUSE):
            # 对初始输入做一个非线性变换(加不加bias呢？)
            new_centroid_feature = conv2d_nobias(point_cloud_central, output_dim, [1, 1], padding='VALID',
                                                 stride=[1, 1], bn=True, is_training=is_training,
                                                 scope='sharp_new_centroid_feature' + str(multi_i)+flag_scope,
                                                 bn_decay=bn_decay)

            # 对初始输入形成的边做一个非线性变换
            new_edge_feature = conv2d_nobias(edge_feature, output_dim, [1, 1], padding='VALID', stride=[1, 1],
                                             bn=True, is_training=is_training,
                                             scope='sharp_new_edge_feature' + str(multi_i)+flag_scope,
                                             bn_decay=bn_decay)

            # 对变换的新点，求一个权值矩阵
            self_new_centroid = tf.concat([new_centroid_feature, new_centroid_feature], axis=-1)
            attention_weight_matrix = conv2d(self_new_centroid, attention_out_dim, [1, 1], padding='VALID',
                                             stride=[1, 1], activation_fn=tf.nn.leaky_relu,
                                             bn=True, is_training=is_training,
                                             scope='sharp_centroid_edge_attention' + str(multi_i)+flag_scope,
                                             bn_decay=bn_decay)
            # 权值矩阵行归一化
            attention_weight_matrix = tf.nn.softmax(attention_weight_matrix)

            # 对非线性变换后的边特征乘上权值,点集也乘上权值矩阵
            weight_new_centroid_feature = tf.matmul(attention_weight_matrix, new_centroid_feature)

            factor = tf.constant(0.75)
            sharp_centroid_mixture = tf.subtract(tf.multiply(1 + factor, new_centroid_feature), tf.multiply(
                factor, weight_new_centroid_feature))

            # 对新点集的点与新边集的边之间，求一个自注意权值矩阵
            new_centroid_edge_concat = tf.concat([new_centroid_feature, new_edge_feature], axis=-1)
            attention_weight_centroid_edge = conv2d(new_centroid_edge_concat, attention_out_dim, [1, 1], padding='VALID',
                                                    stride=[1, 1], activation_fn=tf.nn.leaky_relu,
                                                    bn=True, is_training=is_training,
                                                    scope='centroid_edge_attention' + str(multi_i)+flag_scope,
                                                    bn_decay=bn_decay)

            weight_new_edge_feature = tf.matmul(attention_weight_centroid_edge, new_edge_feature)

            # 光滑后的特征存储到一个列表中
            update_feature2_edge_list.append(weight_new_edge_feature)
            update_centroid_feature_list.append(sharp_centroid_mixture)

    return update_feature2_edge_list, update_centroid_feature_list