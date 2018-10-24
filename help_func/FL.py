import tensorflow as tf


def focal_loss(onehot_labels, logits, alpha=0.25, gamma=2.0):
    """
    Compute sigmoid focal loss between logits and onehot labels: focal loss = -(1-pt)^gamma*log(pt)
    Args:
        onehot_labels: onehot labels with shape (batch_size, num_anchors, num_classes)
        logits: last layer feature output with shape (batch_size, num_anchors, num_classes)
        alpha: The hyperparameter for adjusting biased samples, default is 0.25
        gamma: The hyperparameter for penalizing the easy labeled samples, default is 2.0
    Returns:
        a scalar of focal loss of total batch of anchors
    """
    with tf.name_scope("focal_loss"):
        logits = tf.cast(logits, tf.float32)
        onehot_labels = tf.cast(onehot_labels, tf.float32)
        ce = tf.losses.softmax_cross_entropy(onehot_labels, logits=logits)
        predictions = tf.nn.sigmoid(logits)
        predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)# ensure other position is zero!
        # add small value to avoid 0
        alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
        alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
        weighted_loss = ce * tf.pow(1-predictions_pt, gamma) * alpha_t
        return tf.reduce_sum(weighted_loss)
    #just conduct a experiment: using alpha_t or alpha