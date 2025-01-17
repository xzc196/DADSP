# -*- coding: utf-8 -*-
# @Time    : 2020/2/14 20:59
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : GRL.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.python.framework import ops

class GradientReversalLayer(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y

GRL = GradientReversalLayer()