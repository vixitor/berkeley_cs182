import numpy as np

from deeplearning.layer_utils import *

from hw04.q_coding_bn_drop_cnn.code.deeplearning.layers import conv_forward_naive, max_pool_forward_naive, relu_forward, \
    max_pool_backward_naive, conv_backward_naive
from hw0code.deeplearning.layers import affine_forward
from hw2code.q_coding_optimizer_init.code.deeplearning.layers import softmax_loss, affine_backward, relu_backward


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        self.params["W1"] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params["b1"] = np.zeros(num_filters,)
        self.params["W2"] = np.random.randn(num_filters * (H // 2) * (W // 2), hidden_dim) * weight_scale
        self.params["b2"] = np.zeros(hidden_dim, )
        self.params["W3"] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params["b3"] = np.zeros(num_classes, )
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        conv_out, conv_cache = conv_forward_naive(X, W1, b1, conv_param)
        relu1_out, relu1_cache = relu_forward(conv_out)
        pool_out, pool_cache = max_pool_forward_naive(relu1_out, pool_param)
        affine1_out, affine1_cache = affine_forward(pool_out, W2, b2)
        relu2_out, relu2_cache = relu_forward(affine1_out)
        affine2_out, affine2_cache = affine_forward(relu2_out, W3, b3)
        scores = affine2_out

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dx = softmax_loss(scores, y)
        loss += 1 / 2 * self.reg * sum(np.sum(self.params["W" + str(i + 1)] ** 2) for i in range(3))
        dx, dw, db = affine_backward(dx, affine2_cache)
        grads["W3"] = dw + self.reg * W3
        grads["b3"] = db
        dx = relu_backward(dx, relu2_cache)
        dx, dw, db = affine_backward(dx, affine1_cache)
        grads["W2"] = dw + self.reg * W2
        grads["b2"] = db
        dx = max_pool_backward_naive(dx, pool_cache)
        dx = relu_backward(dx, relu1_cache)
        dx, dw, db = conv_backward_naive(dx, conv_cache)
        grads["W1"] = dw + self.reg * W1
        grads["b1"] = db
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


pass
