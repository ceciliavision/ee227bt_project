"""
(c) November 2015 by Daniel Seita

Before running this code, we first train a NN so we have our weight files ready.
Then we use this for finite difference computation of the Hessian. Obviously, 
don't run this on anything larger than a one-layer FC network. ;)

Note I: I'm assuming we have a numpy array and a text file, respectively, called:

downscaled_60000_images.npy
mnist_labels_train.txt

in the directory. The first has 60000 training images. The shape of this numpy 
array is (60000,10,10,1) and we can just index into that to get (10,10,1)-dim. 
images, which caffe needs because that "1" indicates the number of RGB channels.
The text file has the labels, which are *in order* with the 60000 images from
the numpy array.

Note II: We do not use the testing data at all here.
"""

import numpy as np
import random
import sys
caffe_root = "../caffe"
sys.path.insert(0, caffe_root + "python")
import caffe
caffe.set_mode_gpu()
from os import listdir


def convert_to_vector(net, N):
    """
    Given the current net (with weights), we concatenate the weights together
    into an N-dimensional vector of shape (N,). This let us later call
    np.linalg.norm(w) to get the L2 norm (which we then square).
    """
    weights1 = net.params['ip1'][0].data
    weights2 = net.params['ip2'][0].data
    bias1 = net.params['ip1'][1].data
    bias2 = net.params['ip2'][1].data
    w1 = np.reshape(weights1, (weights1.size,))
    w2 = np.reshape(weights2, (weights2.size,))
    w = np.concatenate((w1,w2,bias1,bias2))
    assert w.size == N
    return w

def convert_to_params(w):
    """
    Given a weight vector w representing all the weights, we convert this back
    to the four individual weight arrays that make up the net. Obviously, this
    assumes we know the structure of the NN in advance! It returns four things.
    NOTE: we assume that the concatenation of (w1, w2, bias1, bias2) = w.
    """
    w1 = w[:100*10]
    w2 = w[100*10:100*10+10*10]
    bias1 = w[100*10+10*10:100*10+10*10+10]
    bias2 = w[100*10+10*10+10:]
    weights1 = np.resize(w1, (10,100))
    weights2 = np.resize(w2, (10,10))
    assert bias1.shape == (10,)
    assert bias2.shape == (10,)
    return (weights1, weights2, bias1, bias2)

def compute_loss(N, net, w, Napprox, images, labels):
    """
    Given a net which has NOT been overrided with vector w yet, we must compute
    (approximate) the loss function. In other words, 'net' has the default weights
    from the caffe model. We are modifying it to see how it changes the loss. This
    is called from compute_hessian(...), which modifies w with epsilons.
    """
    indices = random.sample(range(0,N), Napprox)
    subset_images = images[indices]
    subset_labels = labels[indices]
    total = 0.0
    # TODO compute sum of losses, should probably vectorize
    total = total / float(Napprox) # Don't forget that we are averaging!
    # TODO add the regularization term
    return 0.0

def compute_hessian(N, net, Napprox, images, labels):
    """
    Computes the Hessian (an NxN numpy 2-D array) of the current net. Because
    this can be expensive, we average over Napprox elements, rather than the
    full N training elements.
    """
    hessian = np.zeros((N,N))
    ep1 = 0.00001
    ep2 = 0.00001
    w = convert_to_vector(net, N)

    for x in range(N):
        for y in range(x, N):

            # Form the eps vectors, which we add/subtract to the weight vector.
            ex = np.zeros((N,))
            ex[x] = ep1
            ey = np.zeros((N,))
            ey[y] = ep2

            # Compute f_xy \approx (L(w1) - L(w2) - L(w3) + L(w4))/(4*ep1*ep2)
            Loss1 = compute_loss(N, net, w + ex + ey, Napprox, images, labels)
            Loss2 = compute_loss(N, net, w - ex + ey, Napprox, images, labels)
            Loss3 = compute_loss(N, net, w + ex - ey, Napprox, images, labels)
            Loss4 = compute_loss(N, net, w - ex - ey, Napprox, images, labels)
            val = (Loss1 - Loss2 - Loss3 + Loss4) / (4.0*ep1*ep2)
            hessian[x,y] = val
            hessian[y,x] = val # Note: the above should work fine with x=y

    hessian = (hessian + hessian.T)/2.0 # For enforcing better symmetry
    return hessian

########
# MAIN #
########

# First, download the deployment and net files, and prepare images.
# We can get predictions ((N x 10)-dimensional) by calling net.predict(IMAGES).
DEPLOYMENT_FILE = "downscaled_deployment.prototxt"
PRETRAINED_FILE = "caffe_output/_iter_1000.caffemodel"
net = caffe.Classifier(model_file = DEPLOYMENT_FILE,
                       pretrained_file = PRETRAINED_FILE)
IMAGES = np.load('downscaled_60000_images.npy')
labels = np.loadtxt('mnist_labels_train.txt')
print "Done loading images and labels. len(IMAGES) = {}, len(labels) = {}".format(len(IMAGES), len(labels))

# A debugging message to clarify that the number of weights is 1000+100+10+10=1120.
# We'll obviously need to generalize if we extend this to other types of networks
print "Here are the net.params:"
print [(k, v[0].data.shape) for k, v in net.params.items()]
print [(k, v[1].data.shape) for k, v in net.params.items()]

# Now compute the Hessian, then check eigenvalues.
# We'll be lame and input 1120, but we'll change later for generalization.
approx = 100
hess = compute_hessian(1120, net, approx, IMAGES, labels)
(eigvals,eigvecs) = np.linalg.eig(hess)
negeigs = eigvals[eigvals < 0] # We might consider using 0.001 like in the paper
print "Out of {}, there are {} negative eigenvalues".format(len(eigvals), len(negeigs))

