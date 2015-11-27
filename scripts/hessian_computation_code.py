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


def compute_squared_l2(net):
    """
    Given a network, we'll return ||w||_2^2, made simpler with numpy since the
    np.linalg.norm works on full matrices.
    """
    n_w1 = np.linalg.norm(net.params['ip1'][0].data)
    n_w2 = np.linalg.norm(net.params['ip2'][0].data)
    n_b1 = np.linalg.norm(net.params['ip1'][1].data)
    n_b2 = np.linalg.norm(net.params['ip2'][1].data)
    return n_w1*n_w1 + n_w2*n_w2 + n_b1*n_b1 + n_b2*n_b2

def tweak_weights(net, index, eps):
    """
    Given a net, plus an index, we find the correct weight spot to increment by
    eps (we are always *incrementing*, if it's decrementing, then we should have
    changed the input to this function to negate it). This obviously assumes a
    known structure to the network! This changes the internal state of the net
    via its data arrays. If it's the first two cases, we have to find the correct
    (row, col) combination to index into the weight vectors.
    """
    T1 = 10*100
    T2 = 10*100 + 10*10
    T3 = 10*100 + 10*10 + 10
    if (index < T1): # 'index' is between 0 and 1000, inclusive
        row = index / 100 # This is between 0 and 9
        col = index % 100 # This is between 0 and 100
        val = net.params['ip1'][0].data[row, col]
        net.params['ip1'][0].data[row, col] = val + eps
    elif (index < T2):
        ind = index - T1 # Will be between 0 and 99, inclusive
        row = ind / 10
        col = ind % 10
        val = net.params['ip2'][0].data[row, col]
        net.params['ip2'][0].data[row, col] = val + eps
    elif (index < T3):
        ind = index - T2
        val = net.params['ip1'][1].data[ind]
        net.params['ip1'][1].data[ind] = val + eps
    else:
        ind = index - T3
        val = net.params['ip2'][1].data[ind]
        net.params['ip2'][1].data[ind] = val + eps

def compute_loss(net, x, y, epx, epy, Napprox, images, labels):
    """
    Given a net which has NOT had weights changed (i.e., weights are straight
    from the caffe model) we must approximate the loss function. The x and y are
    indices of the weights that we have to increment or decrement by epsilon. We
    add w[x] by epx and w[y] by epy, which may be negative from external calls.
    """
    # We will only use a random subset of the 60000 total images.
    indices = random.sample(range(0,60000), Napprox)
    subset_images = images[indices]
    subset_labels = [int(x) for x in labels[indices]]

    # Change weights for this net, then get predictions. Deal with x, then y:
    tweak_weights(net, x, epx)
    tweak_weights(net, y, epy)
    predictions = net.predict(subset_images)
    reg = 0.0005*compute_squared_l2(net)  # Don't forget!

    # Predictions done, so make the weight vectors back to their original values
    tweak_weights(net, x, -epx)
    tweak_weights(net, y, -epy)

    # Compute the losses from the predictions. Hopefully this is fast indexing.
    total = sum(-np.log(predictions[np.arange(Napprox),subset_labels]))
    total /= float(Napprox) # Don't forget to average
    total += reg # Add regularizaton (TODO I'm assuming \lambda=1)
    return total

def compute_hessian(N, net, Napprox, images, labels):
    """
    Computes the Hessian (an NxN numpy 2-D array) of the current net. Because
    this can be expensive, we average over Napprox elements, rather than the
    full 60000 training elements. Our loss function is "compute_loss(...)".
    
    Here, N=1120. The first 10*100 are for ip1 weights. The next 10*10 are for
    the ip2 weights. Then the last 20 are for the bias1 and bias2 in that order.
    """
    hessian = np.zeros((N,N))
    ep = 0.00001

    # Now iterate; 'x' and 'y' indicate indices of the weights that we increment/decrement by epsilon
    for x in range(N):
        #if (x % 100 == 0):
        print "Done with {} out of {} for Hessian.".format(x,N)
        for y in range(x, N):
            # Compute f_xy \approx (L(w1) - L(w2) - L(w3) + L(w4))/(4*ep1*ep2)
            Loss1 = compute_loss(net, x, y,  ep,  ep, Napprox, images, labels)
            Loss2 = compute_loss(net, x, y, -ep,  ep, Napprox, images, labels)
            Loss3 = Loss2
            if (x != y): # Save time if x==y
                Loss3 = compute_loss(net, x, y, ep, -ep, Napprox, images, labels)
            Loss4 = compute_loss(net, x, y, -ep, -ep, Napprox, images, labels)
            val = (Loss1 - Loss2 - Loss3 + Loss4) / (4.0*ep*ep)
            hessian[x,y] = val
            hessian[y,x] = val # Note: the above should work fine with x=y

    hessian = (hessian + hessian.T)/2.0 # For enforcing better symmetry
    return hessian

########
# MAIN #
########

# First, download the deployment and net files, and also get images.
# We can get predictions ((N x 10)-dimensional) by calling net.predict(IMAGES).
DEPLOYMENT_FILE = "downscaled_deployment.prototxt"
PRETRAINED_FILE = "caffe_output/_iter_1000.caffemodel"
net = caffe.Classifier(model_file = DEPLOYMENT_FILE, pretrained_file = PRETRAINED_FILE)
IMAGES = np.load('downscaled_60000_images.npy')
labels = np.loadtxt('mnist_labels_train.txt')
print "Done loading images and labels. len(IMAGES) = {}, len(labels) = {}".format(len(IMAGES), len(labels))

# A debugging message to clarify that the number of weights is 1000+100+10+10=1120.
# We'll obviously need to generalize if we extend this to other types of networks
print "Just as a sanity check, here are the net.params:"
print [(k, v[0].data.shape) for k, v in net.params.items()]
print [(k, v[1].data.shape) for k, v in net.params.items()]

# Now compute the Hessian, then check eigenvalues.
# We'll be lame and input 1120 directly, but we'll change later for generalization.
approx = 100
N = 1120
print "Now computing the Hessian ..."
hess = compute_hessian(N, net, approx, IMAGES, labels)
np.save('hess.npy', hess)
print "Hessian computation done. Now computing eigenvalues ..."
(eigvals,eigvecs) = np.linalg.eig(hess)
np.savetxt('eigenvalues.txt', eigvals)
negeigs = eigvals[eigvals < 0] # We might consider using 0.001 like in the paper
print "Out of {}, there are {} negative eigenvalues".format(len(eigvals), len(negeigs))

