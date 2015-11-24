"""
(c) November 2015 by Daniel Seita

This file will take in:

mnist_data_train.txt
mnist_data_test.txt
mnist_labels_train.txt
mnist_labels_test.txt

And produce the files we need for convert_imageset.cpp in caffe. These are:

(1) The output directory that contains all the *.png files.

(2) Two plain text files, one for training and one for testing. These must list uniquely the images
in the training and testing sets, *along with* (don't forget this!!!) the index of the class. Here,
it's 0 through 9 but remember that if for some reason we only do, say, 4 and 5, those have to be 0
and 1 for caffe to understand what we're doing.

We assume we are going to downscale images to 10x10, following Choromanska et al (2015).

Then the commands for CAFFE are:

./build/tools/convert_imageset.bin  --logtostderr=1  --shuffle --gray \
ee227bt_project/MNIST_Stuff/png_files/ \
ee227bt_project/MNIST_Stuff/convertimageset_train.txt \
digits_downscaled_train_lmdb

with changes to the file names as appropriate based on training or testing.
"""

import sys
import numpy as np
import scipy.misc
from PIL import Image

def create_lmdb_input_file(numpy_data, labels_file, text_file, image_directory, train):
    """
    Given the data and labels file (which coincide, so line i in both files mean the data and label,
    respectively), we form the necessary materials for the lmdb part. We create the text file to
    list images along with their classes, and we save the actual pngs to a different directory.
    """
    with open(labels_file, 'r') as labels, open(text_file, 'w') as text:
        for (index, (image_array, label)) in enumerate(zip(numpy_data, labels)):
            name = 'digits_' + str(label).rstrip() + '_'
            if train:
                name += 'train_' + str.zfill(str(index),5) + '.png'
            else:
                name += 'test_' + str.zfill(str(index),5) + '.png'
            text.write(name + ' ' + str(label)) # No newline needed b/c 'label' already has one.
            #newimage = Image.new('L', (28,28))
            #newimage.putdata(image_array)
            #newimage.save(image_directory + name)
            image_array.resize((28,28))
            resized = scipy.misc.imresize(image_array, (10,10), interp='bilinear') # Change if needed
            resized.resize((100,1))
            res_image = Image.new('L', (10,10))
            res_image.putdata(resized)
            res_image.save(image_directory + name)


########
# MAIN #
########

# Change these settings as needed.
root = '/Users/danielseita/caffe/ee227bt_project/MNIST_Stuff/'
train_data = np.loadtxt(root + 'mnist_data_train.txt')
create_lmdb_input_file(train_data, 
                       root + 'mnist_labels_train.txt',
                       root + 'convertimageset_train.txt',
                       root + 'png_files/',
                       True)
test_data = np.loadtxt(root + 'mnist_data_test.txt')
create_lmdb_input_file(test_data,
                       root + 'mnist_labels_test.txt',
                       root + 'convertimageset_test.txt',
                       root + 'png_files/',
                       False)
