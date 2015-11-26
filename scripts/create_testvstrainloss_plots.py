"""
(c) November 2015 by Daniel Seita

Creates test/train loss plots (using matplotlib), mirroring Figure 7 of the Neural Networks paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import sys

def get_train_test_accuracies(file_name):
    """
    The input is the file that contains caffe's output log for 1000 trials, using a given number of
    hidden units. Then this parses the file and searches for where we hit 200 iterations, upon which
    the code will extract the train loss, the test accuracy, and the test loss. We store these in
    three respective lists that we return, so we can plot them.
    """
    with open(file_name, 'r') as r002:
        train_losses = []
        test_losses = []
        accuracies = []
        lines_iter = iter(r002)
        for line in lines_iter:
            line_split = line.split('=')
            if ('solver.cpp:320] Iteration 200, loss ' in line_split[0]):
                train_losses.append(float(line_split[1].strip()))
                nextline = next(lines_iter)
                while ('Test net output #0: accuracy' not in nextline):
                    nextline = next(lines_iter)
                accuracies.append(float(nextline.split('=')[1].strip()))
                nextline = next(lines_iter)
                while ('Test net output #1: loss' not in nextline):
                    nextline = next(lines_iter)
                test_losses.append(float(nextline.split('=')[1].split()[0]))
        assert len(train_losses) == len(test_losses) == len(accuracies) == 1000
    return (train_losses, test_losses, accuracies)

########
# MAIN #
########

# Files that contain the caffe output log for 1000 trials. Change directories as needed.
res_002 = '/Users/danielseita/caffe/ee227bt_project/caffe_output/test_002.txt'
res_005 = '/Users/danielseita/caffe/ee227bt_project/caffe_output/test_005.txt'
res_010 = '/Users/danielseita/caffe/ee227bt_project/caffe_output/test_010.txt'
res_025 = '/Users/danielseita/caffe/ee227bt_project/caffe_output/test_025.txt'
res_050 = '/Users/danielseita/caffe/ee227bt_project/caffe_output/test_050.txt'
res_100 = '/Users/danielseita/caffe/ee227bt_project/caffe_output/test_100.txt'
res_250 = '/Users/danielseita/caffe/ee227bt_project/caffe_output/test_250.txt'
res_500 = '/Users/danielseita/caffe/ee227bt_project/caffe_output/test_500.txt'
res_1k = '/Users/danielseita/caffe/ee227bt_project/caffe_output/test_1k.txt'

files = [res_002, res_005, res_010, res_025, res_050, res_100, res_250, res_500, res_1k]
nums = [2, 5, 10, 25, 50, 100, 250, 500, 1000]
f, axarray = plt.subplots(3, 3)

# Go through each [x,y] coordinate, compute the desired statistics, and format the plot nicely.
# Can interchange 'test_losses' and 'accuracies'.
for x in range(3):
    for y in range(3):
        (train_losses, test_losses, accuracies) = get_train_test_accuracies(files[3*x + y])
        axarray[x][y].scatter(train_losses, accuracies, s=10, c='r')
        pears_corr = pearsonr(train_losses, accuracies)[0]
        pears_corr_trunc = "{0:.4f}".format(pears_corr)
        title = 'N = ' + str(nums[3*x+y]) + ', c = ' + str(pears_corr_trunc)
        axarray[x][y].set_title(title, fontsize='xx-small')
        axarray[x][y].set_xlabel('Train Loss', fontsize='xx-small')
        axarray[x][y].set_ylabel('Accuracy', fontsize='xx-small')
        axarray[x][y].tick_params(axis='x', labelsize=5)
        axarray[x][y].tick_params(axis='y', labelsize=5)
f.tight_layout()
plt.savefig('fig_trainvsacc_plots.png', dpi=240)

