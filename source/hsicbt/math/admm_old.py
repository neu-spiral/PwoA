from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import operator
from numpy import linalg as LA
import yaml
import random
from ..utils.testers import *
from itertools import combinations
import collections

class ADMM:
    def __init__(self, config_dict, model, config, rho=1):
        self.ADMM_U = {}
        self.ADMM_Z = {}
        self.rho = rho
        self.rhos = {}
        self.prune_ratio = config_dict['prune_ratio']
        
        self.init(config, model)

    def init(self, config, model):
        """
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file

        """
        if not isinstance(config, str):
            raise Exception("filename must be a str")
        with open(config, "r") as stream:
            try:
                raw_dict = yaml.load(stream)
                self.prune_ratios = raw_dict['prune_ratios']
                for k, v in self.prune_ratios.items():
                    self.rhos[k] = self.rho
                    
                # if pruning ratio is given
                if isinstance(self.prune_ratio, float):
                    for k in self.prune_ratios:
                        self.prune_ratios[k] = self.prune_ratio
                print(self.prune_ratios)    
                for (name, W) in model.named_parameters():
                    if name not in self.prune_ratios:
                        continue
                    self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
                    self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z
                    # if(len(W.size()) == 4):
                    #     if name not in self.prune_ratios:
                    #         continue
                    #     self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
                    #     self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z


            except yaml.YAMLError as exc:
                print(exc)


def random_pruning(weight, prune_ratio, sparsity_type):
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    if (sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        indices = np.random.choice(shape2d[0], int(shape2d[0] * prune_ratio), replace=False)
        weight2d[indices, :] = 0
        weight = weight2d.reshape(shape)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = i not in indices
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise Exception("not implemented yet")


def L1_pruning(weight, prune_ratio, sparsity_type):
    """
    projected gradient descent for comparison

    """
    percent = prune_ratio * 100
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    row_l1_norm = LA.norm(weight2d, 1, axis=1)
    percentile = np.percentile(row_l1_norm, percent)
    under_threshold = row_l1_norm < percentile
    above_threshold = row_l1_norm > percentile
    weight2d[under_threshold, :] = 0
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[0]):
        expand_above_threshold[i, :] = above_threshold[i]
    weight = weight.reshape(shape)
    expand_above_threshold = expand_above_threshold.reshape(shape)
    return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()


def weight_pruning(weight, prune_ratio, sparsity_type):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    """

    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    percent = prune_ratio * 100
    if (sparsity_type == "irregular"):
        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "column"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        percentile = np.percentile(column_l2_norm, percent)
        under_threshold = column_l2_norm < percentile
        above_threshold = column_l2_norm > percentile
        weight2d[:, under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[1]):
            expand_above_threshold[:, i] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        weight = weight.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "channel"):
        shape = weight.shape
        print("channel pruning...", weight.shape)
        weight3d = weight.reshape(shape[0], shape[1], -1)
        channel_l2_norm = LA.norm(weight3d, 2, axis=(0,2))
        percentile = np.percentile(channel_l2_norm, percent)
        under_threshold = channel_l2_norm <= percentile
        above_threshold = channel_l2_norm > percentile
        weight3d[:,under_threshold,:] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(weight3d.shape, dtype=np.float32)
        for i in range(weight3d.shape[1]):
            expand_above_threshold[:, i, :] = above_threshold[i]
        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "filter"):
        if len(weight.shape) == 2:
            weight = np.transpose(weight)
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm <= percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        # weight2d[weight2d < 1e-40] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = above_threshold[i]
        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        if len(weight.shape) == 2:
            weight = np.transpose(weight)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "bn_filter"):
        ## bn pruning is very similar to bias pruning
        weight_temp = np.abs(weight)
        percentile = np.percentile(weight_temp, percent)
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "aligned_pattern"):
        print("aligned pattern pruning...", weight.shape)
        shape = weight.shape

        """pattern WM mobile"""
        pattern1 = [[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]]

        pattern2 = [[0, 0], [0, 2], [1, 0], [1, 2], [2, 0]]
        pattern3 = [[0, 0], [0, 1], [0, 2], [2, 1], [2, 2]]
        pattern4 = [[0, 2], [1, 0], [1, 2], [2, 0], [2, 2]]
        pattern5 = [[0, 0], [0, 1], [2, 0], [2, 1], [2, 2]]
        pattern6 = [[0, 0], [0, 2], [1, 0], [1, 2], [2, 2]]
        pattern7 = [[0, 1], [0, 2], [2, 0], [2, 1], [2, 2]]
        pattern8 = [[0, 0], [1, 0], [1, 2], [2, 0], [2, 2]]
        pattern9 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 1]]

        pattern10 = [[0, 1], [0, 2], [1, 2], [2, 0], [2, 2]]
        pattern11 = [[0, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
        pattern12 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 1]]
        pattern13 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 2]]
        pattern14 = [[0, 0], [0, 1], [1, 0], [2, 0], [2, 2]]
        pattern15 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 0]]
        pattern16 = [[0, 0], [0, 2], [1, 2], [2, 1], [2, 2]]
        pattern17 = [[0, 2], [1, 0], [2, 0], [2, 1], [2, 2]]

        pattern18 = [[0, 0], [0, 2], [2, 0], [2, 1], [2, 2]]
        pattern19 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 2]]
        pattern20 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]
        pattern21 = [[0, 0], [0, 2], [1, 2], [2, 0], [2, 2]]

        patterns_dict = {1: pattern1,
                         2: pattern2,
                         3: pattern3,
                         4: pattern4,
                         5: pattern5,
                         6: pattern6,
                         7: pattern7,
                         8: pattern8,
                         9: pattern9,
                         10: pattern10,
                         11: pattern11,
                         12: pattern12,
                         13: pattern13,
                         14: pattern14,
                         15: pattern15,
                         16: pattern16,
                         17: pattern17,
                         18: pattern18,
                         19: pattern19,
                         20: pattern20,
                         21: pattern21
                         }

        # weight3d = weight.reshape(weight.shape[0], weight.shape[1], -1)
        for i in range(shape[1]): # loop channel
            current_channel = weight[:,i,:,:].copy()
            temp_dict = {}  # store each pattern's norm value on the same weight kernel
            for key, pattern in patterns_dict.items():
                temp_channel = current_channel.copy()
                for j in range(temp_channel.shape[0]): # loop every kernel in a channel
                    for index in pattern:
                        temp_channel[j,:][index[0], index[1]] = 0
                current_norm = LA.norm(temp_channel)
                temp_dict[key] = current_norm
            best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]
            for index in patterns_dict[best_pattern]:
                weight[:, i, index[0], index[1]] = 0
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        # zeros = weight == 0
        # zeros = zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "pattern"):
        print("pattern pruning...", weight.shape)
        shape = weight.shape

        """pattern Tsinghua hardware (5 pattern)"""
        # pattern1 = [[0, 0], [0, 2], [2, 0], [2, 2]]
        # pattern2 = [[0, 0], [0, 1], [2, 1], [2, 2]]
        # pattern3 = [[0, 0], [0, 1], [2, 0], [2, 1]]
        # pattern4 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        #
        # pattern5 = [[0, 2], [1, 0], [1, 2], [2, 0]]
        # pattern6 = [[0, 0], [1, 0], [1, 2], [2, 2]]
        # pattern7 = [[0, 1], [0, 2], [2, 0], [2, 1]]
        # pattern8 = [[0, 1], [0, 2], [2, 1], [2, 2]]
        #
        # pattern9 = [[1, 0], [1, 2], [2, 0], [2, 2]]
        # pattern10 = [[0, 0], [0, 2], [1, 0], [1, 2]]
        # pattern11 = [[1, 1], [1, 2], [2, 1], [2, 2]]
        # pattern12 = [[1, 0], [1, 1], [2, 0], [2, 1]]
        # pattern13 = [[0, 1], [0, 2], [1, 1], [1, 2]]
        #
        # patterns_dict = {1 : pattern1,
        #                  2 : pattern2,
        #                  3 : pattern3,
        #                  4 : pattern4,
        #                  5 : pattern5,
        #                  6 : pattern6,
        #                  7 : pattern7,
        #                  8 : pattern8,
        #                  9 : pattern9,
        #                  10 : pattern10,
        #                  11 : pattern11,
        #                  12 : pattern12,
        #                  13 : pattern13
        #                  }

        """pattern WM mobile (4 pattern)"""
        # pattern1 = [[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]]
        #
        # pattern2 = [[0, 0], [0, 2], [1, 0], [1, 2], [2, 0]]
        # pattern3 = [[0, 0], [0, 1], [0, 2], [2, 1], [2, 2]]
        # pattern4 = [[0, 2], [1, 0], [1, 2], [2, 0], [2, 2]]
        # pattern5 = [[0, 0], [0, 1], [2, 0], [2, 1], [2, 2]]
        # pattern6 = [[0, 0], [0, 2], [1, 0], [1, 2], [2, 2]]
        # pattern7 = [[0, 1], [0, 2], [2, 0], [2, 1], [2, 2]]
        # pattern8 = [[0, 0], [1, 0], [1, 2], [2, 0], [2, 2]]
        # pattern9 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 1]]
        #
        # pattern10 = [[0, 1], [0, 2], [1, 2], [2, 0], [2, 2]]
        # pattern11 = [[0, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
        # pattern12 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 1]]
        # pattern13 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 2]]
        # pattern14 = [[0, 0], [0, 1], [1, 0], [2, 0], [2, 2]]
        # pattern15 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 0]]
        # pattern16 = [[0, 0], [0, 2], [1, 2], [2, 1], [2, 2]]
        # pattern17 = [[0, 2], [1, 0], [2, 0], [2, 1], [2, 2]]
        #
        # pattern18 = [[0, 0], [0, 2], [2, 0], [2, 1], [2, 2]]
        # pattern19 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 2]]
        # pattern20 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]
        # pattern21 = [[0, 0], [0, 2], [1, 2], [2, 0], [2, 2]]
        #
        # patterns_dict = {1: pattern1,
        #                  2: pattern2,
        #                  3: pattern3,
        #                  4: pattern4,
        #                  5: pattern5,
        #                  6: pattern6,
        #                  7: pattern7,
        #                  8: pattern8,
        #                  9: pattern9,
        #                  10: pattern10,
        #                  11: pattern11,
        #                  12: pattern12,
        #                  13: pattern13,
        #                  14: pattern14,
        #                  15: pattern15,
        #                  16: pattern16,
        #                  17: pattern17,
        #                  18: pattern18,
        #                  19: pattern19,
        #                  20: pattern20,
        #                  21: pattern21
        #                  }

        # distilled pattern
        # pattern1 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 2]]
        #
        # pattern2 = [[0, 0], [0, 2], [2, 0], [2, 1], [2, 2]]
        # pattern3 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 0]]
        # pattern4 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 1]]
        # pattern5 = [[0, 1], [0, 2], [2, 0], [2, 1], [2, 2]]
        # pattern6 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]
        # pattern7 = [[1, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
        # pattern8 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 2]]
        # pattern9 = [[0, 0], [1, 0], [1, 1], [1, 2], [2, 0]]
        #
        # pattern10 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 1]]
        # pattern11 = [[0, 0], [0, 1], [2, 0], [2, 1], [2, 2]]
        # pattern12 = [[0, 2], [1, 0], [1, 1], [1, 2], [2, 2]]
        # pattern13 = [[1, 0], [1, 1], [1, 2], [2, 0], [2, 1]]
        # pattern14 = [[0, 0], [1, 0], [1, 1], [1, 2], [2, 2]]
        # pattern15 = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]
        #
        # patterns_dict = {1: pattern1,
        #                  2: pattern2,
        #                  3: pattern3,
        #                  4: pattern4,
        #                  5: pattern5,
        #                  6: pattern6,
        #                  7: pattern7,
        #                  8: pattern8,
        #                  9: pattern9,
        #                  10: pattern10,
        #                  11: pattern11,
        #                  12: pattern12,
        #                  13: pattern13,
        #                  14: pattern14,
        #                  15: pattern15
        #                  }

        pattern1 = [[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]]

        pattern2 = [[0, 0], [0, 2], [2, 0], [2, 1], [2, 2]]
        pattern3 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 0]]
        pattern4 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 1]]
        pattern5 = [[0, 1], [0, 2], [2, 0], [2, 1], [2, 2]]
        pattern6 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]
        pattern7 = [[1, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
        pattern8 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 2]]
        pattern9 = [[0, 0], [1, 0], [1, 1], [1, 2], [2, 0]]

        pattern10 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
        pattern11 = [[0, 0], [0, 1], [2, 0], [2, 1], [2, 2]]
        pattern12 = [[1, 0], [1, 1], [1, 2], [2, 0], [2, 1]]

        patterns_dict = {1: pattern1,
                         2: pattern2,
                         3: pattern3,
                         4: pattern4,
                         5: pattern5,
                         6: pattern6,
                         7: pattern7,
                         8: pattern8,
                         9: pattern9,
                         10: pattern10,
                         11: pattern11,
                         12: pattern12
                         }


        for i in range(shape[0]):
            for j in range(shape[1]):
                current_kernel = weight[i, j, :, :].copy()
                temp_dict = {}  # store each pattern's norm value on the same weight kernel
                for key, pattern in patterns_dict.items():
                    temp_kernel = current_kernel.copy()
                    for index in pattern:
                        temp_kernel[index[0], index[1]] = 0
                    current_norm = LA.norm(temp_kernel)
                    temp_dict[key] = current_norm
                best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]
                # print(best_pattern)
                for index in patterns_dict[best_pattern]:
                    weight[i, j, index[0], index[1]] = 0
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        # zeros = weight == 0
        # zeros = zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "filter_balance"):
        print("pruning filter with balanced outputs")
        kth_smallest = prune_ratio  # the percent from script is used to represent k-th smallest l2-norm kernel will be pruned in each filter
        shape = weight.shape
        weight3d = weight.reshape(shape[0], shape[1], -1)
        for i in range(shape[0]):
            kernel_l2norm_list = LA.norm(weight3d[i,:,:], 2, axis=1)
            partial_sorted_index = np.argpartition(kernel_l2norm_list, kth_smallest)  # list of all indices, but partially sorted
            kth_smallest_index = partial_sorted_index[:kth_smallest]  # indices of k-th smallest l2-norm
            for idx in kth_smallest_index:
                weight3d[i, idx, :] = 0
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "two_filter_balance"):
        pattern1 = [[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]]

        pattern2 = [[0, 0], [0, 2], [2, 0], [2, 1], [2, 2]]
        pattern3 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 0]]
        pattern4 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 1]]
        pattern5 = [[0, 1], [0, 2], [2, 0], [2, 1], [2, 2]]
        pattern6 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]
        pattern7 = [[1, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
        pattern8 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 2]]
        pattern9 = [[0, 0], [1, 0], [1, 1], [1, 2], [2, 0]]

        pattern10 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
        pattern11 = [[0, 0], [0, 1], [2, 0], [2, 1], [2, 2]]
        pattern12 = [[1, 0], [1, 1], [1, 2], [2, 0], [2, 1]]

        patterns_dict = {1: pattern1,
                         2: pattern2,
                         3: pattern3,
                         4: pattern4,
                         5: pattern5,
                         6: pattern6,
                         7: pattern7,
                         8: pattern8,
                         9: pattern9,
                         10: pattern10,
                         11: pattern11,
                         12: pattern12
                         }

        print("pruning two filter with balanced outputs")

        shape = weight.shape
        numFilter = shape[0]

        weight2d = weight.reshape(shape[0], -1)
        filter_L2 = LA.norm(weight2d, 2, axis=1)
        weight3d = weight.reshape(shape[0], shape[1], -1)

        filter_index_dict = {}
        for index, l2_item in enumerate(filter_L2):
            filter_index_dict[index] = l2_item

        print(filter_index_dict)
        filter_index_dict = sorted(filter_index_dict.items(), key=lambda k: [k[1], k[0]])
        filter_index_dict = collections.OrderedDict(filter_index_dict)
        print(filter_index_dict)


        for i, (filter_idx, _) in enumerate(filter_index_dict.items()):
            if i % 2 == 0:
                first_idx = filter_idx
                second_idx = list(filter_index_dict.keys())[i+1]
                temp = np.array([weight3d[first_idx, :, :], weight3d[second_idx, : ,:]])
                print(temp)

                """add aligned pattern prune for this current pair before aligned connectivity prune"""
                temp = temp.reshape([temp.shape[0], temp.shape[1], 3, 3])
                for k in range(temp.shape[1]):  # loop channel
                    current_channel = temp[:, k, :, :].copy()
                    temp_dict = {}  # store each pattern's norm value on the same weight kernel
                    for key, pattern in patterns_dict.items():
                        temp_channel = current_channel.copy()
                        for j in range(temp_channel.shape[0]):  # loop every kernel in a channel
                            for index in pattern:
                                temp_channel[j, :][index[0], index[1]] = 0
                        current_norm = LA.norm(temp_channel)
                        temp_dict[key] = current_norm
                    best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]
                    for index in patterns_dict[best_pattern]:
                        temp[:, k, index[0], index[1]] = 0
                temp = temp.reshape([temp.shape[0], temp.shape[1], -1])
                print(temp)

                """aligned connectivity prune"""
                channel_l2_norm = LA.norm(temp, 2, axis=(0, 2))
                if i <= numFilter / 4:
                    percentile = np.percentile(channel_l2_norm, percent / 1.2)
                elif numFilter / 4 < i <= numFilter / 2:
                    percentile = np.percentile(channel_l2_norm, percent / 1.1)
                elif numFilter / 2 < i <= numFilter:
                    percentile = np.percentile(channel_l2_norm, percent)
                under_threshold = channel_l2_norm <= percentile
                above_threshold = channel_l2_norm > percentile
                temp[:, under_threshold, :] = 0
                weight3d[first_idx] = temp[0]
                weight3d[second_idx] = temp[1]
                print(temp)
                print("========\n\n")

        weight = weight3d.reshape(shape)

        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise SyntaxError("Unknown sparsity type")

def hard_prune(ADMM, model, sparsity_type, option=None):
    """
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda

    """

    for (name, W) in model.named_parameters():
        if name not in ADMM.prune_ratios:  # ignore layers that do not have rho
            continue
        cuda_pruned_weights = None
        if option == None:
            _, cuda_pruned_weights = weight_pruning(W, ADMM.prune_ratios[name], sparsity_type)  # get sparse model in cuda
        elif option == "random":
            _, cuda_pruned_weights = random_pruning(W, ADMM.prune_ratios[name], sparsity_type)

        elif option == "l1":
            _, cuda_pruned_weights = L1_pruning(W, ADMM.prune_ratios[name], sparsity_type)
        else:
            raise Exception("not implmented yet")
        W.data = cuda_pruned_weights  # replace the data field in variable



def admm_initialization(config_dict, ADMM, model):

    sparsity_type = config_dict['sparsity_type']
    for i, (name, W) in enumerate(model.named_parameters()):
        if name in ADMM.prune_ratios:
            i_, updated_Z = weight_pruning(W, ADMM.prune_ratios[name], sparsity_type)  # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
            ADMM.ADMM_Z[name] = updated_Z


def z_u_update(config_dict, ADMM, model, epoch, batch_idx, hsic_grad=None):

    admm_epochs, sparsity_type = config_dict['admm_epochs'], config_dict['sparsity_type']
    if epoch != 1 and (epoch - 1) % admm_epochs == 0 and batch_idx == 0:
        for i, (name, W) in enumerate(model.named_parameters()):
            if name not in ADMM.prune_ratios:
                continue
            if config_dict['multi_rho']:
                 admm_multi_rho_scheduler(ADMM,name) # call multi rho scheduler every admm update
             
            if hsic_grad:
                ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name] + hsic_grad[name]  # Z(k+1) = W(k+1)+U[k]
            else:
                ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name]
            _, updated_Z = weight_pruning(ADMM.ADMM_Z[name], ADMM.prune_ratios[name], sparsity_type)  # equivalent to Euclidean Projection
            ADMM.ADMM_Z[name] = updated_Z
            if hsic_grad:
                ADMM.ADMM_U[name] = W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name] + hsic_grad[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)
            else:
                ADMM.ADMM_U[name] = W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name]

def append_admm_loss(ADMM, model, ce_loss, hsic_grad=None):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm_loss = {}

    for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
        if name not in ADMM.prune_ratios:
            continue
        if hsic_grad and hsic_grad[name] is not None:
            admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name] + hsic_grad[name], p=2) ** 2)
        else:
            admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2) ** 2)
    mixed_loss = 0
    mixed_loss += ce_loss
    for k, v in admm_loss.items():
        mixed_loss += v
    return ce_loss, admm_loss, mixed_loss


def admm_multi_rho_scheduler(ADMM, name):
    """
    It works better to make rho monotonically increasing
    rho: 0.01   ->  50epochs -> 1
         0.0001 -> 100epochs -> 1
    """
    current_rho = ADMM.rhos[name]
    ADMM.rhos[name] = max(1, 1.1*current_rho)  # choose whatever you like
    
def admm_adjust_learning_rate(optimizer, epoch, config_dict):
    """ (The pytorch learning rate scheduler)
Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default 
    admm epoch is 9)

    """
    admm_epoch = config_dict['admm_epochs']
    lr = None
    if epoch % admm_epoch == 0:
        lr = config_dict['learning_rate']
    else:
        admm_epoch_offset = epoch % admm_epoch

        admm_step = admm_epoch / 3  # roughly every 1/3 admm_epoch.

        lr = config_dict['learning_rate'] * (0.1 ** (admm_epoch_offset // admm_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, smooth):
    return lam * criterion(pred, y_a, smooth=smooth) + \
           (1 - lam) * criterion(pred, y_b, smooth=smooth)

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)
    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
