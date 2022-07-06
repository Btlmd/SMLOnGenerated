"""
Evaluation Scripts
"""
from __future__ import absolute_import
from __future__ import division
from collections import namedtuple, OrderedDict
import imp
from network import mynn
import argparse
import logging
import os
import torch
import time
import numpy as np

from IPython import embed
from config import cfg, assert_and_infer_cfg
import network
import optimizer
from ood_metrics import fpr_at_95_tpr
from tqdm import tqdm

from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import torchvision.transforms as standard_transforms
import math
#from network.deepv3 import BoundarySuppressionWithSmoothing as custom_BoundarySuppressionWithSmoothing



dirname = os.path.dirname(__file__)
pretrained_model_path = os.path.join(dirname, '')

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepR101V3PlusD_OS8',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='selfgen',
                    help='possible datasets for statistics; cityscapes')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')
parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')
parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')

parser.add_argument('--snapshot', type=str, default=pretrained_model_path)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Use Synchronized BN')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--backbone_lr', type=float, default=0.0,
                    help='different learning rate on backbone network')
parser.add_argument('--pooling', type=str, default='mean',
                    help='pooling methods, average is better than max')

parser.add_argument('--ood_dataset_path', type=str,
                    default='/home/nas1_userB/dataset/ood_segmentation/fishyscapes',
                    help='OoD dataset path')

# Anomaly score mode - msp, max_logit, standardized_max_logit
parser.add_argument('--score_mode', type=str, default='standardized_max_logit',
                    help='score mode for anomaly [msp, max_logit, standardized_max_logit]')

# Boundary suppression configs
parser.add_argument('--enable_boundary_suppression', type=bool, default=True,
                    help='enable boundary suppression')
parser.add_argument('--boundary_width', type=int, default=4,
                    help='initial boundary suppression width')
parser.add_argument('--boundary_iteration', type=int, default=4,
                    help='the number of boundary iterations')

# Dilated smoothing configs
parser.add_argument('--enable_dilated_smoothing', type=bool, default=True,
                    help='enable dilated smoothing')
parser.add_argument('--smoothing_kernel_size', type=int, default=7,
                    help='kernel size of dilated smoothing')
parser.add_argument('--smoothing_kernel_dilation', type=int, default=6,
                    help='kernel dilation rate of dilated smoothing')

# FS LostAndFound data structure cannot be transformed to the desired structure
# Therefore, when using it, extract images and masks and store then into lists,
# and substitute images and masks in the code with those in the lists.
parser.add_argument('--fs_lost_and_found', type=bool, default=False)
parser.add_argument('--use_unprocessed_anomaly_scores_from_segment_open', type=bool, default=False)


parser.add_argument('--threshold', type=float, default=math.inf,
                    help='threshold for anormaly')

parser.add_argument('--th_type', type=str, default='tpr')

parser.add_argument('--save_mask_path', type=str,default='./results')

def get_net():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)

    net = network.get_net(args, criterion=None, criterion_aux=None)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)

    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, None, None,
                            args.snapshot, args.restore_optimizer)
        print(f"Loading completed. Epoch {epoch} and mIoU {mean_iu}")
    else:
        raise ValueError(f"snapshot argument is not set!")
    # Load mean and variance from .npy files
    # if args.use_unprocessed_anomaly_scores_from_segment_open:
    #     # if evaluate segment_open, load stats computed using segment_open on the training set
    #     class_mean = np.load(f'stats/{args.dataset}_mean_segopen.npy', allow_pickle=True)
    #     class_var = np.load(f'stats/{args.dataset}_var_segopen.npy', allow_pickle=True)
    # else:
    #     # otherwise, use stats computed using the deepv3plus model on the training set
    class_mean = np.load(f'stats/{args.dataset}_mean.npy', allow_pickle=True)
    class_var = np.load(f'stats/{args.dataset}_var.npy', allow_pickle=True)
    net.module.set_statistics(mean=class_mean.item(), var=class_var.item())

    torch.cuda.empty_cache()
    net.eval()

    return net

def preprocess_image(x, mean_std):
    x = Image.fromarray(x)
    x = standard_transforms.ToTensor()(x)
    x = standard_transforms.Normalize(*mean_std)(x)

    x = x.cuda()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    return x


import sys
def progressBar(i, max, text):
    """
    Print a progress bar during training.
    :param i: index of current iteration/epoch.
    :param max: max number of iterations/epochs.
    :param text: Text to print on the right of the progress bar.
    :return: None
    """
    bar_size = 60
    j = (i + 1) / max
    sys.stdout.write('\r')
    sys.stdout.write(
        f"[{'=' * int(bar_size * j):{bar_size}s}] {int(100 * j)}%  {text}")
    sys.stdout.flush()

def save_anormaly_mask(subfolder, file_name, a_score, th):
    # Configurations
    save_root = args.save_mask_path
    save_file = os.path.join(save_root, subfolder)
    os.makedirs(save_file, exist_ok=True)
    save_file = os.path.join(save_file, file_name)

    val = a_score.squeeze(0).cpu().numpy()
    img = np.full(val.shape, 255, dtype=np.uint8)
    img[val < th] = 0
    img = Image.fromarray(img)
    img.save(save_file)

def save_anormaly_set(subfolder, file_name, a_score, image, orig_mask, th):
    save_root = args.save_mask_path
    save_file = os.path.join(save_root, subfolder)
    os.makedirs(save_file, exist_ok=True)
    save_file = os.path.join(save_file, file_name)

    val = a_score.squeeze(0)
    val = val.cpu().view(*val.shape, 1).expand(*val.shape, 3).numpy()
    img = np.full(val.shape, 255, dtype=np.uint8)
    img[val < th] = 0
    img = np.concatenate((image, orig_mask, img), axis=1)
    img = Image.fromarray(img)
    img.save(save_file)


def iter_over(save=False, th=None):
    anomaly_score_list = []
    ood_gts_list = []

    anormaly_root = "/DATA2/gaoha/liumd/sml/sml/selfgen/selfgen/anomaly"

    cities = os.listdir(anormaly_root)
    for city in cities[:]:
        city_path = os.path.join(anormaly_root, city)
        for image_file in tqdm(os.listdir(os.path.join(city_path, "rgb_v"))[:]):
            image_path = os.path.join(city_path, "rgb_v", image_file)
            mask_path = os.path.join(city_path, "mask_v", image_file)

            # 3 x H x W
            orig_image = np.array(Image.open(image_path).convert('RGB')).astype('uint8')

            orig_mask = Image.open(mask_path)
            mask_copy = np.asarray(orig_mask).astype(np.uint32)
            mask = mask_copy[:,:,0] + mask_copy[:,:,1] * 256 + mask_copy[:,:,2] * 256 * 256

            seg_mask_copy = np.full(mask_copy.shape[:2], -1, dtype=int)
            for k, v in id_to_trainid.items():
                seg_mask_copy[mask == k] = v

            assert seg_mask_copy.min() >= 0, "original image should be fully labelled @ " + mask_path
            assert seg_mask_copy.max() < 23, "General Anormaly @ " +  mask_path
            ood_gts = np.array(seg_mask_copy)

            if not save:
                ood_gts_list.append(np.expand_dims(ood_gts, 0))

            with torch.no_grad():
                image = preprocess_image(orig_image, mean_std)
                main_out, anomaly_score = net(image)
            del main_out

            if not save:
                anomaly_score_list.append(anomaly_score.cpu().numpy())
            else:
                save_anormaly_set(city, image_file, anomaly_score, orig_image, mask_copy.astype(np.uint8), th)
    if not save:
        return ood_gts_list, anomaly_score_list

if __name__ == '__main__':

    args = parser.parse_args()

    print(args.snapshot)
    ############################################################################
    # get Fishyscapes LostAndFound images and masks
    # if args.fs_lost_and_found:
    #     # Load tensorflow fs lost_and_found dataset
    #     print("Loading LostAndFound images and masks")
    #     import bdlb
    #     fs = bdlb.load(benchmark="fishyscapes", download_and_prepare=False)
    #     fs.download_and_prepare('LostAndFound')

    #     import tensorflow_datasets as tfds

    #     ds = tfds.load('fishyscapes/LostAndFound', split='validation')

    #     # Transform tf dataset into images and masks. Store them into lists.
    #     basedata_id_list = []
    #     image_id_list = []
    #     image_list = []
    #     mask_list = []
    #     num_images_in_fs_lost_and_found_ = 100
    #     for i, blob in enumerate(ds.take(num_images_in_fs_lost_and_found_)):
    #         progressBar(i, num_images_in_fs_lost_and_found_,
    #                     'Storing FS LostAndFound images and masks')
    #         basedata_id = blob['basedata_id'].numpy()
    #         image_id = blob['image_id'].numpy()
    #         image = blob['image_left'].numpy()
    #         mask = blob['mask'].numpy()
    #         # map 255 to 2 such that difference between labels is better visible
    #         mask[mask == 255] = 2

    #         basedata_id_list.append(basedata_id)
    #         image_id_list.append(image_id)
    #         image_list.append(image)
    #         mask_list.append(mask[..., 0])

    #     # Free the memory and gpu memory used by tensorflow and tf dataset
    #     del ds
    #     del fs
    #     from numba import cuda
    #     device = cuda.get_current_device()
    #     device.reset()
    ############################################################################
    # ds = np.load('data.npz')
    # image_list = ds['arr_0']
    # mask_list = ds['arr_1']
    # num_images_in_fs_lost_and_found_ = len(image_list)

    # Enable CUDNN Benchmarking optimization
    # torch.backends.cudnn.benchmark = True
    print('Environment Setup')
    random_seed = cfg.RANDOM_SEED
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    args.world_size = 1

    print(f'World Size: {args.world_size}')
    if 'WORLD_SIZE' in os.environ:
        # args.apex = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])
        print("Total world size: ", int(os.environ['WORLD_SIZE']))

    torch.cuda.set_device(args.local_rank)
    print('My Rank:', args.local_rank)
    # Initialize distributed communication
    args.dist_url = args.dist_url + str(8000 + (int(time.time() % 1000)) // 10)

    torch.distributed.init_process_group(backend='nccl',
                                         init_method=args.dist_url,
                                         world_size=args.world_size,
                                         rank=args.local_rank)


    # Build the network
    print("Building the network")
    net = get_net()


    ############################################################################
    # FS LostAndFound
    # if args.fs_lost_and_found:
    #     mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     anomaly_score_list = []
    #     ood_gts_list = []

    #     # Iterate over all images
    #     for i in range(len(image_list)):
    #         progressBar(i, num_images_in_fs_lost_and_found_,
    #                     'Evaluating on FS LostAndFound')

    #         # get the ith mask from the mask list
    #         mask = mask_list[i]

    #         # get the ith image from the image list
    #         image = image_list[i]
    #         image = image.astype('uint8')

    #         ood_gts = mask
    #         ood_gts_list.append(np.expand_dims(ood_gts, 0))

    #         with torch.no_grad():
    #             image = preprocess_image(image, mean_std)
    #             main_out, anomaly_score = net(image)
    #             del main_out
    #         anomaly_score_list.append(anomaly_score.cpu().numpy())
                
    #     print()
    # ############################################################################
    # # If not evaluating on the FS LostAndFound, then evaluate on
    # # the dataset specified in args
    # else:
    from datasets.selfgen import std, mean
    from datasets.selfgen_labels import rbg2num as id_to_trainid

    mean_std = (mean, std)

    # ood_data_root = args.ood_dataset_path
    # image_root_path = os.path.join(ood_data_root, 'leftImg8bit_trainvaltest/leftImg8bit/val')
    # mask_root_path = os.path.join(ood_data_root, 'gtFine_trainvaltest/gtFine/val')

    # if not os.path.exists(image_root_path):
    #     raise ValueError(f"Dataset directory {image_root_path} doesn't exist!")

    if args.threshold:
        if args.th_type == 'absolute':
            iter_over(save=False, th=args.threshold)
            exit(0)
        

    ood_gts_list, anomaly_score_list = iter_over(save=False)
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    # drop void pixels
    ood_mask = (ood_gts == 0)
    ind_mask = (ood_gts != 0)

    ood_out = -1 * anomaly_scores[ood_mask]
    ind_out = -1 * anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    print('Measuring metrics...')

    fpr, tpr, ths = roc_curve(val_label, val_out)
    
    if args.threshold:
        if args.th_type == 'tpr':
            index = np.searchsorted(tpr, args.threshold)
            selected_th = -ths[index]
            print("Using th=", selected_th)
            iter_over(save=True, th=selected_th)
            exit(0)



    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(val_label, val_out)
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)
    print(f'AUROC score: {roc_auc}')
    print(f'AUPRC score: {prc_auc}')
    print(f'FPR@TPR95: {fpr}')

