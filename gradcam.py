# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_gradcam

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

from model.utils.parser_func import parse_args, set_dataset_args

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)
  args = set_dataset_args(args, test=True)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic, gradcam_on=True)  # NOTE
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic, gradcam_on=True)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic, gradcam_on=True)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic, gradcam_on=True)
  else:
    print("network is not defined")
    # pdb.set_trace()

  fasterRCNN.create_architecture()
  print(fasterRCNN)

  print("load checkpoint %s" % (args.load_name))
  checkpoint = torch.load(args.load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05
    vis_dir = os.path.join(args.load_name.replace('models', 'visual'), 'gradcam')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
  else:
    thresh = 0.0

  save_name =  args.load_name.split('/')[-1]
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize=False, path_return=vis)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,  # NOTE: assert batch_size=1
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()  # NOTE: requires grads of features for visualization, not grads of params
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

  for i in range(num_images):
      # NOTE: clear activations and gradients for each test image
      target_activations = []
      cls_gradients, loc_gradients = [], []

      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      det_tic = time.time()
      rois, cls_prob_origin, bbox_pred_origin, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      # # NOTE: this is before nms applied
      # if fasterRCNN.gradcam_on:
      #   # cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
      #   # bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
      #   # batch_size, num_bbox_pred = cls_prob.size(0), cls_prob.size(1)
      #   num_bbox_pred, cls_dim = cls_prob.size(1), cls_prob.size(2)  # batch_size=1
      #   target_activations.append(fasterRCNN.target_activations.pop())
      #   for idx in range(min(num_bbox_pred, 10)):  # cuda out of memory
      #       print('processing id_{}'.format(idx))
      #       # backward for classification head
      #       # cls = cls_prob[:, idx, :]  # NOTE: replace slice by mask multiplication
      #       cls_index = np.argmax(cls_prob[:, idx, :].cpu().data.numpy())  # load gt cls_index
      #       cls_one_hot = np.zeros((1, num_bbox_pred, cls_dim), dtype=np.float32)
      #       cls_one_hot[0][idx][cls_index] = 1
      #       cls_one_hot = torch.from_numpy(cls_one_hot).requires_grad_(True)
      #       cls_one_hot = torch.sum(cls_one_hot.cuda() * cls_prob)
      #
      #       fasterRCNN.zero_grad()
      #       cls_one_hot.backward(retain_graph=True)
      #       # target_activations.append(fasterRCNN.target_activations.pop())
      #       cls_gradients.append(fasterRCNN.gradients.pop())
      #
      #       # backward for localization head
      #       # bbox = bbox_pred[:, idx, :]
      #       assert args.class_agnostic is False
      #       # loc_boxes = bbox_pred[:, idx, 4*(cls_index-1): 4*cls_index]
      #       loc_one_hot = np.zeros((1, num_bbox_pred, 4*cls_dim), dtype=np.float32)
      #       loc_one_hot[0][idx][4*(cls_index-1): 4*cls_index] = 1
      #       loc_one_hot = torch.from_numpy(loc_one_hot).requires_grad_(True)
      #       loc_one_hot = torch.sum(loc_one_hot.cuda() * bbox_pred)
      #
      #       fasterRCNN.zero_grad()
      #       loc_one_hot.backward(retain_graph=True)
      #       loc_gradients.append(fasterRCNN.gradients.pop())
      #       # target_activations, gradients = fasterRCNN.target_activations.pop(), fasterRCNN.gradients.pop()

      scores = cls_prob_origin.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred_origin.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
          pred_boxes = _.cuda() if args.cuda > 0 else _

      pred_boxes /= data[1][0][2].item()

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)
      if fasterRCNN.gradcam_on:
          assert len(fasterRCNN.target_activations) == 1
          target_activations.append(fasterRCNN.target_activations.pop())  # NOTE: only once
      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:, j]>thresh).view(-1)  # NOTE: bigger than thresh is good enough, do not have to be bigger than all other classes
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]  # use this final result for gradcam visualization

            # NOTE: this is after nms applied
            if fasterRCNN.gradcam_on:
              cls_prob = cls_prob_origin.squeeze(0)  # NOTE: slicing operated for two times
              assert len(cls_prob.shape) == 2, "shape: {}".format(cls_prob)
              cls_prob = cls_prob[:, j][inds]
              cls_prob = cls_prob[order]
              cls_prob = cls_prob[keep.view(-1).long()]

              assert args.class_agnostic is False
              bbox_pred = bbox_pred_origin.squeeze(0)
              assert len(bbox_pred.shape) == 2
              bbox_pred = bbox_pred[inds][:, j * 4:(j + 1) * 4]
              bbox_pred = bbox_pred[order]
              bbox_pred = bbox_pred[keep.view(-1).long()]

              num_bbox_pred_current_cls = cls_prob.size(0)
              for idx in range(num_bbox_pred_current_cls):
                  # print('processing id_{}'.format(idx))
                  # backward for classification head
                  # cls_index = np.argmax(cls_prob[idx, :].cpu().data.numpy())  # todo: load gt cls_index
                  cls_one_hot = np.zeros((num_bbox_pred_current_cls), dtype=np.float32)
                  cls_one_hot[idx] = 1
                  cls_one_hot = torch.from_numpy(cls_one_hot).requires_grad_(True)
                  cls_one_hot = torch.sum(cls_one_hot.cuda() * cls_prob)

                  fasterRCNN.zero_grad()
                  assert len(fasterRCNN.gradients) == 0
                  cls_one_hot.backward(retain_graph=True)
                  # target_activations.append(fasterRCNN.target_activations.pop())
                  cls_gradients.append(fasterRCNN.gradients.pop())
                  assert len(fasterRCNN.gradients) == 0

                  # backward for localization head
                  # bbox = bbox_pred[:, idx, :]
                  # loc_boxes = bbox_pred[:, idx, 4*(cls_index-1): 4*cls_index]
                  assert args.class_agnostic is False
                  loc_one_hot = np.zeros((num_bbox_pred_current_cls, 4), dtype=np.float32)
                  loc_one_hot[idx][:] = 1
                  loc_one_hot = torch.from_numpy(loc_one_hot).requires_grad_(True)
                  loc_one_hot = torch.sum(loc_one_hot.cuda() * bbox_pred)

                  fasterRCNN.zero_grad()
                  assert len(fasterRCNN.gradients) == 0
                  loc_one_hot.backward(retain_graph=True)
                  loc_gradients.append(fasterRCNN.gradients.pop())
                  assert len(fasterRCNN.gradients) == 0

                  im2show_gradcam = np.copy(im)

                  # print('target_activations: ', len(target_activations), target_activations[0].shape)
                  # print('cls_gradients: ', len(cls_gradients), cls_gradients[0].shape)
                  # print('loc_gradients: ', len(loc_gradients), cls_gradients[0].shape)
                  # print("cls_dets: ", cls_dets.shape)
                  # print("cls_prob: ", cls_prob.shape)
                  # print("bbox_pred: ", bbox_pred.shape)
                  # print('im2show_gradcam: ', im2show_gradcam.shape)
                  # # NOTE: check only one roi was utilized for backward
                  cls_gradients_tmp = torch.cat(cls_gradients, dim=0).squeeze()
                  loc_gradients_tmp = torch.cat(loc_gradients, dim=0).squeeze()
                  cls_gradients_tmp = torch.sum(cls_gradients_tmp, dim=3)
                  cls_gradients_tmp = torch.sum(cls_gradients_tmp, dim=2)
                  cls_gradients_tmp = torch.sum(cls_gradients_tmp, dim=1)
                  loc_gradients_tmp = torch.sum(loc_gradients_tmp, dim=3)
                  loc_gradients_tmp = torch.sum(loc_gradients_tmp, dim=2)
                  loc_gradients_tmp = torch.sum(loc_gradients_tmp, dim=1)
                  # print('cls_gradients_tmp: ', cls_gradients_tmp.shape)
                  # print('cls_gradients_tmp: ', torch.nonzero(cls_gradients_tmp))
                  # print('loc_gradients_tmp: ', loc_gradients_tmp.shape)
                  # print('loc_gradients_tmp: ', torch.nonzero(loc_gradients_tmp))
                  assert torch.nonzero(loc_gradients_tmp) == torch.nonzero(cls_gradients_tmp)
                  # print('pred_boxes: ', pred_boxes.shape)
                  # print('scores: ', scores.shape)
                  input_path = imdb.image_path_at(i)
                  img_name, ext = os.path.basename(input_path).split(".")
                  if cls_dets[idx, 4] > 0.3:
                      vis_path = os.path.join(vis_dir, img_name + '_{}_cls.' + ext)
                      im2show_cls = vis_gradcam(im2show_gradcam, target_activations[0], cls_gradients.pop(), cls_dets[idx, :], imdb.classes[j])
                      cv2.imwrite(vis_path.format(idx), im2show_cls)
                  else:
                      cls_gradients.pop()

                  # for idx in range(len(target_activations)):
                  if cls_dets[idx, 4] > 0.3:
                      vis_path = os.path.join(vis_dir, img_name + '_{}_loc.' + ext)
                      im2show_loc = vis_gradcam(im2show_gradcam, target_activations[0], loc_gradients.pop(), cls_dets[idx, :], imdb.classes[j])
                      cv2.imwrite(vis_path.format(idx), im2show_loc)
                  else:
                      loc_gradients.pop()

            # # do same operation as the cls_dets
            # cls_gradients = torch.cat(cls_gradients, dim=0)
            # cls_gradients = cls_gradients[inds]
            # cls_gradients = cls_gradients[order]
            # cls_gradients = cls_gradients[keep.view(-1).long()]

            # print("cls_dets: ", cls_dets.shape)
            # print("cls_gradients: ", torch.cat(cls_gradients, dim=0).shape)
            # print("cls_gradients: ", torch.cat(loc_gradients, dim=0).shape)
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

      if vis:
          input_path = imdb.image_path_at(i)
          vis_path = os.path.join(vis_dir, os.path.basename(input_path))
          cv2.imwrite(vis_path, im2show)
          # pdb.set_trace()
          #cv2.imshow('test', im2show)
          #cv2.waitKey(0)

  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
