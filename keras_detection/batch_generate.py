import pandas as pd
import numpy as np
import cv2, utils
import random
import copy
import threading
import itertools
import numpy.random as npr
pos_max_overlaps = 0.7
neg_min_overlaps = 0.3
batchsize = 256
fraction = 0.5
RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
RPN_POSITIVE_WEIGHT = -1.0
img_channel_mean = [103.939, 116.779, 123.68]
img_scaling_factor = 1.0
def _generate_all_bbox(feat_h, feat_w, feat_stride = 16, num_anchors = 9):
    # Create lattice (base points to shift anchors)
    shift_x = np.arange(0, feat_w) * feat_stride
    shift_y = np.arange(0, feat_h) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),shift_x.ravel(), shift_y.ravel())).transpose()

    # Create all bbox
    A = num_anchors
    K = len(shifts)  # number of base points = feat_h * feat_w
    anchors = utils.anchor()
    bbox = anchors.reshape(1, A, 4) + shifts.reshape(K, 1, 4)
    bbox = bbox.reshape(K * A, 4)
    return bbox

def get_img_by_name(df,ind,size=(640,300)):
    file_name = df['File_Path'][ind]
    #print(file_name)
    img = cv2.imread(file_name)
    img_size = np.shape(img)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,size)
    name_str = file_name.split('/')
    name_str = name_str[-1]
    #print(name_str)
    #print(file_name)
    bb_boxes = df[df['Frame'] == name_str].reset_index()
    img_size_post = np.shape(img)
    #TODO,(add data augment support)

    bb_boxes['xmin'] = np.round(bb_boxes['xmin']/img_size[1]*img_size_post[1])
    bb_boxes['xmax'] = np.round(bb_boxes['xmax']/img_size[1]*img_size_post[1])
    bb_boxes['ymin'] = np.round(bb_boxes['ymin']/img_size[0]*img_size_post[0])
    bb_boxes['ymax'] = np.round(bb_boxes['ymax']/img_size[0]*img_size_post[0])
    bb_boxes['Area'] = (bb_boxes['xmax']- bb_boxes['xmin'])*(bb_boxes['ymax']- bb_boxes['ymin'])
    #bb_boxes = bb_boxes[bb_boxes['Area']>400]

    return name_str,img,bb_boxes

def batch_generate(data, batch_size = 1):
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            name_str, img, bb_boxes = get_img_by_name(data, i_line, size = (960, 640))
            #print(name_str)
            #TODO
            #create anchor based groundtruth here(target bbox(1,40,60,9) & target regression(1,40,60,36))
            gta = np.zeros((len(bb_boxes), 4))
            #print(gta.shape)
            #bbox groundtruth before bbox_encode
            for i in range(len(bb_boxes)):
                gta[i, 0] = int(bb_boxes.iloc[i]['xmin'])
                gta[i, 1] = int(bb_boxes.iloc[i]['ymin'])
                gta[i, 2] = int(bb_boxes.iloc[i]['xmax'])
                gta[i, 3] = int(bb_boxes.iloc[i]['ymax'])
            #print(gta)
            #anchor product here by using utils.ancho func
            anchor_box = _generate_all_bbox(40, 60)
            total_anchors = anchor_box.shape[0]
            #print(anchor_box.shape)
            #filter outside box
            _allowed_border = 0
            im_info = img.shape[:2]
            inds_inside = np.where(
            (anchor_box[:, 0] >= -_allowed_border) &
            (anchor_box[:, 1] >= -_allowed_border) &
            (anchor_box[:, 2] < im_info[1] + _allowed_border) &  # width
            (anchor_box[:, 3] < im_info[0] + _allowed_border)    # height
            )[0]
            #print(inds_inside.shape)
            anchors = anchor_box[inds_inside, :]
            #print(anchors.shape)
            labels = np.empty((len(inds_inside), ), dtype=np.float32)
            labels.fill(-1)
            #print(labels.shape)
            overlaps = utils.bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float),np.ascontiguousarray(gta, dtype=np.float))
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
            #print(gt_argmax_overlaps)
            labels[max_overlaps < neg_min_overlaps] = 0
            labels[gt_argmax_overlaps] = 1
            # fg label: above threshold IOU
            labels[max_overlaps >= pos_max_overlaps] = 1
            labels[max_overlaps < neg_min_overlaps] = 0

            # subsample positive labels if we have too many
            num_fg = int(fraction * batchsize)
            fg_inds = np.where(labels == 1)[0]
            if len(fg_inds) > num_fg:
                disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                labels[disable_inds] = -1

            # subsample negative labels if we have too many
            num_bg = batchsize - np.sum(labels == 1)
            bg_inds = np.where(labels == 0)[0]
            if len(bg_inds) > num_bg:
                disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                labels[disable_inds] = -1
            #print ("was %s inds, disabling %s, now %s inds" % (len(bg_inds), len(disable_inds), np.sum(labels == 0)))

            bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
            bbox_targets = utils._compute_targets(anchors, gta[argmax_overlaps, :])

            bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
            bbox_inside_weights[labels == 1, :] = np.array(RPN_BBOX_INSIDE_WEIGHTS)

            bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
            if RPN_POSITIVE_WEIGHT < 0:
                # uniform weighting of examples (given non-uniform sampling)
                num_examples = np.sum(labels >= 0)
                positive_weights = np.ones((1, 4)) * 1.0 / num_examples
                negative_weights = np.ones((1, 4)) * 1.0 / num_examples
            else:
                assert ((RPN_POSITIVE_WEIGHT > 0) &(RPN_POSITIVE_WEIGHT < 1))
                positive_weights = (RPN_POSITIVE_WEIGHT / np.sum(labels == 1))
                negative_weights = ((1.0 - RPN_POSITIVE_WEIGHT)/ np.sum(labels == 0))

            bbox_outside_weights[labels == 1, :] = positive_weights
            bbox_outside_weights[labels == 0, :] = negative_weights

            #back to original size(:,40,60,:)
            labels = utils._unmap(labels, total_anchors, inds_inside, fill=-1)
            bbox_targets = utils._unmap(bbox_targets, total_anchors, inds_inside, fill=0)
            bbox_inside_weights = utils._unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
            bbox_outside_weights = utils._unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
            #print ('rpn: max max_overlap', np.max(max_overlaps))
            #print ('rpn: num_positive', np.sum(labels == 1))
            #print ('rpn: num_negative', np.sum(labels == 0))
            # labels
            height, width = (40, 60)
            labels = labels.reshape((1, height, width, 9))
            #print(labels.shape)
            # bbox_targets
            bbox_targets = bbox_targets.reshape((1, height, width, 9 * 4))
            #print(bbox_targets.shape)
            x_img = img.astype(np.float32)
            x_img[:, :, 0] -= img_channel_mean[0]
            x_img[:, :, 1] -= img_channel_mean[1]
            x_img[:, :, 2] -= img_channel_mean[2]
            x_img /= img_scaling_factor
            x_img = np.expand_dims(x_img, axis=0)
        yield x_img, [np.copy(labels), np.copy(bbox_targets)]

#data = pd.read_csv('voc.csv')
#data = data.drop('Unnamed: 0', 1)
#data['File_Path'] = './VOCdevkit2007/VOC2007/JPEGImages/' + data['Frame']
#print(data.head())
#batch_generate(data, batch_size=10)
