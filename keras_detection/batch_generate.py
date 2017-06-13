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

            x_img = img.astype(np.float32)
            x_img[:, :, 0] -= img_channel_mean[0]
            x_img[:, :, 1] -= img_channel_mean[1]
            x_img[:, :, 2] -= img_channel_mean[2]
            x_img /= img_scaling_factor
            x_img = np.expand_dims(x_img, axis=0)
            #label generate( regression target & postive/negative samples)
            y_rpn_cls,y_rpn_regr = label_generate(img, gta)
            #y_rpn_cls,y_rpn_regr = utils.calc_rpn(bb_boxes, gta)
            #print(y_rpn_cls)
            #print(y_rpn_regr.shape)
        yield x_img, [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], gta

#TODO
#fast version(inprogress)
def label_generate(img, gta):
    #inti base matrix
    (output_width, output_height) = (60, 40)
    num_anchors = 9

    #40,60,9
    #40,60,9,4
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height * output_width * num_anchors , 4))

    #anchor box generate(generate anchors in each shifts box)
    anchor_box = _generate_all_bbox(output_width, output_height)
    total_anchors = anchor_box.shape[0]
    #print('the shape of anchor_box', np.asarray(anchor_box).shape)
    #print('the total number os anchors',total_anchors)

    #Only inside anchors are valid
    _allowed_border = 0
    im_info = img.shape[:2]
    inds_inside = np.where(
    (anchor_box[:, 0] >= -_allowed_border) &
    (anchor_box[:, 1] >= -_allowed_border) &
    (anchor_box[:, 2] < im_info[1] + _allowed_border) &  # width
    (anchor_box[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]
    #print('inside anchor index',inds_inside)
    #print('number of valid anchors',len(inds_inside))

    valid_anchors = anchor_box[inds_inside, :]
    #print('valid_anchors display',valid_anchors)
    #print('shape of valid_anchors',np.asarray(valid_anchors).shape)

    y_rpn_regr[inds_inside] = anchor_box[inds_inside, :]
    #print('rpn overlap display', y_rpn_regr)
    #print('shape of rpn overlap',np.asarray(y_rpn_regr).shape)
    #print('rpn overlap[inds_inside] display', y_rpn_regr[inds_inside])
    #print('shape of inds_inside rpn overlaps',np.asarray(y_rpn_regr[inds_inside]).shape)

    #calculate iou(overlaps)
    #print('y_rpn_overlap')
    overlaps = utils.bbox_overlaps(np.ascontiguousarray(y_rpn_regr, dtype=np.float),np.ascontiguousarray(gta, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = np.zeros((output_height * output_width * num_anchors))
    max_overlaps[inds_inside] = overlaps[np.arange(len(inds_inside)), argmax_overlaps[inds_inside]]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    #print('overlaps display',overlaps)
    #print('shape of overlaps', np.asarray(overlaps).shape)
    #print('argmax_overlaps', argmax_overlaps)
    #print('shape of argmax_overlaps',argmax_overlaps.shape)
    #print('max overlaps display', max_overlaps)
    #print('total number of max overlaps', len(max_overlaps))
    #print('shape of max overlaps', max_overlaps.shape)
    #print('gt_max_overlaps display', gt_max_overlaps)
    #print('total number of gt_max_overlaps', len(gt_max_overlaps))
    #print('gt_argmax_overlaps', gt_argmax_overlaps)
    #print('number of gt_argmax_overlaps', len(gt_argmax_overlaps))

    #y_rpn_overlap, y_is_box_valid
    y_rpn_overlap = y_rpn_overlap.reshape(output_height * output_width * num_anchors)
    y_is_box_valid = y_is_box_valid.reshape(output_height * output_width * num_anchors)
    #negative
    #print('shape of y_rpn_overlap', y_rpn_overlap.shape)
    #print('shape of y_is_box_valid',y_is_box_valid.shape)
    y_rpn_overlap[max_overlaps < neg_min_overlaps] = 0
    y_is_box_valid[inds_inside] = 1
    #y_is_box_valid[max_overlaps < neg_min_overlaps] = 1#not good way to set all box as valid, because we also have outside box here

    #neutral
    #np.logical_and
    y_rpn_overlap[np.logical_and(neg_min_overlaps < max_overlaps, max_overlaps < pos_max_overlaps)] = 0
    y_is_box_valid[np.logical_and(neg_min_overlaps < max_overlaps, max_overlaps < pos_max_overlaps)] = 0
    #y_rpn_overlap[neg_min_overlaps < max_overlaps and max_overlaps < pos_max_overlaps] = 0
    #y_is_box_valid[neg_min_overlaps < max_overlaps and max_overlaps < pos_max_overlaps] = 0

    #positive
    y_rpn_overlap[gt_argmax_overlaps] = 1
    y_is_box_valid[gt_argmax_overlaps] = 1
    y_rpn_overlap[max_overlaps >= pos_max_overlaps] = 1
    y_is_box_valid[max_overlaps >= pos_max_overlaps] = 1



    # subsample positive labels if we have too many
    num_fg = int(fraction * batchsize)
    #print('balanced fg',num_fg)
    disable_inds = []
    fg_inds = np.where(np.logical_and(y_rpn_overlap == 1, y_is_box_valid == 1))[0]
    #print('fg number',len(fg_inds))
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        #labels[disable_inds] = -1
        y_is_box_valid[disable_inds] = 0
        y_rpn_overlap[disable_inds] = 0
    fg_inds = np.where(np.logical_and(y_rpn_overlap == 1, y_is_box_valid == 1))[0]
    # subsample negative labels if we have too many
    num_bg = batchsize - np.sum(np.logical_and(y_rpn_overlap == 1, y_is_box_valid == 1))
    bg_inds = np.where(np.logical_and(y_rpn_overlap == 0, y_is_box_valid == 1))[0]
    #print('bg number',len(bg_inds))
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        #labels[disable_inds] = -1
        y_is_box_valid[disable_inds] = 0
        y_rpn_overlap[disable_inds] = 0
    #print ("was %s inds, disabling %s, now %s %sinds" % (len(bg_inds), len(disable_inds), np.sum(np.logical_and(y_rpn_overlap == 1, y_is_box_valid == 0))))
    #print('negative samples',np.where(np.logical_and(y_rpn_overlap == 0, y_is_box_valid == 1))[0])
    #print('postive samples',np.where(np.logical_and(y_rpn_overlap == 1, y_is_box_valid == 1))[0])
    print('number of postive samples',len(np.where(np.logical_and(y_rpn_overlap == 1, y_is_box_valid == 1))[0]))
    print('number of negative samples',len(np.where(np.logical_and(y_rpn_overlap == 0, y_is_box_valid == 1))[0]))

    #bbox transfer for all valid postive samples
    y_rpn_regr[fg_inds] = utils._compute_targets(y_rpn_regr[fg_inds], gta[argmax_overlaps[fg_inds], :])
    #print('bbox targets shape', y_rpn_regr.shape)
    #print('bbox targets value', y_rpn_regr)
    #print('bbox targets[inds_inside]', y_rpn_regr[inds_inside])
    y_rpn_overlap = y_rpn_overlap.reshape(output_height, output_width, num_anchors)
    y_is_box_valid = y_is_box_valid.reshape(output_height, output_width, num_anchors)
    #print('y rpn overlaps',y_rpn_overlap)
    #print('y is valid',y_is_box_valid)
    #print('')

    y_rpn_regr = y_rpn_regr.reshape(output_height, output_width, num_anchors * 4)
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis = 0)
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis = 0)
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis = 0)

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=3)
    #print('shape of rpn cls',y_rpn_cls.shape)
    overlaps = np.repeat(y_rpn_overlap, 4, axis=3)
    #print('repeat', overlaps.shape)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=3), y_rpn_regr], axis=3)
    #print('shape is ',y_rpn_cls.shape, y_rpn_regr.shape, y_is_box_valid.shape)
    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


#data = pd.read_csv('voc.csv')
#data = data.drop('Unnamed: 0', 1)
#data['File_Path'] = './VOCdevkit2007/VOC2007/JPEGImages/' + data['Frame']
#print(data.head())
#batch_generate(data, batch_size=10)
