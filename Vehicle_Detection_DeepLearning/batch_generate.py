import pandas as pd
import numpy as np
import cv2
import vis
import data
import anchor_generate
from bbox import bbox_overlaps
import bbox_encode
import bbox_decode
#import data_augment


RPN_NEGATIVE_OVERLAP = 0.3
RPN_POSITIVE_OVERLAP = 0.7
RPN_FG_FRACTION = 0.5
RPN_BATCHSIZE = 256
def get_image_name2(df,ind,size=(640,300),augmentation = False,trans_range = 20,scale_range=20):
    ### Get image by name
    
    file_name = df['File_Path'][ind]
    print(file_name)
    img = cv2.imread(file_name)
    img_size = np.shape(img)
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,size)
    #name_str = file_name.split('/')
    #name_str = name_str[-1]
    name_str = df['index'][ind]
    #print(name_str)
    #print(name_str)
    #print(file_name)
    bb_boxes = df[df['index'] == name_str].reset_index()
    img_size_post = np.shape(img)
    
    #if augmentation == True:
    #    img,bb_boxes = trans_image(img,bb_boxes,trans_range)
    #    img,bb_boxes = stretch_image(img,bb_boxes,scale_range)
    #    img = augment_brightness_camera_images(img)
        
    bb_boxes['x1'] = np.round(bb_boxes['x1']/img_size[1]*img_size_post[1])
    bb_boxes['x2'] = np.round(bb_boxes['x2']/img_size[1]*img_size_post[1])
    bb_boxes['y1'] = np.round(bb_boxes['y1']/img_size[0]*img_size_post[0])
    bb_boxes['y2'] = np.round(bb_boxes['y2']/img_size[0]*img_size_post[0])
    bb_boxes['Area'] = (bb_boxes['x2']- bb_boxes['x1'])*(bb_boxes['y2']- bb_boxes['y1']) 
    #bb_boxes = bb_boxes[bb_boxes['Area']>400]
        
    
    return name_str,img,bb_boxes

def _generate_all_bboxes(_num_anchors, _anchors, feat_h, feat_w, feat_stride = 16):
    shift_x = np.arange(0, feat_w) * feat_stride
    shift_y = np.arange(0, feat_h) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Create all bbox
    A = _num_anchors
    K = len(shifts)  # number of base points = feat_h * feat_w

    bbox = _anchors.reshape(1, A, 4) + shifts.reshape(K, 1, 4)
    bbox = bbox.reshape(K * A, 4)
    return bbox

def _generate_all_bbox_use_array_info(_anchors, rpn_bbox_pred):
    _, feat_h, feat_w = rpn_bbox_pred.shape
    return np.asarray(_generate_all_bbox(_anchors, feat_h, feat_w),
                      dtype=xp.float32)

def keep_inside(anchors, img_info):
    """Calc indicies of anchors which are inside of the image size.

    Calc indicies of anchors which are located completely inside of the image
    whose size is speficied by img_info ((height, width, scale)-shaped array).
    """
    inds_inside = np.where(
                           (anchors[:, 0] >= 0) &
                           (anchors[:, 1] >= 0) &
                           (anchors[:, 2] < img_info[1]) &  # width
                           (anchors[:, 3] < img_info[0])  # height
                          )[0]
    return inds_inside, anchors[inds_inside]

def _calc_overlaps(anchors, gt_boxes, inds_inside):
    overlaps = bbox_overlaps(
                            np.ascontiguousarray(anchors, dtype=np.float),
                            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    overlaps = np.asarray(overlaps)
    argmax_overlaps_inds = overlaps.argmax(axis=1)
    gt_argmax_overlaps_inds = overlaps.argmax(axis=0)
    max_overlaps = overlaps[
                            np.arange(len(inds_inside)), argmax_overlaps_inds]
    gt_max_overlaps = overlaps[
                                gt_argmax_overlaps_inds, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps_inds = np.where(overlaps == gt_max_overlaps)[0]

    return argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds

def _create_bbox_labels(inds_inside, anchors, gt_boxes):
    labels = np.ones((len(inds_inside),), dtype=np.int32) * -1
    argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = _calc_overlaps(anchors, gt_boxes, inds_inside)
    # assign bg labels first so that positive labels can clobber them
    labels[max_overlaps < RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps_inds] = 1

    # fg label: above threshold IOU
    labels[max_overlaps >= RPN_POSITIVE_OVERLAP] = 1

    # assign bg labels last so that negative labels can clobber positives
    labels[max_overlaps < RPN_NEGATIVE_OVERLAP] = 0

    # subsample positive labels if we have too many
    num_fg = int(RPN_FG_FRACTION * RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    
    if len(fg_inds) > num_fg:
        #fg_inds = cuda.to_cpu(fg_inds)
        fg_inds = np.asarray(fg_inds)
        disable_inds = np.random.choice(fg_inds, size=int(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1
        
    # subsample negative labels if we have too many
    num_bg = RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        #bg_inds = cuda.to_cpu(bg_inds)
        bg_inds = np.asarray(bg_inds)
        disable_inds = np.random.choice(bg_inds, size=int(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    labels = np.asarray(labels)

    return argmax_overlaps_inds, labels

def calc_rpn(img, bbox, feat_stride = 16, anchor_ratios=(0.5, 1, 2), anchor_scales=(8, 16, 32), feat_h = 40, feat_w = 60):
    _anchors = anchor_generate.generate_anchors(ratios=anchor_ratios, scales=anchor_scales)
    _num_anchors = len(_anchors)
    img_info = img.shape[0:2]
    all_bbox = np.asarray(_generate_all_bboxes(_num_anchors,_anchors, feat_h, feat_w))
    inds_inside, all_inside_bbox = keep_inside(all_bbox, img_info)
    gt_boxes = bbox
    argmax_overlaps_inds, bbox_labels = _create_bbox_labels(inds_inside, all_inside_bbox, gt_boxes)
    gt_boxes = gt_boxes[argmax_overlaps_inds]
    bbox_reg_targets = bbox_encode.bbox_transform(all_inside_bbox, gt_boxes)
    bbox_reg_targets = bbox_reg_targets.astype(np.float32)
    return bbox_labels, bbox_reg_targets

def generate_train_anchor_batch(data, batch_size = 1, img_rows = 640, img_cols = 960):
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    while 1:
        batch_bbox = []
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data)-2000)
            name_str,img,bb_boxes = get_image_name2(data,i_line,
                                                   size=(img_cols, img_rows),
                                                   augmentation=False,
                                                   trans_range=50,
                                                   scale_range=50
                                                  )
            for i in range(len(bb_boxes)):
                bb_box_i = [bb_boxes.iloc[i]['x1'],bb_boxes.iloc[i]['y1'],
                            bb_boxes.iloc[i]['x2'],bb_boxes.iloc[i]['y2']]
                batch_bbox.append(bb_box_i)
            batch_bbox = np.asarray(batch_bbox)
            y_rpn_cls, y_rpn_regr = calc_rpn(img, batch_bbox)
            batch_images[i_batch] = img
        yield np.copy(batch_images), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)]
        
#### Training generator, generates augmented images
def generate_train_batch(data,batch_size = 32):
    
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_masks = np.zeros((batch_size, img_rows, img_cols, 1))
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data)-2000)
            name_str,img,bb_boxes = data_augment.get_image_name(df_vehicles,i_line,
                                                   size=(img_cols, img_rows),
                                                  augmentation=True,
                                                   trans_range=50,
                                                   scale_range=50
                                                  )
            img_mask = get_mask_seg(img,bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] =img_mask
        yield batch_images, batch_masks
        
#### Testing generator, generates augmented images
def generate_test_batch(data,batch_size = 32):
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_masks = np.zeros((batch_size, img_rows, img_cols, 1))
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(2000)
            i_line = i_line+len(data)-2000
            name_str,img,bb_boxes = data_augment.get_image_name(df_vehicles,i_line,
                                                   size=(img_cols, img_rows),
                                                  augmentation=False,
                                                   trans_range=0,
                                                   scale_range=0
                                                  )
            img_mask = get_mask_seg(img,bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] =img_mask
        yield batch_images, batch_masks