import numpy as np
import tensorflow as tf
import keras.backend
import random
import bbox_encode, bbox_decode, batch_generate
"""
anchor generate;
shifts Generate;
filters;
nms;
bbox_overlaps;
etc.
"""
def anchor(base_size=16, ratios=None, scales=None):
    """
    Generates a regular grid of multi-aspect and multi-scale anchor boxes.
    """
    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([8, 16, 32])

    base_anchor = np.array([1, 1, base_size, base_size]) - 1

    ratio_anchors = _ratio_enum(base_anchor, ratios)

    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])

    return anchors

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                            y_ctr - 0.5 * (hs - 1),
                            x_ctr + 0.5 * (ws - 1),
                            y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def clip_cpu(boxes, shape):
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], int(shape[1] - 1)), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], int(shape[0] - 1)), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], int(shape[1] - 1)), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], int(shape[0] - 1)), 0)
    return boxes


def clip(boxes, shape):
    proposals = [
        keras.backend.maximum(keras.backend.minimum(boxes[:, 0::4], shape[1] - 1), 0),
        keras.backend.maximum(keras.backend.minimum(boxes[:, 1::4], shape[1] - 1), 0),
        keras.backend.maximum(keras.backend.minimum(boxes[:, 2::4], shape[1] - 1), 0),
        keras.backend.maximum(keras.backend.minimum(boxes[:, 3::4], shape[0] - 1), 0)
    ]

    return keras.backend.concatenate(proposals)

def union(au, bu):
	x = min(au[0], bu[0])
	y = min(au[1], bu[1])
	w = max(au[2], bu[2]) - x
	h = max(au[3], bu[3]) - y
	return x, y, w, h

def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0, 0, 0, 0
	return x, y, w, h

def iou(a, b):
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	i = intersection(a, b)
	u = union(a, b)

	area_i = i[2] * i[3]
	area_u = u[2] * u[3]
	return float(area_i) / float(area_u)

def calc_rpn(bbox, gta, width = 960, height = 640, resized_width = 960, resized_height = 640):
    downscale = float(16)
    anchor_sizes = [128, 256, 512]
    anchor_ratios = [[1, 1], [1, 2], [2, 1]]
    num_anchors = len(anchor_sizes) * len(anchor_ratios)
    # calculate the output map size based on the network architecture
    (output_width, output_height) = (60, 40)
    n_anchratios = len(anchor_ratios)
    # initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))
    num_bboxes = len(gta)
    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)
    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]
            for ix in range(output_width):
                # x-coordinates of the current anchor box
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2
                # ignore boxes that go across image boundaries
                if x1_anc < 0 or x2_anc > resized_width:
                    continue
                for jy in range(output_height):
                    # y-coordinates of the current anchor box
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2
                    # ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue
                    # bbox_type indicates whether an anchor should be a target
                    bbox_type = 'neg'
                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    best_iou_for_loc = 0.0
                    for bbox_num in range(num_bboxes):
                        # get IOU of the current GT box and the current anchor box
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
                        # calculate the regression targets if they will be needed
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > 0.7:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc)/2.0
                            cya = (y1_anc + y2_anc)/2.0
                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
                        #if bbox['label'][bbox_num] != 'bg':
                            # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                        if curr_iou > best_iou_for_bbox[bbox_num]:
                            best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                            best_iou_for_bbox[bbox_num] = curr_iou
                            best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
                            best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]
                        # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                        if curr_iou > 0.7:
                            bbox_type = 'pos'
                            num_anchors_for_bbox[bbox_num] += 1
                            # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                            if curr_iou > best_iou_for_loc:
                                best_iou_for_loc = curr_iou
                                best_regr = (tx, ty, tw, th)
                        # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                        if 0.3 < curr_iou < 0.7:
                            # gray zone between neg and pos
                            if bbox_type != 'pos':
                                bbox_type = 'neutral'
                    # turn on or off outputs depending on IOUs
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start+4] = best_regr
    # we ensure that every bbox has at least one positive RPN region
    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                best_anchor_for_bbox[idx,3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                best_anchor_for_bbox[idx,3]] = 1
            start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
            y_rpn_regr[best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]
	#y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = y_rpn_overlap.reshape((output_height, output_width, 9))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)
    #y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = y_is_box_valid.reshape((output_height, output_width, 9))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)
    #y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = y_rpn_regr.reshape((output_height, output_width, 36))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)
    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))
    num_pos = len(pos_locs[0])
    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256
    if len(pos_locs[0]) > num_regions/2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions/2
    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0
    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=3)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=3), y_rpn_regr], axis=3)
    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)

def shift(shape, stride):
    shift_x = np.arange(0, shape[0]) * stride
    shift_y = np.arange(0, shape[1]) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    anchors = anchor()

    # Create all bbox
    number_of_anchors = len(anchors)

    k = len(shifts)  # number of base points = feat_h * feat_w

    bbox = anchors.reshape(1, number_of_anchors, 4) + shifts.reshape(k, 1, 4)

    bbox = bbox.reshape(k * number_of_anchors, 4)

    return bbox

def cal_accuracy(gta, bbox, scores):
    bbox = bbox[0]

    (output_width, output_height) = (60, 40)
    num_anchors = 9
    #print('bbox',bbox)
    #print('gta',gta)
    #print('shape of bbox',bbox.shape)
    #anchor box generate(generate anchors in each shifts box)
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height * output_width * num_anchors , 4))

    anchor_box = batch_generate._generate_all_bbox(output_width, output_height)
    total_anchors = anchor_box.shape[0]
    #print('the shape of anchor_box', np.asarray(anchor_box).shape)
    #print('the total number os anchors',total_anchors)

    #Only inside anchors are valid
    _allowed_border = 0
    im_info = (640, 960)
    inds_inside = np.where(
    (anchor_box[:, 0] >= -_allowed_border) &
    (anchor_box[:, 1] >= -_allowed_border) &
    (anchor_box[:, 2] < im_info[1] + _allowed_border) &  # width
    (anchor_box[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]
    overlaps = bbox_overlaps(np.ascontiguousarray(bbox, dtype=np.float),np.ascontiguousarray(gta, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    #max_overlaps = np.zeros((output_height * output_width * num_anchors))
    max_overlaps = overlaps[np.arange(300), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    #print('max_overlaps',max_overlaps)
    #print('gt_argmax_overlaps',gt_argmax_overlaps)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    #print('overlaps display',overlaps)
    #print('shape of overlaps', np.asarray(overlaps).shape)
    #print('argmax_overlaps', argmax_overlaps)
    #print('shape of argmax_overlaps',argmax_overlaps.shape)
    #print('max overlaps display', max_overlaps)
    #print('total number of max overlaps', len(max_overlaps))
    #print('shape of max overlaps', max_overlaps.shape)
    #print('1 gt_max_overlaps display', gt_max_overlaps)
    #print('1 total number of gt_max_overlaps', len(gt_max_overlaps))
    #print('1 gt_argmax_overlaps', gt_argmax_overlaps)
    #print('1 number of gt_argmax_overlaps', len(gt_argmax_overlaps))
    y_rpn_overlap = y_rpn_overlap.reshape(output_height * output_width * num_anchors)
    y_is_box_valid = y_is_box_valid.reshape(output_height * output_width * num_anchors)

    #y_rpn_overlap[gt_argmax_overlaps] = 1
    #y_is_box_valid[gt_argmax_overlaps] = 1
    y_rpn_overlap[max_overlaps >= 0.7] = 1
    y_is_box_valid[max_overlaps >= 0.7] = 1

    valid_anchors = anchor_box[inds_inside, :]
    overlaps = bbox_overlaps(np.ascontiguousarray(valid_anchors, dtype=np.float),np.ascontiguousarray(gta, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    #max_overlaps = np.zeros((output_height * output_width * num_anchors))
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
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
    #print('2 gt_max_overlaps display', gt_max_overlaps)
    #print('2 total number of gt_max_overlaps', len(gt_max_overlaps))
    #print('2 gt_argmax_overlaps', gt_argmax_overlaps)
    #print('2 number of gt_argmax_overlaps', len(gt_argmax_overlaps))

    #postive = 0
    #print(overlaps.shape)
    #for i in range(300):
    #    index = np.where(overlaps[i,:] > 0.1)[0]
    #    postive += len(index)
        #print(index)
        #if(overlaps[i,:] > 0.7):
        #    postive +=1
    #print('fg sample number',postive)
    pos = np.where(np.logical_and(y_rpn_overlap == 1, y_is_box_valid == 1))[0]
    print('fg sample number', len(pos))
    print('groundtruth number', len(gta))
    print('calculate overlaps shape',overlaps.shape)
    #argmax_overlaps = overlaps.argmax(axis=1)
    #max_overlaps = overlaps[np.arange(300), argmax_overlaps]
    #print('max overlaps',max_overlaps)
    #gt_argmax_overlaps = overlaps.argmax(axis=0)
    #gt_max_overlaps = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]
    #gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    #print(gt_argmax_overlaps)

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.7, max_boxes=300):
    if len(boxes) == 0:
        return []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    #print('coordinates', x1, y1, x2, y2)
    #print('coordinates shape', x1.shape, y1.shape, x2.shape, y2.shape)
    #np.testing.assert_array_less(x1, x2)
    #np.testing.assert_array_less(y1, y2)

    #boxes = boxes.astype('float')
    pick = []
    #print('probs',probs)
    #print('shape of probs', probs.shape)
    probs = probs.reshape(-1)
    #print(probs.shape)
    idx = np.argsort(probs[:])
    #print('sorted index',idx)
    while(len(idx)> 0):
        last = len(idx) - 1
        i = idx[last]
        pick.append(i)
        # find the intersection
        xx1_int = np.maximum(x1[i], x1[idx[:last]])
        yy1_int = np.maximum(y1[i], y1[idx[:last]])
        xx2_int = np.minimum(x2[i], x2[idx[:last]])
        yy2_int = np.minimum(y2[i], y2[idx[:last]])

        # find the union
        xx1_un = np.minimum(x1[i], x1[idx[:last]])
        yy1_un = np.minimum(y1[i], y1[idx[:last]])
        xx2_un = np.maximum(x2[i], x2[idx[:last]])
        yy2_un = np.maximum(y2[i], y2[idx[:last]])

        # compute the width and height of the bounding box
        ww_int = xx2_int - xx1_int
        hh_int = yy2_int - yy1_int
        ww_un = xx2_un - xx1_un
        hh_un = yy2_un - yy1_un

        ww_un = np.maximum(0, ww_un)
        hh_un = np.maximum(0, hh_un)

        # compute the ratio of overlap
        overlap = (ww_int*hh_int)/(ww_un*hh_un + 1e-9)

        # delete all indexes from the index list that have
        idx = np.delete(idx, np.concatenate(([last],np.where(overlap > overlap_thresh)[0])))
        if len(pick) >= max_boxes:
            break
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    #print('bbox', boxes)
    #print('probs',probs)
    return boxes, probs

def propose_cpu(boxes, scores, maximum):
    shape = (40, 60)
    shifted = shift(shape, 16)
    proposals = np.reshape(boxes, (-1, 4))
    #print('before decode',proposals.shape)
    #print(proposals)
    proposals = bbox_decode.bbox_transform_inv_cpu(shifted, proposals)
    proposals = np.reshape(proposals, (-1, 4))
    #print('proposals', proposals)
    #print('after decode',proposals.shape)
    proposals = clip_cpu(proposals, shape)
    indicies = filter_boxes_cpu(proposals, 1)
    proposals = proposals[indicies]
    #print(proposals.shape)
    #print('input score shape',scores.shape)
    scores = scores[:,:,:,:9]

    scores = np.reshape(scores, (-1, 1))
    #print('reshape score',scores.shape)
    scores = scores[indicies]
    #print('valid score shape',scores.shape)
    #print('score', scores)
    #print('shape', scores.shape)
    #print(scores[0])
    idx = np.where(scores[:] > 0.5)[0]
    print('> 0.7', len(idx))
    #print('> 0.  scores',len(np.where(scores>0.7)[0]))
    #print('diplay', np.where(scores[:,0]>0.2)[0])
    boxes, scores = non_max_suppression_fast(proposals, scores[:])
    #print('after nms box shape',boxes.shape)
    #print('after nms score display',scores)
    boxes = np.expand_dims(boxes, axis = 0)
    scores = np.expand_dims(scores, axis = 0)
    return boxes, scores

def propose(boxes, scores, maximum):
    #shape = keras.backend.int_shape(boxes)[1:3]
    shape = (40,60)
    shifted = shift(shape, 16)

    proposals = keras.backend.reshape(boxes, (-1, 4))

    proposals = bbox_decode.bbox_transform_inv(shifted, proposals)

    proposals = clip(proposals, shape)

    indicies = filter_boxes(proposals, 1)

    proposals = keras.backend.gather(proposals, indicies)

    scores = scores[:, :, :, :9]
    scores = keras.backend.reshape(scores, (-1, 1))
    scores = keras.backend.gather(scores, indicies)
    scores = keras.backend.flatten(scores)

    proposals = keras.backend.cast(proposals, tf.float32)
    scores = keras.backend.cast(scores, tf.float32)

    indicies = non_maximum_suppression(proposals, scores, maximum, 0.7)

    proposals = keras.backend.gather(proposals, indicies)

    return keras.backend.expand_dims(proposals, 0)

def resize_images(images, shape):
    return tf.image.resize_images(images, shape)

#CPU version
#TODO, ADD
def filter_boxes_cpu(proposals, minimum):
    #ws = proposals[:, 2] - proposals[:, 0] + 1
    #hs = proposals[:, 3] - proposals[:, 1] + 1

    #indicies = np.where((ws >= minimum) & (hs >= minimum))

    #indicies = np.flatten(indicies)
    #print(proposals.shape)
    #proposals = proposals.reshape()
    ws = proposals[:, 2] - proposals[:, 0] + 1
    hs = proposals[:, 3] - proposals[:, 1] + 1
    keep = np.where((ws >= minimum) & (hs >= minimum))[0]
    return keep
    #return np.cast(indicies, np.int32)

#GPU version
def filter_boxes(proposals, minimum):
    ws = proposals[:, 2] - proposals[:, 0] + 1
    hs = proposals[:, 3] - proposals[:, 1] + 1

    indicies = tf.where((ws >= minimum) & (hs >= minimum))

    indicies = keras.backend.flatten(indicies)

    return keras.backend.cast(indicies, tf.int32)


def non_maximum_suppression(boxes, scores, maximum, threshold=0.5):
    return tf.image.non_max_suppression(
        boxes=boxes,
        iou_threshold=threshold,
        max_output_size=maximum,
        scores=scores
    )

def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = np.zeros((N, K), dtype=np.float)

    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1))

        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1)

            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1)

                if ih > 0:
                    ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)

                    overlaps[n, k] = iw * ih / ua

    return overlaps

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_encode.bbox_transform_cpu(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
