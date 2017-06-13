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
    #boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], int(shape[1] - 1)), 0)
    # y1 >= 0
    #boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], int(shape[0] - 1)), 0)
    # x2 < im_shape[1]
    #boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], int(shape[1] - 1)), 0)
    # y2 < im_shape[0]
    #oxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], int(shape[0] - 1)), 0)
    boxes[:,0]
    boxes[:,1]
    boxes[:,2]
    boxes[:,3]
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

    overlaps = bbox_overlaps(np.ascontiguousarray(bbox, dtype=np.float),np.ascontiguousarray(gta, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    pos = np.where(argmax_overlaps > 0.6)[0]
    print('after nms we have postive samples', len(pos))
    #max_overlaps = np.zeros((output_height * output_width * num_anchors))
    max_overlaps = overlaps[np.arange(len(bbox)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    #print('max_overlaps',max_overlaps)
    #print('gt_argmax_overlaps',gt_argmax_overlaps)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

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

    (output_width, output_height) = (60, 40)
    num_anchors = 9
    _allowed_border = 0
    im_info = (640,960)
    anchor_box = batch_generate._generate_all_bbox(output_width, output_height)
    total_anchors = anchor_box.shape[0]
    inds_inside = np.where(
    (anchor_box[:, 0] >= -_allowed_border) &
    (anchor_box[:, 1] >= -_allowed_border) &
    (anchor_box[:, 2] < im_info[1] + _allowed_border) &  # width
    (anchor_box[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]
    #shape = (40, 60)
    #shifted = shift(shape, 16)
    proposals = np.reshape(boxes, (-1, 4))
    #print('before decode',proposals.shape)
    #print(proposals)
    proposals[inds_inside] = bbox_decode.bbox_transform_inv_cpu(anchor_box[inds_inside], proposals[inds_inside])
    proposals = np.reshape(proposals, (-1, 4))
    #print(proposals)
    #print('proposals', proposals)
    #print('after decode',proposals.shape)
    #proposals = clip_cpu(proposals, shape)
    #indicies = filter_boxes_cpu(proposals, 1)
    #proposals = proposals[indicies]
    #print(proposals.shape)
    #print('input score shape',scores.shape)
    scores = scores[:,:,:,:9]

    scores = np.reshape(scores, (-1, 1))
    #print('reshape score',scores.shape)
    #scores = scores[indicies]
    #print('valid score shape',scores.shape)
    #print('score', scores)
    #print('shape', scores.shape)
    #print(scores[0])
    #idx = np.where(scores[:] > 0.5)[0]
    #print('> 0.7', len(idx))
    #print('> 0.  scores',len(np.where(scores>0.7)[0]))
    #print('diplay', np.where(scores[:,0]>0.2)[0])
    boxes, scores = non_max_suppression_fast(proposals[inds_inside], scores[inds_inside,:])
    print('> 0.7  scores',len(np.where(scores>0.7)[0]))
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
