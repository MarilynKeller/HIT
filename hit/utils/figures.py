import numpy as np

tissue_palette = [
[1.0, 1.0, 1.0, 0.0], #[0.0, 0.0, 0.0],
[0.803921568627451, 0.3607843137254902, 0.3607843137254902, 1.0],
[0.8549019607843137, 0.6470588235294118, 0.12549019607843137, 1.0],
[0.8784313725490196, 0.8901960784313725, 0.8431372549019608, 1.0],
]

def stack_colored(im_pred, im_gt, colors=None):
    """

    :param im_pred WxH or WxHx3
    :param s2: mask (GT) orange
    :param colors:
    :return:
    """
    if colors is not None:
        assert len(colors) == 3
        pred_col, gt_col, intersect_col = colors
    else:
        pred_col = [0.5,0.9,0.3] #light green
        gt_col = [0.9,0.5,0] #orange
        intersect_col = [1,1,1]

    if len(im_pred.shape) == 3:
        im_pred = np.sum(im_pred, axis=2)
    if len(im_gt.shape) == 3:
        im_gt = np.sum(im_gt, axis=2)

    im = np.zeros(im_pred.shape + (3,))
    for ci in range(3):

        im[im_pred>0, ci] = pred_col[ci]
        im[im_gt>0, ci] = gt_col[ci]

        i_mask = np.logical_and(im_pred>0, im_gt>0)
        im[i_mask, ci] = intersect_col[ci]
    return im

def trim_up_down(im, pad=0):
    """Given a numpy 2d or 3d images, trim the upper and lower part of the image of rows with all zeros"""
    if len(im.shape) == 3:
        imsum = np.sum(im, axis=2)
    else:
        imsum = im
    row_sum = np.sum(imsum, axis=1)
    start_idx = np.argmax(row_sum>0)
    row_sum = np.flipud(row_sum)
    end_idx = im.shape[0] - np.argmax(row_sum>0)
    
    # pad
    start_idx = max(0, start_idx-pad)
    end_idx = min(im.shape[0], end_idx+pad)
    
    im = im[start_idx:end_idx, :]
    return im