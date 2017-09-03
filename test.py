import cv2
import cPickle
import numpy as np




# from utilities import visualize_gt_label
#
# project_dir = "/home/fregu856/2D_detection/"
# # project_dir = "/root/2D_detection"
#
# train_label_paths = cPickle.load(open(project_dir + "data/train_label_paths.pkl"))
# train_img_paths = cPickle.load(open(project_dir + "data/train_img_paths.pkl"))
#
# for img_path, label_path in zip(train_img_paths, train_label_paths):
#     img_with_bboxes = visualize_gt_label(img_path, label_path)
#     cv2.imwrite(img_path.split(".png")[0] + "_gt.png", img_with_bboxes)








from utilities import batch_IOU, sparse_to_dense

img_height = 375
img_width = 1242
no_of_classes = 3

project_dir = "/home/fregu856/2D_detection/"
#project_dir = "/root/2D_segmentation/"

data_dir = project_dir + "data/"

def set_anchors():
    # TODO! understand this code!

    H, W, B = 24, 78, 9

    anchor_shapes = np.reshape(
        [np.array(
            [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
            [ 162.,  87.], [  38.,  90.], [ 258., 173.],
            [ 224., 108.], [  78., 170.], [  72.,  43.]])]*H*W,
        (H, W, B, 2)
    )

    center_x = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, W+1)*float(img_width)/(W+1)]*H*B),
                (B, H, W)
            ),
            (1, 2, 0)
        ),
        (H, W, B, 1)
    )

    center_y = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, H+1)*float(img_height)/(H+1)]*W*B),
                (B, W, H)
            ),
            (2, 1, 0)
        ),
        (H, W, B, 1)
    )

    anchors = np.reshape(
        np.concatenate((center_x, center_y, anchor_shapes), axis=3),
        (-1, 4)
    )

    return anchors

anchor_boxes = set_anchors() # (anchor_boxes has shape [anchors_per_img, 4])
anchors_per_img = len(anchor_boxes)

# (inspired by read_batch in the official implementation)
def read_batch():
    train_bboxes_per_img = cPickle.load(open(data_dir + "train_bboxes_per_img.pkl"))
    train_img_paths = cPickle.load(open(data_dir + "train_img_paths.pkl"))

    class_to_label = {"car": 0, "pedestrian": 1, "cyclist": 2}

    images = [] # (list of length batch_size, each element is an image of shape [img_height, img_width, 3])
    class_labels_per_img = [] # (list of length batch_size, each element is a list of length no_of_gt_bboxes_in_img containing the classes (0=car, 1=pedestrian etc.) of the ground truth bboxes in the image)
    gt_bboxes_per_img  = [] # (list of length batch_size, each element is a 2D array of shape [no_of_gt_bboxes_in_img, 4], where each row is [center_x, center_y, w, h] of each ground truth bbox in the image)
    gt_deltas_per_img = [] # (list of length batch_size, each element is a list of length no_of_gt_bboxes_in_img, where each element in turn is a list [delta_x, delta_y, delta_w, delta_h] which describes how to transform the assigned anchor bbox into the ground truth bbox for each ground truth bbox in the image)
    anchor_indices_per_img  = [] # (list of length batch_size, each element is a list of length no_of_gt_bboxes_in_img containing the index of the assigned anchor bbox for each ground truth bbox in the image)

    batch_size = 4

    for i in range(batch_size):
        img_path = train_img_paths[i]
        img_bboxes = train_bboxes_per_img[i] # (bbox format: [center_x, center_y, w, h, class_label])

        # load the image:
        img = cv2.imread(img_path, -1).astype(np.float32)
        images.append(img)

        # load annotations:
        class_labels_per_img.append([class_to_label[b[4]] for b in img_bboxes])
        img_gt_bboxes = np.array([[b[0], b[1], b[2], b[3]] for b in img_bboxes]) # (bbox format: [center_x, center_y, w, h])
        # (img_gt_bboxes has shape [no_of_gt_bboxes_in_img, 4])
        gt_bboxes_per_img.append(img_gt_bboxes)

        img_gt_deltas = []
        img_anchor_indices = []
        assigned_anchor_indices = []
        for gt_bbox in img_gt_bboxes:
            IOUs = batch_IOU(anchor_boxes, gt_bbox)
            # (IOUs has shape [anchors_per_img, ])

            anchor_idx = -1
            sorted_anchor_indices_IOU = np.argsort(IOUs)[::-1] # (first element is the index of the anchor with the LARGEST IOU with the gt bbox etc.)
            for idx in sorted_anchor_indices_IOU:
                if IOUs[idx] <= 0:
                    break
                if idx not in assigned_anchor_indices:
                    assigned_anchor_indices.append(idx)
                    anchor_idx = idx
                    break
            if anchor_idx == -1: # (if all available IOUs equal 0:)
                # choose the available anchor which is closest to the gt bbox w.r.t L2 norm:
                norms = np.sum(np.square(gt_bbox - anchor_boxes), axis=1)
                sorted_anchor_indices_norm = np.argsort(norms)
                for idx in sorted_anchor_indices_norm:
                    if idx not in assigned_anchor_indices:
                        assigned_anchor_indices.append(idx)
                        anchor_idx = idx
                        break
            img_anchor_indices.append(anchor_idx)

            assigned_anchor_bbox = anchor_boxes[anchor_idx]
            anchor_cx, anchor_cy, anchor_w, anchor_h = assigned_anchor_bbox

            gt_cx, gt_cy, gt_w, gt_h = gt_bbox

            gt_delta = [0]*4
            gt_delta[0] = (gt_cx - anchor_cx)/anchor_w
            gt_delta[1] = (gt_cy - anchor_cy)/anchor_h
            gt_delta[2] = np.log(gt_w/anchor_w)
            gt_delta[3] = np.log(gt_h/anchor_h)

            img_gt_deltas.append(gt_delta)

        gt_deltas_per_img.append(img_gt_deltas)
        anchor_indices_per_img.append(img_anchor_indices)

    return images, class_labels_per_img, gt_deltas_per_img, anchor_indices_per_img, gt_bboxes_per_img

# (inspired by load_batch in the official implementation)
def load_data():
    # read batch input
    images, class_labels_per_img, gt_deltas_per_img, anchor_indices_per_img, gt_bboxes_per_img = read_batch()

    batch_size = len(class_labels_per_img)

    mask_indices, class_label_indices, gt_bbox_indices, gt_delta_values, gt_bbox_values  = [], [], [], [], []
    for i in range(batch_size):
        no_of_gt_bboxes_in_img = len(class_labels_per_img[i])

        img_class_labels = class_labels_per_img[i]
        img_anchor_indices = anchor_indices_per_img[i]
        img_gt_deltas = gt_deltas_per_img[i]
        img_gt_bboxes = gt_bboxes_per_img[i]

        for j in range(no_of_gt_bboxes_in_img):
            class_label = img_class_labels[j]
            anchor_idx = img_anchor_indices[j]
            gt_delta = img_gt_deltas[j]
            gt_bbox = img_gt_bboxes[j]

            class_label_indices.append([i, anchor_idx, class_label])
            mask_indices.append([i, anchor_idx])
            gt_bbox_indices.extend([[i, anchor_idx, k] for k in range(4)])
            gt_delta_values.extend(gt_delta)
            gt_bbox_values.extend(gt_bbox)

    #print class_label_indices
    #print mask_indices
    # print gt_bbox_indices
    # print gt_delta_values
    # print gt_bbox_values

    image_input = images,

    input_mask = sparse_to_dense(mask_indices, [batch_size, anchors_per_img],
                [1.0]*len(mask_indices))
    input_mask = np.reshape(input_mask, [batch_size, anchors_per_img, 1])

    box_delta_input = sparse_to_dense(gt_bbox_indices,
                [batch_size, anchors_per_img, 4], gt_delta_values)

    box_input = sparse_to_dense(gt_bbox_indices,
                [batch_size, anchors_per_img, 4], gt_bbox_values)

    labels = sparse_to_dense(class_label_indices,
                [batch_size, anchors_per_img, no_of_classes],
                [1.0]*len(class_label_indices))

    #print input_mask
    #print box_delta_input
    #print box_input
    #print labels

load_data()
