import warnings
import numpy as np

import cv2

import torch

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


# Reference: https://bitbucket.org/william_rusnack/minimumboundingbox/src/master/
# This helps to generate the rotated rectangle with minimum area that covers the
# quadrangle bbox ground. It uses convex hull under the hoods to solve this problem.
# The entire concept is well explained here: https://stackoverflow.com/q/13542855/5353128
def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an n * 2 matrix of coordinates
    :rval: an n * 2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles=0
    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    best_idx= np.argmin(angles)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def aihub_collate(batch):
    """
    Collate function for ICDAR dataset. It receives a batch of ground truths
    and formats it in required format.
    """
    image_paths, img, boxes, transcripts, score_map, geo_map, training_mask = zip(*batch)
    batch_size = len(score_map)
    images, score_maps, geo_maps, training_masks = [], [], [], [] 

    # convert all numpy arrays to tensors
    for idx in range(batch_size):
        if img[idx] is not None:
            images.append(torch.from_numpy(img[idx]))
            score_maps.append(torch.from_numpy(score_map[idx]))
            geo_maps.append(torch.from_numpy(geo_map[idx]))
            training_masks.append(torch.from_numpy(training_mask[idx]))

    images = torch.stack(images, 0)
    score_maps = torch.stack(score_maps, 0)
    geo_maps = torch.stack(geo_maps, 0)
    training_masks = torch.stack(training_masks, 0)

    texts, bboxs, mapping = [], [], []
    for idx, (text, bbox) in enumerate(zip(transcripts, boxes)):
        # for txt, box in zip(text, bbox):/
            mapping.append(idx)
            texts.append(text)
            bboxs.append(bbox)

    # mapping = np.array(mapping)
    # texts = np.array(texts)
    # bboxs = np.stack(bboxs, axis=0)
    # bboxs = np.concatenate([bboxs, np.ones((len(bboxs), 1))], axis = 1).astype(np.float32)

    return image_paths, images, bboxs, training_masks, texts, score_maps, geo_maps, mapping


def icdar_collate(batch):
    """
    Collate function for ICDAR dataset. It receives a batch of ground truths
    and formats it in required format.
    """
    image_paths, img, boxes, transcripts, score_map, geo_map, training_mask = zip(*batch)
    batch_size = len(score_map)
    images, score_maps, geo_maps, training_masks = [], [], [], [] 

    # convert all numpy arrays to tensors
    for idx in range(batch_size):
        if img[idx] is not None:
            images.append(torch.from_numpy(img[idx]))
            score_maps.append(torch.from_numpy(score_map[idx]))
            geo_maps.append(torch.from_numpy(geo_map[idx]))
            training_masks.append(torch.from_numpy(training_mask[idx]))

    images = torch.stack(images, 0)
    score_maps = torch.stack(score_maps, 0)
    geo_maps = torch.stack(geo_maps, 0)
    training_masks = torch.stack(training_masks, 0)

    texts, bboxs, mapping = [], [], []
    for idx, (text, bbox) in enumerate(zip(transcripts, boxes)):
        for txt, box in zip(text, bbox):
            mapping.append(idx)
            texts.append(txt)
            bboxs.append(box)

    mapping = np.array(mapping)
    texts = np.array(texts)
    bboxs = np.stack(bboxs, axis=0)
    bboxs = np.concatenate([bboxs, np.ones((len(bboxs), 1))], axis = 1).astype(np.float32)

    return image_paths, images, bboxs, training_masks, texts, score_maps, geo_maps, mapping

def synth800k_collate(batch):
    """
    Collate function for ICDAR dataset. It receives a batch of ground truths
    and formats it in required format.
    """
    image_paths, img, bboxes, training_mask, transcripts, score_map, geo_map = zip(*batch)
    batch_size = len(score_map)
    images, score_maps, geo_maps, training_masks = [], [], [], []

    # convert all numpy arrays to tensors
    for idx in range(batch_size):
        if img[idx] is not None:
            images.append(torch.from_numpy(img[idx]).permute(2, 0, 1))
            score_maps.append(torch.from_numpy(score_map[idx]).permute(2, 0, 1))
            geo_maps.append(torch.from_numpy(geo_map[idx]).permute(2, 0, 1))
            training_masks.append(torch.from_numpy(training_mask[idx]).permute(2, 0, 1))

    images = torch.stack(images, 0)
    score_maps = torch.stack(score_maps, 0)
    geo_maps = torch.stack(geo_maps, 0)
    training_masks = torch.stack(training_masks, 0)

    return image_paths, images, bboxes, transcripts, score_maps, geo_maps, training_masks


def l2_norm(p1, p2=np.array([0, 0])):
    """
    Calculates the L2 norm (euclidean distance) between given two points.

    point (pi) should be in format (x, y)
    """
    return np.linalg.norm(p1 - p2)


def shrink_bbox(bbox, reference_lens, shrink_ratio=0.3):
    """
    Shrink the given bbox by given ratio.
    
    It first shrinks the two longer edges of a quadrangle, and then the
    two shorter ones. For each pair of two opposing edges, it determines
    the ???longer??? pair by comparing the mean of their lengths.

    For each edge (pi, p(i mod 4)+1),
    it shrinks it by moving its two endpoints inward along the
    edge by shrink_ratio*reference_lens[i] and 
    shrink_ratio*reference_lens[(i mod 4)+1] respectively.

    bbox shape: (4, 2) (clock wise from top left).
    """

    reference_lens = shrink_ratio * reference_lens

    # First find the "longer" edge pair
    if (
        # top horizontal edge + bottom horizontal edge
        l2_norm(bbox[0] - bbox[1]) + l2_norm(bbox[2] - bbox[3]) >
        # left vertical edge + right vertical edge
        l2_norm(bbox[0] - bbox[3]) + l2_norm(bbox[1] - bbox[2])
    ):
        # This means pair of horizontal edge is "longer" than pair
        # of vertical edges. So first shrink (p0, p1) and (p2, p3)
        # then shrink (p1, p2) and (p3, p0)

        # angle of edge between p0 and p1. Which is tan-1((y2-y1)/(x2-x1))
        theta = np.arctan2((bbox[1][1] - bbox[0][1]), (bbox[1][0] - bbox[0][0]))
        bbox[0][0] += reference_lens[0] * np.cos(theta)
        bbox[0][1] += reference_lens[0] * np.sin(theta)
        bbox[1][0] -= reference_lens[1] * np.cos(theta)
        bbox[1][1] -= reference_lens[1] * np.sin(theta)

        # shrink p2 and p3
        theta = np.arctan2((bbox[2][1] - bbox[3][1]), (bbox[2][0] - bbox[3][0]))
        bbox[2][0] -= reference_lens[2] * np.cos(theta)
        bbox[2][1] -= reference_lens[2] * np.sin(theta)
        bbox[3][0] += reference_lens[3] * np.cos(theta)
        bbox[3][1] += reference_lens[3] * np.sin(theta)

        # Then shrink p0 and p3
        theta = np.arctan2((bbox[3][0] - bbox[0][0]), (bbox[3][1] - bbox[0][1]))
        bbox[0][0] += reference_lens[0] * np.sin(theta)
        bbox[0][1] += reference_lens[0] * np.cos(theta)
        bbox[3][0] -= reference_lens[3] * np.sin(theta)
        bbox[3][1] -= reference_lens[3] * np.cos(theta)

        # shrink p1 and p2
        theta = np.arctan2((bbox[2][0] - bbox[1][0]), (bbox[2][1] - bbox[1][1]))
        bbox[1][0] += reference_lens[1] * np.sin(theta)
        bbox[1][1] += reference_lens[1] * np.cos(theta)
        bbox[2][0] -= reference_lens[2] * np.sin(theta)
        bbox[2][1] -= reference_lens[2] * np.cos(theta)
    else:
        # This means pair of vertical edge is "longer" than pair
        # of horizontal edges. So first shrink (p1, p2) and (p3, p0)
        # then shrink (p0, p1) and (p2, p3)
        theta = np.arctan2((bbox[3][0] - bbox[0][0]), (bbox[3][1] - bbox[0][1]))
        bbox[0][0] += reference_lens[0] * np.sin(theta)
        bbox[0][1] += reference_lens[0] * np.cos(theta)
        bbox[3][0] -= reference_lens[3] * np.sin(theta)
        bbox[3][1] -= reference_lens[3] * np.cos(theta)
        # shrink p1, p2
        theta = np.arctan2((bbox[2][0] - bbox[1][0]), (bbox[2][1] - bbox[1][1]))
        bbox[1][0] += reference_lens[1] * np.sin(theta)
        bbox[1][1] += reference_lens[1] * np.cos(theta)
        bbox[2][0] -= reference_lens[2] * np.sin(theta)
        bbox[2][1] -= reference_lens[2] * np.cos(theta)
        # shrink p0, p1
        theta = np.arctan2((bbox[1][1] - bbox[0][1]), (bbox[1][0] - bbox[0][0]))
        bbox[0][0] += reference_lens[0] * np.cos(theta)
        bbox[0][1] += reference_lens[0] * np.sin(theta)
        bbox[1][0] -= reference_lens[1] * np.cos(theta)
        bbox[1][1] -= reference_lens[1] * np.sin(theta)
        # shrink p2, p3
        theta = np.arctan2((bbox[2][1] - bbox[3][1]), (bbox[2][0] - bbox[3][0]))
        bbox[3][0] += reference_lens[3] * np.cos(theta)
        bbox[3][1] += reference_lens[3] * np.sin(theta)
        bbox[2][0] -= reference_lens[2] * np.cos(theta)
        bbox[2][1] -= reference_lens[2] * np.sin(theta)

    return bbox


def _point_to_line_dist(p1, p2, p3):
    """
    Find perpendicular distance from point p3 to line passing through
    p1 and p2.

    Reference: https://stackoverflow.com/a/39840218/5353128
    """
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def _align_vertices(bbox):
    """
    Align (sort) the vertices of the given bbox (rectangle) in such a way
    that the base of the rectangle forms minimum angle with horizontal axis.
    This is required because a single rectangle can be written in many
    ways (just by rotating the vertices in the list notation) such that the
    base of the rectangle will get changed in different notations and will form
    the angle which is multiple of original minimum angle.

    Reference: EAST implementation for ICDAR-2015 dataset:
    https://github.com/argman/EAST/blob/dca414de39a3a4915a019c9a02c1832a31cdd0ca/icdar.py#L352
    """
    p_lowest = np.argmax(bbox[:, 1])
    if np.count_nonzero(bbox[:, 1] == bbox[p_lowest, 1]) == 2:
        # This means there are two points in the horizantal axis (because two lowest points).
        # That means 0 angle.
        # The bottom edge is parallel to the X-axis, then p0 is the upper left corner.
        p0_index = np.argmin(np.sum(bbox, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return bbox[[p0_index, p1_index, p2_index, p3_index]], 0.0
    else:
        # Find the point to the right of the lowest point.
        p_lowest_right = (p_lowest - 1) % 4

        if bbox[p_lowest][0] == bbox[p_lowest_right][0]:
            print(p_lowest)

        angle = np.arctan(
            -(bbox[p_lowest][1] - bbox[p_lowest_right][1]) / (bbox[p_lowest][0] - bbox[p_lowest_right][0])
        )
        if angle / np.pi * 180 > 45:
            # Lowest point is p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return bbox[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
        else:
            # Lowest point is p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return bbox[[p0_index, p1_index, p2_index, p3_index]], angle


def generate_rbbox(image, bboxes, transcripts):
    """
    Generate RBOX (Rotated bbox) as per this paper:
    https://arxiv.org/pdf/1704.03155.pdf
    """
    img_h, img_w, _ = image.shape
    # geo_map is pixel/bbox location map which stores distances of
    # pixels from top, right, bottom and left from corresponding bbox edges
    # and angle of rotation of the bbox (4+1=5 channels).
    geo_map = np.zeros((img_h, img_w, 5), dtype = np.float32)
    # Single channel which indicates whether the pixel is part of text or
    # background.
    score_map = np.zeros((img_h, img_w), dtype = np.uint8)
    # Temporary bbox mask which is used as a helper.
    bbox_mask = np.zeros((img_h, img_w), dtype = np.uint8)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((img_h, img_w), dtype = np.uint8)

    # For each bbox, first shrink the bbox as per the paper.
    # For that, for each bbox, calculate the reference length (r_i)
    # for each bbox vertex p_i.
    
    final_bboxes = []
    for bbox_idx, bbox in enumerate(bboxes):
        # Reference length calculation
        reference_lens = []
        for idx in range(1, 5):
            reference_lens.append(
                min(
                    l2_norm(bbox[idx-1], bbox[(idx%4)+1-1]),
                    l2_norm(bbox[idx-1], bbox[((idx+2)%4)+1-1]),
                )
            )
        
        shrink_ratio = 0.3  # from papar
        shrunk_bbox = shrink_bbox(bbox.copy(), np.array(reference_lens), shrink_ratio).astype(np.int32)[np.newaxis, :, :]



        # if the poly is too small, then ignore it during training
        bbox_h = min(np.linalg.norm(bbox[0] - bbox[3]), np.linalg.norm(bbox[1] - bbox[2]))
        bbox_w = min(np.linalg.norm(bbox[0] - bbox[1]), np.linalg.norm(bbox[2] - bbox[3]))

        if bbox_h < 2*bbox_w:
            cv2.fillPoly(score_map, shrunk_bbox, 1)
            cv2.fillPoly(bbox_mask, shrunk_bbox, bbox_idx+1)


        if min(bbox_h, bbox_w) < 10 or bbox_h > 2*bbox_w:
            cv2.fillPoly(training_mask, bbox.astype(np.int32)[np.newaxis, :, :], 0)
        # if tag:
        #     cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        # Get all the points (in current bbox) in a helper mask
        bbox_points = np.argwhere(bbox_mask == (bbox_idx+1))

        # Now, as per the assumption, the bbox can be of any shape (quadrangle).
        # Therefore, to get the angle of rotation and pixel distances from the
        # bbox edges, fit a minimum area rectangle to bbox quadrangle.
        if bbox_h < 2*bbox_w:
            try:
                rectangle = minimum_bounding_rectangle(bbox)
            except Exception:
                # If could not find the min area rectangle, ignore that bbox while training
                cv2.fillPoly(training_mask, bbox.astype(np.int32)[np.newaxis, :, :], 0)
                continue
            else:
                rectangle, rotation_angle = _align_vertices(rectangle)
                final_bboxes.append(rectangle)  # TODO: Filter very small bboxes here

            # This rectangle has 4 vertices as required. Now, we can construct
            # the geo_map.
            for bbox_y, bbox_x in bbox_points:
                bbox_point = np.array([bbox_x, bbox_y], dtype=np.float32)
                # distance from top
                geo_map[bbox_y, bbox_x, 0] = _point_to_line_dist(rectangle[0], rectangle[1], bbox_point)
                # distance from right
                geo_map[bbox_y, bbox_x, 1] = _point_to_line_dist(rectangle[1], rectangle[2], bbox_point)
                # distance from bottom
                geo_map[bbox_y, bbox_x, 2] = _point_to_line_dist(rectangle[2], rectangle[3], bbox_point)
                # distance from left
                geo_map[bbox_y, bbox_x, 3] = _point_to_line_dist(rectangle[3], rectangle[0], bbox_point)
                # bbox rotation angle
                geo_map[bbox_y, bbox_x, 4] = rotation_angle

    # Size of the feature map from shared convolutions is 1/4th of
    # original image size. So all this geo_map should be of the
    # same size.
    # score_map = score_map[::4, ::4, np.newaxis].astype(np.float32)
    # geo_map = geo_map[::4, ::4].astype(np.float32)
    # training_mask = training_mask[::4, ::4, np.newaxis].astype(np.float32)
    score_map = score_map[:, :, np.newaxis].astype(np.float32)
    geo_map = geo_map[:, :].astype(np.float32)
    training_mask = training_mask[:, :, np.newaxis].astype(np.float32)

    return score_map, geo_map, training_mask, np.vstack(final_bboxes)


def shrink_poly(poly, r):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
            np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            [k, b] = np.polyfit(p1, p2, deg = 1)
            return [k, -1., b]
        


def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1 - b2) / (k1 - k2)
        y = k1 * x + b1
    return np.array([x, y], dtype = np.float32)


def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1. / line[0], -1, point[1] - (-1 / line[0] * point[0])]
    return verticle


def rectangle_from_parallelogram(poly):
    '''
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    '''
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            ## p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype = np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype = np.float32)
    else:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p1 and p3
            ## p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            ## p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype = np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype = np.float32)


def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        p0_index = np.argmin(np.sum(poly, axis = 1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        p_lowest_right = (p_lowest - 1) % 4
        angle = np.arctan(
            -(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0]))
        if angle / np.pi * 180 > 45:
            # ????????????p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
        else:
            # ????????????p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def generate_rbbox_v2(image, polys, tags):
    """
    Generate RBOX (Rotated bbox) as per this paper:
    https://arxiv.org/pdf/1704.03155.pdf
    """
    h, w, _ = image.shape
    poly_mask = np.zeros((h, w), dtype = np.uint8)
    score_map = np.zeros((h, w), dtype = np.uint8)
    geo_map = np.zeros((h, w, 5), dtype = np.float32)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((h, w), dtype = np.uint8)
    rectanges = []

    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]

        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        # score map
        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)
        cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
        # if the poly is too small, then ignore it during training
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        # if min(poly_h, poly_w) < FLAGS.min_text_size:
        if min(poly_h, poly_w) < 10:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        # if tag:
        #     cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
        # if geometry == 'RBOX':
        # ?????????????????????????????????????????????????????????
        fitted_parallelograms = []
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]
            try:
                edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
                backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
                forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            except:
                # ignore this bbox
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                cv2.fillPoly(score_map, shrinked_poly, 0)
                break
                
            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                # ???????????????p2
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                # ??????p3
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
            # move forward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p2 = line_cross_point(forward_edge, edge_opposite)
            if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            # or move backward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p3 = line_cross_point(backward_edge, edge_opposite)
            if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        areas = [Polygon(t).area for t in fitted_parallelograms]
        if len(areas) == 0:
            continue
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype = np.float32)
        # sort thie polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis = 1)
        min_coord_idx = np.argmin(parallelogram_coord_sum)
        parallelogram = parallelogram[
            [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

        rectange = rectangle_from_parallelogram(parallelogram)
        rectange, rotate_angle = sort_rectangle(rectange)
        rectanges.append(rectange.flatten())

        p0_rect, p1_rect, p2_rect, p3_rect = rectange
        for y, x in xy_in_poly:
            point = np.array([x, y], dtype = np.float32)
            # top
            geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
            # right
            geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
            # down
            geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
            # left
            geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
            # angle
            geo_map[y, x, 4] = rotate_angle
    
    # Size of the feature map from shared convolutions is 1/4th of
    # original image size. So all this geo_map should be of the
    # same size.
    score_map = score_map[::4, ::4, np.newaxis].astype(np.float32)
    geo_map = geo_map[::4, ::4].astype(np.float32)
    training_mask = training_mask[::4, ::4, np.newaxis].astype(np.float32)

    return score_map, geo_map, training_mask, np.vstack(rectanges)


def resize_image(image, image_size=512):
    """
    Resize the given image to image_size * image_size
    shaped square image.
    """
    # First pad the given image to match the image_size or image's larger
    # side (whichever is larger). [Create a square image]
    img_h, img_w, _ = image.shape
    max_size = max(image_size, img_w, img_h)

    # Create new square image of appropriate size
    img_padded = np.zeros((max_size, max_size, 3), dtype=np.float32)
    # Copy the original image into new image
    # (basically, new image is padded version of original image).
    img_padded[:img_h, :img_w, :] = image.copy()
    img_h, img_w, _ = img_padded.shape

    # if image_size higher that image sides, then the current padded
    # image will be of size image_size * image_size. But if not, resize the
    # padded iamge. This is done to keep the aspect ratio same even after
    # square resize.
    img_padded = cv2.resize(img_padded, dsize=(image_size, image_size))

    # We need the ratio of resized image width and heights to its
    # older dimensions to scale the bounding boxes accordingly
    scale_x = image_size / img_w
    scale_y = image_size / img_h

    return img_padded, scale_x, scale_y
