from bbox import Toolbox
from utils import TranscriptEncoder, classes
import os
import cv2
import numpy as np
transcript_encoder = TranscriptEncoder(classes)

def get_transcript(input_img, input_orig, img_arr, preds, boxes, mapping, indices, with_img, output_dir):

    ratio_w =img_arr.shape[1]
    ratio_h =img_arr.shape[0]


    if len(boxes) != 0:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    pred_transcripts = []
    if len(mapping) > 0:
        pred, lengths = preds
        _, pred = pred.max(2)
        for idx in range(lengths.numel()):
            l = lengths[idx]
            p = pred[:l, idx]
            txt = transcript_encoder.decode(p, l)
            pred_transcripts.append(txt)
        pred_transcripts = np.array(pred_transcripts)
        pred_transcripts = pred_transcripts[indices]

    polys = []
    if len(boxes) != 0:
        for box, txt in zip(boxes, pred_transcripts):
            box = Toolbox.sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                # print('wrong direction')
                continue
            poly = np.array([[box[0, 0], box[0, 1]], [box[1, 0], box[1, 1]], [box[2, 0], box[2, 1]],
                             [box[3, 0], box[3, 1]]])
            polys.append(polys)
            p_area = Toolbox.polygon_area(poly)
            if p_area > 0:
                poly = poly[(0, 3, 2, 1), :]

            if with_img:
                cv2.polylines(input_orig[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                              color=(255, 255, 0), thickness=1)
                cv2.putText(input_orig[:, :, ::-1], txt, (box[0][0], box[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 0), 1)

    if with_img and output_dir:
        img_path = os.path.join(output_dir ,input_img)
        cv2.imwrite(img_path, input_orig[:, :, ::-1])

    return polys, pred_transcripts