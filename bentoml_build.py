import bentoml
import argparse
import torch
from model import FOTSModel
import numpy as np
# import FOTSModel
from bentoml.io import JSON
from bentoml._internal.types import JSONSerializable
import cv2
import os
from data_helpers.data_utils import resize_image
from utils import TranscriptEncoder, classes
from bbox import Toolbox
from eval_functions import get_transcript


transcript_encoder = TranscriptEncoder(classes)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _load_model(model_path):
    """Load model from given path to available device."""
    model = FOTSModel()
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE)["model"])
    return model

def inference(args):
    """FOTS Inference on give images."""
    model = _load_model(args.model)
    # model.eval()
    # model.training=False
    saved_model = bentoml.pytorch.save(
            model = model,
            name = "fots_model",
        signatures={"__call__": {"batchable": False, "batchdim": 0}},
    )
    print(f"Model saved: {saved_model}")
    return saved_model


def test_runner(saved_model,input_img = "img_513.jpg",with_img=True,output_dir="./data_folder/output_eval"):
    input_orig=cv2.imread(input_img)
    runner = bentoml.pytorch.get(saved_model.tag).to_runner()

    input_np = cv2.cvtColor(input_orig, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_np, _, _ = resize_image(input_np, 512)
    img_arr = np.array(input_np) / 255.0
    input_arr = np.expand_dims(img_arr, 0).astype("float32")
    input_arr = np.transpose(input_arr, (0, 3, 1, 2))
    runner.init_local()
    score, geometry, preds, boxes, mapping, indices = runner.run(input_arr)
    return get_transcript(input_img, input_orig, img_arr, preds, boxes, mapping, indices, with_img, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="./models/FOTS_last_checkpoint.pt", type=str,
        help='Path to trained model'
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="./data_folder/output_eval",
        help="Output directory to save predictions"
    )
    parser.add_argument(
        "-i", "--input_dir", type=str, default="./data_folder/image",
        help="Input directory having images to be predicted"
    )
    parser.add_argument(
        "-g", "--input_img", type=str, default="img_513.jpg",
        help="Input directory having images to be predicted"
    )
    parser.add_argument(
        "-w", "--with_img", type=str, default="img_513.jpg",
        help="Input directory having images to be predicted"
    )
    args = parser.parse_args()
    saved_model = inference(args)
    polys, pred_transcripts = test_runner(saved_model,args.input_img,args.with_img,args.output_dir)
    print("pred_transcripts",pred_transcripts)