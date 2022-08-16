import bentoml
from bentoml.io import JSON
from bentoml._internal.types import JSONSerializable
from bentoml.io import NumpyNdarray, Image
from PIL.Image import Image as PILImage
import numpy as np
from eval_functions import get_transcript
from data_helpers.data_utils import resize_image
import cv2
from numpy.typing import NDArray
import typing as t

from bentoml._internal.store import Store
from bentoml._internal.store import StoreItem

# class DummyStore(Store[DummyItem]):
#     def __init__(self, base_path: "t.Union[PathType, FS]"):
#         super().__init__(base_path, DummyItem)

# store = Store("/root/bentoml/models/fots_model/")
# latest = store.get("fots_model:latest")

runner = bentoml.pytorch.get("fots_model:latest").to_runner()
svc = bentoml.Service(name="fots_model_runner", runners=[runner])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


@svc.api(input=Image(), output=NumpyNdarray(dtype="str"))
async def predict(input_img: PILImage) -> NDArray[t.Any]:
    input_orig = np.array(input_img)
    input_orig = cv2.cvtColor(input_orig, cv2.COLOR_RGB2BGR).astype(np.float32)
    img_arr, ratio_h, ratio_w = resize_image(np.array(input_img), 512)
    img_arr =img_arr/255.0
    input_arr = np.expand_dims(img_arr, 0).astype("float32")

    input_arr = np.transpose(input_arr,(0,3,1,2))

    score, geometry, preds, boxes, mapping, indices = await runner.async_run(input_arr)
    polys, pred_transcripts = get_transcript(input_img, input_orig, img_arr, preds, boxes, mapping, indices, False, None)
    # print("pred_transcripts",type(pred_transcripts))
    return np.array(pred_transcripts)