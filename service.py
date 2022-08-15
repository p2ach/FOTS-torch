import bentoml
from bentoml.io import JSON
from bentoml._internal.types import JSONSerializable
from bentoml.io import NumpyNdarray, Image
from PIL.Image import Image as PILImage
import numpy as np

runner = bentoml.pytorch.get("fots_model:cr5zh2a4w25fdwos").to_runner()

svc = bentoml.Service(name="fots_model_runner", runners=[runner])

@svc.api(input=Image(), output=NumpyNdarray(dtype="int64"))
async def predict(input_img: PILImage):
    img_arr = np.array(input_img)/255.0
    input_arr = np.expand_dims(img_arr, 0).astype("float32")
    batch_ret = await runner.async_run([input_arr])
    return batch_ret