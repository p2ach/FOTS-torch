import bentoml
import argparse
import torch
from model import FOTSModel
# import FOTSModel
from bentoml.io import JSON
from bentoml._internal.types import JSONSerializable
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
    saved_model = bentoml.pytorch.save_model(
            model = model,
            name = "fots_model",
        signatures={"__call__": {"batchable": False, "batchdim": 0}},
    )
    print(f"Model saved: {saved_model}")
    # runner = bentoml.pytorch.get("fots_model").to_runner()
    #
    # svc = bentoml.Service(name="fots_model", runners=[runner])
    #
    # @svc.api(input=JSON(), output=JSON())
    # async def predict(json_obj: JSONSerializable) -> JSONSerializable:
    #     batch_ret = await runner.async_run([json_obj])
    #     return batch_ret[0]
    # @svc.api(input=JSON(), output=JSON())
    # async def predict(json_obj: JSONSerializable) -> JSONSerializable:
    #     batch_ret = await runner.async_run([json_obj])
    #     return batch_ret[0]



    # x = torch.randn(1, 3, 224, 224, requires_grad=True)
    #
    # # Export the model
    # torch.onnx.export(
    #     model,
    #     x,
    #     "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
    #     #export_params=True,  # store the trained parameter weights inside the model file
    #     opset_version=11,  # the ONNX version to export the model to
    #     do_constant_folding=True,  # whether to execute constant folding for optimization
    #     input_names=["input"],  # the model's input names
    #     output_names=["output"] #the model's output names
    # )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="./models/FOTS_last_checkpoint.pt", type=str,
        help='Path to trained model'
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="/app/FOTS-torch/data_folder/output_eval",
        help="Output directory to save predictions"
    )
    parser.add_argument(
        "-i", "--input_dir", type=str, default="/app/FOTS-torch/data_folder/image",
        help="Input directory having images to be predicted"
    )
    args = parser.parse_args()
    inference(args)