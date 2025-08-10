import torch
import torch.onnx
from models.model import MyModel
import sys

def export_model_to_onnx(model_path, onnx_path, input_size):
    model = MyModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = torch.randn(input_size)

    torch.onnx.export(model, dummy_input, onnx_path, 
                      export_params=True, 
                      opset_version=11, 
                      do_constant_folding=True, 
                      input_names=['input'], 
                      output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size'}, 
                                    'output': {0: 'batch_size'}})

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python export_onnx.py <model_path> <onnx_path> <input_size>")
        sys.exit(1)

    model_path = sys.argv[1]
    onnx_path = sys.argv[2]
    input_size = tuple(map(int, sys.argv[3].strip('()').split(',')))

    export_model_to_onnx(model_path, onnx_path, input_size)