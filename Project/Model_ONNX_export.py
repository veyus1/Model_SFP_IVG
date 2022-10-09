import torch
import torchvision
import onnx
import onnxruntime
from Model_maskRCNN_different_backbone import get_model_w_new_bb
from Model_maskRCNN_droplets import get_transform, get_model_object_detection



def pth_to_onnx(pth_path, name):
    """function to load a .pth model and export it as .onnx"""
    # Input to the model
    #model = get_model_object_detection(num_classes=2) # this for res50
    model = get_model_w_new_bb(num_classes=2) # this for res18
    model.load_state_dict(torch.load(pth_path))
    model.eval()

    # create dummy input
    x = torch.randn(1, 3, 1024, 144, requires_grad=True)

    # Export the model
    torch.onnx.export(model,                        # model being run
                      x,                            # model input (or a tuple for multiple inputs)
                      name,      # where to save the model (can be a file or file-like object)
                      export_params=True,           # store the trained parameter weights inside the model file
                      opset_version=11,             # the ONNX version to export the model to
                      do_constant_folding=True,     # whether to execute constant folding for optimization
                      input_names=['input'],        # the model's input names
                      output_names=['output'],      # the model's output names
                      dynamic_axes={'input': {0: 'batch_size', 2:"height", 3:"width"},  # variable length axes
                                    'output': {0: 'batch_size'}})


pth_to_onnx("Weights/final/weights_500_50_20_res18_2b_final.pth", "Weights/ONNX/weights_500_50_20_res18_2b_final.onnx")