import torch

from model import Unet4to4

if __name__ == "__main__":
    input_shape = (1, 1, 1112, 2028)
    checkpoint_path = "Unet4to4-8ch.pth"

    _input = torch.randn(input_shape, dtype=torch.float32)
    model = Unet4to4()
    model.load_state_dict(torch.load(checkpoint_path))
    model._set_alpha(1)
    output = model(_input)
    
    torch.onnx.export(
        model,
        _input,
        'Unet4to4-8ch.onnx',
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}})

