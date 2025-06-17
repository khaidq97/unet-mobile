from .models import *

def get_model(model_name, num_classes=1, input_shape=(64, 64, 3), **kwargs):
    if model_name == "unet_micro":
        model = unet_micro(num_classes=num_classes, input_shape=input_shape)
    elif model_name == "unet_nano":
        model = unet_nano(num_classes=num_classes, input_shape=input_shape)
    else:
        raise ValueError(f"Model {model_name} not found")
    
    # Load pretrained weights if provided
    pretrained = kwargs.get("pretrained", None)
    if pretrained:
        print(f"Loading pretrained weights from {pretrained}")
        model.load_weights(pretrained)
    return model