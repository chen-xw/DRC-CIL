from models.mafdrc import Mafdrc
from models.mafdrc_imagenet import Mafdrc_imagenet

def get_model(model_name, args):
    name = model_name.lower()
    if name == "mafdrc":
        return Mafdrc(args)
    elif name == "mafdrc_imagenet":
        return Mafdrc_imagenet(args)
    else:
        assert 0, "Not Implemented!"
