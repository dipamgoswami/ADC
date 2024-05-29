from models.lwf import LwF

def get_model(model_name, args):
    name = model_name.lower()
    if name == "lwf":
        return LwF(args)
    else:
        assert 0
