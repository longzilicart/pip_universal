# if load_state_dict = False but with same key name. still load the model without raise exception


def load_partial_state_dict(model, state_dict):
    """
    intro:
        When the shapes do not match, attempt partial weight loading or skip that weight. This can be used to load old checkpoints   
    """
    own_state = model.state_dict()

    for name, param in state_dict.items():
        if name not in own_state:
            print(f'skip key: {name}')
            continue
        own_param = own_state[name]
        
        # the shape is match
        if own_param.shape == param.shape:
            own_param.copy_(param)

        # the shape is not match, just skip
        elif len(own_param.shape) == len(param.shape):
            # channel is the first dimension
            print(f"skip loadding: {name}")
            continue
            min_channels = min(own_param.shape[1], param.shape[1])
            own_param.data[:, :min_channels].copy_(param.data[:, :min_channels])
            print(f"partial load: {name}")

        # the shape is totally different
        else:
            print(f"shape do not match, skip key: {name}")

    # load the updated state dict
    model.load_state_dict(own_state, strict=False)
    return model




