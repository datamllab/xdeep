
def find_layer(arch, target_layer_name):
    if target_layer_name is None:
        if 'resnet' in str(type(arch)):
            target_layer_name = 'layer4'
        elif 'alexnet' in str(type(arch)) or 'vgg' in str(type(arch)) or 'squeezenet' in str(
                type(arch)) or 'densenet' in str(type(arch)):
            target_layer_name = 'features'
        else:
            raise Exception('Invalid layer name! Please specify layer name.', target_layer_name)

    hierarchy = target_layer_name.split('_')

    if hierarchy[0] not in arch._modules.keys():
        raise Exception('Invalid layer name!', target_layer_name)

    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        if hierarchy[1] not in target_layer._modules.keys():
            raise Exception('Invalid layer name!', target_layer_name)
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        if hierarchy[2] not in target_layer._modules.keys():
            raise Exception('Invalid layer name!', target_layer_name)
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) >= 4:
        if hierarchy[3] not in target_layer._modules.keys():
            raise Exception('Invalid layer name!', target_layer_name)
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer