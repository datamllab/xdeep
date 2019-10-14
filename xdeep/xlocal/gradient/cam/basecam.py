

class BaseCAM(object):

    """
        Base class for Class Activation Mapping.
    """

    def __init__(self, model_dict):
        """Init

            # Arguments
                model_dict: dict. A dict with format model_dict = dict(arch=self.model, layer_name=target_layer_name).

        """

        layer_name = model_dict['layer_name']

        self.model_arch = model_dict['arch']
        self.model_arch.eval()
        self.gradients = dict()
        self.activations = dict()

        # save gradient
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        # save activation map
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        target_layer = self.find_layer(self.model_arch, layer_name)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def find_layer(self, arch, target_layer_name):

        if target_layer_name is None:
            if 'resnet' in str(type(arch)):
                target_layer_name = 'layer4'
            elif 'alexnet' in str(type(arch)) or 'vgg' in str(type(arch)) or 'squeezenet' in str(type(arch)) or 'densenet' in str(type(arch)):
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

    def forward(self, input_, class_idx=None, retain_graph=False):
        return None

    def __call__(self, input_, class_idx=None, retain_graph=False):
        return self.forward(input_, class_idx, retain_graph)
