import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple


# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def train_dropout(x, droprate):
    x = torch.nn.functional.dropout(x, p=droprate, training=True)
    return x


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class bottleneck_IR(Module):
    def __init__(self, drate, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            Dropout(drate),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            Dropout(drate),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['drate', 'in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(drate, in_channel, depth, num_units, stride=2):

    return [Bottleneck(drate, in_channel, depth, stride)] + [Bottleneck(drate, depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers, drate):
    if num_layers == 50:
        blocks = [
            get_block(drate = drate, in_channel=64, depth=64, num_units=3),
            get_block(drate = drate, in_channel=64, depth=128, num_units=4),
            get_block(drate = drate, in_channel=128, depth=256, num_units=14),
            get_block(drate = drate, in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(drate = drate, in_channel=64, depth=64, num_units=3),
            get_block(drate = drate, in_channel=64, depth=128, num_units=13),
            get_block(drate = drate, in_channel=128, depth=256, num_units=30),
            get_block(drate = drate, in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(drate = drate, in_channel=64, depth=64, num_units=3),
            get_block(drate = drate, in_channel=64, depth=128, num_units=8),
            get_block(drate = drate, in_channel=128, depth=256, num_units=36),
            get_block(drate = drate, in_channel=256, depth=512, num_units=3)
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, drate, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers, drate)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      Dropout(drate),
                                      BatchNorm2d(64),
                                      PReLU(64))
        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(drate,
                                bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        x = x - 127.5
        x = x * 0.078125
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


def IR_50_drop(input_size, drate):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, drate, 50, 'ir')

    return model


def IR_101_drop(input_size, drate):
    """Constructs a ir-101 model.
    """
    model = Backbone(input_size, drate, 100, 'ir')

    return model


def IR_152_drop(input_size, drate):
    """Constructs a ir-152 model.
    """
    model = Backbone(input_size, drate, 152, 'ir')

    return model



