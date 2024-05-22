from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4, resnet116, resnet200, resnet110x2, resnet20x4, resnet26x4, resnet44x4, resnet56x4, resnet110x4
from .vgg import VGG19_bn, VGG16_bn, VGG13_bn, VGG11_bn, VGG8_bn
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .mobilenetv2 import mobile_half, mobile_half_double
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2, ShuffleV2_1_5

from .resnet_imagenet import resnet18, resnet34, resnet50, resnet101, wide_resnet50_2, resnext50_32x4d, wide_resnet10_2, wide_resnet18_2, wide_resnet34_2
from .mobilenetv2_imagenet import MobileNetV2
from .shuffleNetv2_imagenet import shufflenet_v2_x1_0
from .resnetv2 import ResNet50

model_dict = {
    'resnet8'  : resnet8,
    'resnet14'  : resnet14,
    'resnet20'  : resnet20,
    'resnet32'  : resnet32,
    'resnet44'  : resnet44,
    'resnet56'  : resnet56,
    'resnet101' : resnet101,
    'resnet110'  : resnet110,
    'resnet116' : resnet116,
    'resnet200' : resnet200,
    'resnet110x2' : resnet110x2,
    'resnet8x4'  : resnet8x4,
    'resnet20x4' : resnet20x4,
    'resnet26x4' : resnet26x4,
    'resnet32x4'  : resnet32x4,
    'resnet44x4' : resnet44x4,
    'resnet56x4' : resnet56x4,
    'resnet110x4' : resnet110x4,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8' : VGG8_bn,
    'vgg11' : VGG11_bn,
    'vgg13' : VGG13_bn,
    'vgg16' : VGG16_bn,
    'vgg19' : VGG19_bn,
    'mobilenetv2': mobile_half,
    'mobilenetv2_1_0': mobile_half_double,
    'shufflev1': ShuffleV1,
    'shufflev2': ShuffleV2,
    'shufflev2_1_5': ShuffleV2_1_5,
    
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'ResNet50': ResNet50,
    'resnext50_32x4d': resnext50_32x4d,
    'wrn_10_2': wide_resnet10_2,
    'wrn_18_2': wide_resnet18_2,
    'wrn_34_2': wide_resnet34_2,
    'wrn_50_2': wide_resnet50_2,
    
    'mobilenetv2_imagenet': MobileNetV2,
    'shufflev2_imagenet': shufflenet_v2_x1_0,
}