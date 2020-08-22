from collections import OrderedDict

import efficientnet_pytorch
import torch
import torch.nn as nn
import torchvision


class DenseNetModel(nn.Module):
    def __init__(self, arch, num_classes):
        super(DenseNetModel, self).__init__()
        model_func = getattr(torchvision.models, arch)
        self.model = model_func(pretrained=True)

        # check if classifier is Linear or Sequential Layer
        if isinstance(self.model.classifier, nn.Linear):
            in_features = self.model.classifier.in_features
            hidden_features = self.model.classifier.out_features
        else:
            for x in self.model.classifier:
                if isinstance(x, nn.Linear):
                    in_features = x.in_features
                    hidden_features = x.out_features
                    break

        # build a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_features, out_features=num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class ResNetModel(nn.Module):
    def __init__(self, arch, num_classes):
        super(ResNetModel, self).__init__()
        model_func = getattr(torchvision.models, arch)
        self.model = model_func(pretrained=True)

        # check if classifier is Linear or Sequential Layer
        if isinstance(self.model.fc, nn.Linear):
            in_features = self.model.fc.in_features
            hidden_features = self.model.fc.out_features
        else:
            for x in self.model.fc:
                if isinstance(x, nn.Linear):
                    in_features = x.in_features
                    hidden_features = x.out_features
                    break

        # build a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_features, out_features=num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class EfficientNetModel(nn.Module):
    def __init__(self, arch, num_classes):
        super(EfficientNetModel, self).__init__()
        self.model = efficientnet_pytorch.EfficientNet.from_pretrained(arch)

        if isinstance(self.model._fc, nn.Linear):
            in_features = self.model._fc.in_features
            hidden_features = self.model._fc.out_features
        else:
            for x in self.model._fc:
                if isinstance(x, nn.Linear):
                    in_features = x.in_features
                    hidden_features = x.out_features
                    break

        self.model._fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_features, out_features=num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def load_from_checkpoint(self, ckp_path, device=torch.device('cpu')):
        model_checkpoint = torch.load(ckp_path, map_location=device)
        state_dict = model_checkpoint['state_dict']

        new_state_dict = OrderedDict()
        for k in state_dict:
            short_key = k.replace('module.efficientnet.', '')
            new_state_dict[short_key] = state_dict[k]

        old_state_dict = self.model.state_dict()
        for k in new_state_dict:
            if k not in old_state_dict:
                print('Unexpected key %s in state_dict' % k)
        for k in old_state_dict:
            if k not in new_state_dict:
                print('Missing key %s in state_dict' % k)

        self.model.load_state_dict(new_state_dict, strict=False)


def DenseNet121(num_classes=14):
    return DenseNetModel('densenet121', num_classes)


def DenseNet169(num_classes=14):
    return DenseNetModel('densenet169', num_classes)


def DenseNet201(num_classes=14):
    return DenseNetModel("densenet201", num_classes)


def ResNet34(num_classes=14):
    return ResNetModel("resnet34", num_classes)


def ResNet50(num_classes=14):
    return ResNetModel('resnet50', num_classes)


def ResNet101(num_classes=14):
    return ResNetModel("resnet101", num_classes)


def EfficientNet4(num_classes=14):
    return EfficientNetModel("efficientnet-b4", num_classes)


def EfficientNet5(num_classes=14):
    return EfficientNetModel('efficientnet-b5', num_classes)


def EfficientNet6(num_classes=14):
    return EfficientNetModel('efficientnet-b6', num_classes)
