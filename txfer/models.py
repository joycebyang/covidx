from collections import OrderedDict

import efficientnet_pytorch
import torch
import torch.nn as nn
import torchvision


class DenseNetModel(nn.Module):
    def __init__(self, arch, freeze_pretrain_weights=True, num_classes=3):
        super(DenseNetModel, self).__init__()
        model_func = getattr(torchvision.models, arch)
        self.model = model_func(pretrained=True)
        self.freeze_pretrain_weights = freeze_pretrain_weights

        # freezing the pretrained weights
        if self.freeze_pretrain_weights:
            for param in self.model.parameters():
                param.requires_grad = False

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
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

    def get_optimizer_parameters(self):
        if self.freeze_pretrain_weights:
            return self.model.classifier.parameters()
        else:
            return self.model.parameters()


class ResNetModel(nn.Module):
    def __init__(self, arch, freeze_pretrain_weights=True, num_classes=3):
        super(ResNetModel, self).__init__()
        model_func = getattr(torchvision.models, arch)
        self.model = model_func(pretrained=True)
        self.freeze_pretrain_weights = freeze_pretrain_weights

        # freezing the pretrained weights
        if self.freeze_pretrain_weights:
            for param in self.model.parameters():
                param.requires_grad = False

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
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

    def get_optimizer_parameters(self):
        if self.freeze_pretrain_weights:
            return self.model.fc.parameters()
        else:
            return self.model.parameters()


class EfficientNetModel(nn.Module):
    def __init__(self, arch, freeze_pretrain_weights=True, num_classes=3):
        super(EfficientNetModel, self).__init__()
        self.model = efficientnet_pytorch.EfficientNet.from_pretrained(arch)
        self.freeze_pretrain_weights = freeze_pretrain_weights

        # freezing the pretrained weights
        if self.freeze_pretrain_weights:
            for param in self.model.parameters():
                param.requires_grad = False

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
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

    def get_optimizer_parameters(self):
        if self.freeze_pretrain_weights:
            return self.model._fc.parameters()
        else:
            return self.model.parameters()

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


def DenseNet121(freeze_pretrain_weights):
    return DenseNetModel('densenet121', freeze_pretrain_weights)


def DenseNet169(freeze_pretrain_weights):
    return DenseNetModel('densenet169', freeze_pretrain_weights)


def DenseNet201(freeze_pretrain_weights):
    return DenseNetModel("densenet201", freeze_pretrain_weights)


def ResNet34(freeze_pretrain_weights):
    return ResNetModel("resnet34", freeze_pretrain_weights)


def ResNet50(freeze_pretrain_weights):
    return ResNetModel('resnet50', freeze_pretrain_weights)


def ResNet101(freeze_pretrain_weights):
    return ResNetModel("resnet101", freeze_pretrain_weights)


def EfficientNet4(freeze_pretrain_weights):
    return EfficientNetModel("efficientnet-b4", freeze_pretrain_weights)


def EfficientNet5(freeze_pretrain_weights):
    return EfficientNetModel('efficientnet-b5', freeze_pretrain_weights)


def EfficientNet6(freeze_pretrain_weights):
    return EfficientNetModel('efficientnet-b6', freeze_pretrain_weights)
