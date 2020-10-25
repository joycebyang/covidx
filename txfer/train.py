import argparse
import os
import random
import time
from collections import OrderedDict

import models
import numpy as np
import torch
import torchvision
from PIL import Image
from dataset_generator import DatasetGenerator
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from weighted_covid_sampler import WeightedCovidSampler

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class_to_idx = {
    'normal': 0,
    'pneumonia': 1,
    'COVID-19': 2
}


def get_input_args():
    parser = argparse.ArgumentParser()

    available_models = [m for m in dir(models) if 'Net' in m and 'Model' not in m]
    parser.add_argument('--arch', choices=available_models, type=str, help='name of pretrained model used')

    parser.add_argument('--epoch', type=int, default=2, help='number of training epochs')

    parser.add_argument('--gpu', action='store_true', help='use gpu if available')

    parser.add_argument('--batch', type=int, default=32, help='number of images per batch')

    parser.add_argument('--print_every', type=int, default=50, help='frequency of validation ')

    parser.add_argument('--image_dir', type=str, default='data', help='directory of training images')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='directory of saved checkpoint file')

    sampler_choices = ['intact', 'balanced', 'weighted']
    parser.add_argument('--sampler_choice', choices=sampler_choices, type=str, default='intact',
                        help='sampler method; if weighted, must have covid percent')

    parser.add_argument('--covid_percent', required=False, type=float, default=0.3,
                        help='percent of covid images in batch')

    parser.add_argument('--prior_checkpoint', type=str, required=False,
                        help='checkpoint path of previously trained model')

    parser.add_argument('--learning_rate', type=float, required=True, help='step size in optimizer')

    parser.add_argument('--unfreeze_weights', action='store_true',
                        help='when specified, pretrained weighted are unfrozen and trained')

    parser.add_argument('--image_size', type =int, default=224, help='image resolution')

    return parser.parse_args()


print(torch.__version__)
print(torchvision.__version__)

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# TODO: Build and train your network
def load_pretrained_model(arch, unfreeze_weights):
    model_func = getattr(models, arch)
    model = model_func(freeze_pretrain_weights=not unfreeze_weights)
    model.arch = arch

    return model


# TODO: convert labels to tensor
def labels_to_tensor(labels):
    label_indices = [class_to_idx[label] for label in labels]
    label_indices = np.array(label_indices, int)
    label_indices = torch.LongTensor(label_indices)
    return label_indices


def validate(model, valid_dataloader, loss_func, device):
    # TODO: track accuracy and loss
    accuracy = 0
    test_loss = 0

    with torch.no_grad():  # TODO: deactivates requires_grad flag, disables tracking of gradients
        for images, labels in valid_dataloader:  # iterate over images and labels in valid dataset
            labels = labels_to_tensor(labels)
            images, labels = images.to(device), labels.to(device)  # move a tensor to a device
            log_ps = model.forward(images)  # log form of probabilities for each label
            test_loss += loss_func(log_ps,
                                   labels).item()  # .item() returns loss value as float, compare prob to actual

            ps = torch.exp(log_ps)  # gets rid of log
            equality = (labels.data == ps.max(dim=1)[1])  # takes highest probability
            accuracy += torch.mean(equality.type(torch.FloatTensor))

    return test_loss, accuracy


# TODO: Do validation on the test set
def test_model(model, test_loader, device):
    # track accuracy, move to device, switch on eval mode
    accuracy = 0
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in iter(test_loader):
            labels = labels_to_tensor(labels)
            images, labels = images.to(device), labels.to(device)  # move a tensor to a device
            log_ps = model.forward(images)
            ps = torch.exp(log_ps)

            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        model_accuracy = accuracy / len(test_loader)
        model.accuracy = model_accuracy

    return model.accuracy


# TODO: Save the checkpoint
def save_checkpoint(checkpoint_path, model):
    checkpoint = {
        # TODO: Save the model arch, accuracy, classifier, class_to_idx
        'arch': model.arch,
        'metrics': model.metrics,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict(),
        'freeze_pretrain_weights': model.freeze_pretrain_weights
    }
    torch.save(checkpoint, checkpoint_path)
    print('Saved the trained model: %s' % checkpoint_path)


# TODO: load checkpoint dictionary
# TODO: perform sanity check and debugging, load checkpoint state_dict into pretrained model
def load_checkpoint(checkpoint_path, device, pretrained_model):
    model_checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    freeze_pretrain_weights = True

    if 'freeze_pretrain_weights' in model_checkpoint:
        freeze_pretrain_weights = model_checkpoint['freeze_pretrain_weights']
    if freeze_pretrain_weights != pretrained_model.freeze_pretrain_weights:
        print('Freeze pretrain weights value mismatch: Checkpoint={} vs Pretrained={}'.format(
            freeze_pretrain_weights,
            pretrained_model.freeze_pretrain_weights))

    checkpoint_state_dict = model_checkpoint['state_dict']
    new_state_dict = OrderedDict()

    state_dict = pretrained_model.state_dict()
    for k in checkpoint_state_dict:
        if k not in state_dict:
            print('Unexpected key %s in state_dict' % k)
        elif checkpoint_state_dict[k].shape != state_dict[k].shape:
            print('Size mismatch for {}: Checkpoint={} vs Pretrained={}'.format(k, checkpoint_state_dict[k].shape,
                                                                                state_dict[k].shape))
        else:
            new_state_dict[k] = checkpoint_state_dict[k]

    for k in state_dict:
        if k not in new_state_dict:
            print('Missing key %s in state_dict' % k)

    pretrained_model.load_state_dict(new_state_dict, strict=False)


# TODO: return dataloader for balanced sampler method
def get_balanced_dataloader(train_dataset, batch_size, num_workers):
    train_label_cnt = [0, 0, 0]
    train_labels = []

    # TBD shuffle the training data
    # TBD this is best done inside the DatasetGenerator class
    # train_dataset.csv_df = train_dataset.csv_df.sample(frac=1)
    for label in train_dataset.csv_df[2]:
        train_label_cnt[class_to_idx[label]] += 1
        train_labels.append(class_to_idx[label])

    train_num_samples = sum(train_label_cnt)
    train_class_weights = [train_num_samples / train_label_cnt[i] for i in range(len(train_label_cnt))]
    train_weights = [train_class_weights[train_labels[i]] for i in range(int(train_num_samples))]

    # TBD create WeightedRandomSampler to balance the training data set
    train_sampler = WeightedRandomSampler(torch.DoubleTensor(train_weights), int(train_num_samples))

    return DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                      sampler=train_sampler)


# TODO: return dataloader for Weighted COVID Sampler Method
def get_weighted_dataloader(train_dataset, batch_size, covid_percent, num_workers):
    # list of indices for each class
    train_noncovid_list = []
    train_covid_list = []

    for index in range(len(train_dataset.csv_df)):
        label = train_dataset.csv_df.iloc[index, 2]
        if label == 'COVID-19':
            train_covid_list.append(index)
        else:
            train_noncovid_list.append(index)

    weighted_covid_sampler = WeightedCovidSampler(train_noncovid_list, train_covid_list, batch_size, covid_percent)

    return DataLoader(train_dataset, batch_sampler=weighted_covid_sampler, num_workers=num_workers)


# TODO: calculate Precision, Recall, Average Precision
def calc_prc_on_test_data(pretrained_model, test_loader, device):
    pred_out = torch.FloatTensor().to(device)
    labels_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            # TODO: append the labels
            images = images.to(device)
            labels_list.extend(labels)
            log_ps = pretrained_model.forward(images)
            ps = torch.exp(log_ps).squeeze()
            # TODO: append the prediction output using torch.cat
            pred_out = torch.cat((pred_out, ps), 0)

    pred_out = pred_out.cpu()

    precision = dict()
    recall = dict()
    average_precision = dict()

    for clz in class_to_idx:
        clz_pred_out = pred_out[:, class_to_idx[clz]].numpy()
        # TBD calculate precision and recall curve and store in precision, recall
        precision[clz], recall[clz], _ = precision_recall_curve(y_true=labels_list,
                                                                probas_pred=clz_pred_out,
                                                                pos_label=clz)

        clz_labels_list = [clz == label for label in labels_list]
        # TBD calculate average precision score (the area under the curve)
        average_precision[clz] = average_precision_score(y_true=clz_labels_list,
                                                         y_score=clz_pred_out)

    # TODO: calc confusion matrix values
    max_pred_out = pred_out.max(dim=1)
    max_pred_indices = max_pred_out.indices.numpy()

    labels_indices = [class_to_idx[label] for label in labels_list]
    cm = confusion_matrix(y_true=labels_indices, y_pred=max_pred_indices)
    normalized_row_cm = confusion_matrix(y_true=labels_indices, y_pred=max_pred_indices, normalize='true')
    normalized_col_cm = confusion_matrix(y_true=labels_indices, y_pred=max_pred_indices, normalize='pred')

    f1_scores = f1_score(y_true=labels_indices, y_pred=max_pred_indices, average=None)

    return {'precision': precision,
            'recall': recall,
            'average_precision': average_precision,
            'cm': cm,
            'normalized_row_cm': normalized_row_cm,
            'normalized_col_cm': normalized_col_cm,
            'f1_scores': f1_scores}


def train(model, loss_func, optimizer, dataloaders, device, epochs, checkpoint_prefix, print_every):
    device = torch.device(device)
    model.to(device)

    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    test_loader = dataloaders['test']

    epoch_start = time.time()
    max_acc = 0.0

    # loop to train for number of epochs
    for e in range(epochs):
        running_loss = 0
        batch_start = time.time()
        steps = 0

        for images, labels in train_loader:
            # within each loop, iterate train_loader, and print loss
            label_indices = labels_to_tensor(labels)
            images, labels = images.to(device), label_indices.to(device)
            optimizer.zero_grad()
            log_ps = model.forward(images)
            loss = loss_func(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            steps += 1

            if steps % print_every == 0:
                model.eval()
                valid_loss, valid_accuracy = validate(model, valid_loader, loss_func, device)
                model.train()
                batch_time = time.time() - batch_start
                print(
                    "Epoch: {}/{}..".format(e + 1, epochs),
                    "Step: {}..".format(steps),
                    "Training Loss: {:.3f}..".format(running_loss / len(train_loader)),
                    "Test Loss: {:.3f}..".format(valid_loss / len(valid_loader)),
                    "Test Accuracy: {:.3f}..".format(valid_accuracy / len(valid_loader)),
                    "Batch Time: {:.3f}, avg: {:.3f}".format(batch_time, batch_time / steps)
                )

        model.eval()
        metrics = calc_prc_on_test_data(model, test_loader, device)
        model.train()
        epoch_time = time.time() - epoch_start

        if metrics['average_precision']['COVID-19'] > max_acc:
            max_acc = metrics['average_precision']['COVID-19']
            model.metrics = metrics
            save_checkpoint('%s.pth.tar' % checkpoint_prefix, model)
            print('Epoch [{}] [save] Average Precision={} f1_score={} time: {:.3f}, avg: {:.3f}'
                  .format(e + 1, metrics['average_precision'], metrics['f1_scores'], epoch_time, epoch_time / (e + 1)))
        else:
            print('Epoch [{}] [----] Average Precision={} f1_score={} time: {:.3f}, avg: {:.3f}'
                  .format(e + 1, metrics['average_precision'], metrics['f1_scores'], epoch_time, epoch_time / (e + 1)))

        print('Confusion Matrix: {}'.format(metrics['cm']))

    return model


def main():
    in_arg = get_input_args()

    training_start = time.time()
    checkpoint_prefix = os.path.join(in_arg.checkpoint_dir, '%s-%d' % (in_arg.arch, training_start))
    print('{} --> {}'.format(in_arg, checkpoint_prefix))

    train_csv_file = 'newtrain_split.txt'
    test_csv_file = 'test_split.txt'
    valid_csv_file = 'valid_split.txt'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomOrder([
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15, resample=Image.BILINEAR),
            transforms.RandomResizedCrop(in_arg.image_size, scale=(0.9, 1.0)),
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(in_arg.image_size),
        transforms.CenterCrop(in_arg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transforms = valid_transforms

    # TODO: Load the datasets with DatasetGenerator and DataLoader

    train_dir = os.path.join(in_arg.image_dir, 'train')
    train_dataset = DatasetGenerator(train_csv_file, train_dir, transform=train_transforms)

    test_dir = os.path.join(in_arg.image_dir, 'test')
    test_dataset = DatasetGenerator(test_csv_file, test_dir, transform=test_transforms)

    valid_dir = os.path.join(in_arg.image_dir, 'train')
    valid_dataset = DatasetGenerator(valid_csv_file, valid_dir, transform=valid_transforms)

    # TODO: check sampler method, based on choice, create dataloader for train dataset
    if in_arg.sampler_choice == 'balanced':
        train_loader = get_balanced_dataloader(train_dataset, batch_size=in_arg.batch, num_workers=16)
    elif in_arg.sampler_choice == 'weighted':
        train_loader = get_weighted_dataloader(train_dataset, batch_size=in_arg.batch, num_workers=16,
                                               covid_percent=in_arg.covid_percent)
    else:
        train_loader = DataLoader(train_dataset, batch_size=in_arg.batch, num_workers=16, shuffle=True)

    dataloaders = {"train": train_loader,
                   "test": DataLoader(test_dataset, batch_size=in_arg.batch, num_workers=16, shuffle=True),
                   "valid": DataLoader(valid_dataset, batch_size=in_arg.batch, num_workers=16, shuffle=True)}

    # TODO: make sure if GPU is available, we can use it by model.to(device) in train
    device = 'cpu'
    if in_arg.gpu:
        cuda = torch.cuda.is_available()
        if cuda:
            device = 'cuda'

    # TODO: define train function parameters
    pretrained_model = load_pretrained_model(in_arg.arch, in_arg.unfreeze_weights)
    pretrained_model.to(device)

    if in_arg.prior_checkpoint:
        load_checkpoint(in_arg.prior_checkpoint, device=device, pretrained_model=pretrained_model)

    # for name, param in pretrained_model.named_parameters():
    #     if param.requires_grad:
    #         print('Trainable Paramater Name: {}, Shape: {}'.format(name, param.data.shape))

    optimizer = optim.Adam(pretrained_model.get_optimizer_parameters(),
                           lr=in_arg.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

    criterion = nn.NLLLoss()

    trained_model = train(pretrained_model, criterion, optimizer, dataloaders, device, in_arg.epoch,
                          checkpoint_prefix,
                          in_arg.print_every)

    print('%.2f seconds taken for model training' % (time.time() - training_start))

    # TODO: test trained model on test dataset
    test_loader = dataloaders['test']

    test_accuracy = test_model(trained_model, test_loader, device)

    print('Test Accuracy: {}'.format(test_accuracy))

    metrics = trained_model.metrics
    print('{} --> {}'.format(in_arg, checkpoint_prefix))
    print('Final Metrics: Average Precision={}, F1 Score= {}'
          .format(metrics['average_precision'], metrics['f1_scores']))


if __name__ == "__main__":
    main()
