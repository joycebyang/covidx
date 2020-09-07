import argparse
import os
import random
import time

import models
import numpy as np
import torch
import torchvision
from PIL import Image
from dataset_generator import DatasetGenerator
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_size = 224
image_resize = 256
class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to created and defined these 3 command line arguments. If
    the user fails to provide some or all of the 3 arguments, then the default
    values are used for the missing arguments.
    """
    # create Parse using Argument Parser
    parser = argparse.ArgumentParser()

    # create command line arguments using add argument from the Argument Parser method
    available_models = [m for m in dir(models) if 'Net' in m and 'Model' not in m]
    parser.add_argument('--arch', choices=available_models, type=str, default='ResNet50',
                        help='string containing used architecture')

    parser.add_argument('--epoch', type=int, default=2, help='number of epochs')

    parser.add_argument('--gpu', action='store_true', help='use gpu if available')

    parser.add_argument('--batch', type=int, default=64, help='number of images per batch')

    parser.add_argument('--image_dir', type=str, default='database', help='image directory')

    parser.add_argument('--dataset_dir', type=str, default='dataset', help='csv label file directory')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='used in save checkpoint path')

    parser.add_argument('--prior_checkpoint', type=str, required=False,
                        help='existing checkpoint from prior training to bootstrap from')

    parser.add_argument('--learning_rate', type=float, required=True, help='step size in optimizer')

    return parser.parse_args()


print(torch.__version__)
print(torchvision.__version__)

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# load the pretrained model corresponding to the specified architecture
def load_pretrained_model(arch):
    model_func = getattr(models, arch)
    model = model_func()
    model.arch = arch

    return model


# used during training: disable gradients, feedforward, calculate test loss
def validate(model, valid_dataloader, loss_func, device):
    # track accuracy and loss
    test_loss = 0

    with torch.no_grad():  # deactivates requires_grad flag, disables tracking of gradients
        for images, labels in valid_dataloader:  # iterate over images and labels in valid dataset
            images, labels = images.to(device), labels.to(device)  # move a tensor to a device
            ps = model.forward(images)  # probabilities for each label
            test_loss += loss_func(ps, labels)  # take all values

    return test_loss


# used at the end of each epoch: get true labels and predicted probabilities, call calc_roc_metrics
def test_model(model, test_loader, device):
    # track accuracy, move to device, switch on eval mode
    model.to(device)
    model.eval()
    pred_out = torch.FloatTensor().to(device)
    labels_out = torch.FloatTensor().to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # move a tensor to a device
            labels_out = torch.cat((labels_out, labels), 0)
            ps = model.forward(images)
            pred_out = torch.cat((pred_out, ps), 0)

    metrics = calc_roc_metrics(pred_out, labels_out)
    return metrics


# evaluation metric: roc and auc metrics
def calc_roc_metrics(pred_out, labels_out):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # https://github.com/rachellea/glassboxmedicine/blob/master/2020-07-14-AUROC-AP/main.py
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

    y_true = labels_out.cpu().numpy()
    y_score = pred_out.cpu().numpy()

    for idx, clz in enumerate(class_names):
        fpr[clz], tpr[clz], _ = roc_curve(y_true[:, idx], y_score[:, idx])
        roc_auc[clz] = auc(fpr[clz], tpr[clz])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[clz] for clz in class_names]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for clz in class_names:
        mean_tpr += np.interp(all_fpr, fpr[clz], tpr[clz])

    # Finally average it and compute AUC
    mean_tpr /= len(class_names)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    average_precision = average_precision_score(y_true, y_score)
    roc_auc_mean = roc_auc_score(y_true, y_score)

    return {
        'average_precision_score': average_precision_score(y_true, y_score),
        'roc_auc_score': roc_auc_score(y_true, y_score),
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }


# save the checkpoint
def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch):
    checkpoint = {
        'arch': model.arch,
        'metrics': model.metrics,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_path)
    print('Saved the trained model: %s' % checkpoint_path)


def load_checkpoint(ckp_path, pretrained_model, optimizer, scheduler, device, arch):
    model_checkpoint = torch.load(ckp_path, map_location=torch.device(device))

    if model_checkpoint['arch'] != arch:
        print('Mismatched arch: Model Checkpoint=%s vs Arch=%s' % (model_checkpoint['arch'], arch))
        return False

    state_dict = model_checkpoint['state_dict']
    saved_optimizer = model_checkpoint['optimizer']
    saved_scheduler = model_checkpoint['scheduler']

    # check the state dictionary matches
    old_state_dict = pretrained_model.state_dict()
    for k in state_dict:
        if k not in old_state_dict:
            print('Unexpected key %s in state_dict' % k)
    for k in old_state_dict:
        if k not in state_dict:
            print('Missing key %s in state_dict' % k)

    # TBD: load checkpoint into pretrained_model
    pretrained_model.load_state_dict(state_dict)
    optimizer.load_state_dict(saved_optimizer)
    scheduler.load_state_dict(saved_scheduler)

    print('Model checkpoint successfully loaded from %s' % ckp_path)

    return True


# train the model
def train(model, loss_func, optimizer, scheduler, dataloaders, device, epochs, checkpoint_prefix, print_every=100):
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
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            ps = model.forward(images)
            loss = loss_func(ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            steps += 1

            if steps % print_every == 0:
                model.eval()
                valid_loss = validate(model, valid_loader, loss_func, device)
                model.train()
                batch_time = time.time() - batch_start
                print(
                    "Epoch: {}/{}..".format(e + 1, epochs),
                    "Step: {}..".format(steps),
                    "Training Loss: {:.5f}..".format(running_loss / len(train_loader)),
                    "Test Loss: {:.5f}..".format(valid_loss / len(valid_loader)),
                    "Batch Time: {:.1f}, avg: {:.3f}".format(batch_time, batch_time / steps)
                )

        model.eval()
        valid_loss = validate(model, valid_loader, loss_func, device)
        model.train()

        scheduler.step(valid_loss / len(valid_loader))

        model.eval()
        metrics = test_model(model, test_loader, device)
        roc_aoc_mean = metrics['roc_auc_score']
        average_precision_mean = metrics['average_precision_score']
        model.train()
        epoch_time = time.time() - epoch_start

        if roc_aoc_mean > max_acc:
            max_acc = roc_aoc_mean
            model.metrics = metrics
            save_checkpoint('%s.pth.tar' % checkpoint_prefix, model, optimizer, scheduler, e + 1)
            print(
                "Epoch: {}/{} [save] Epoch Time={:.1f}..".format(e + 1, epochs, epoch_time),
                "Average Precision: {:.3f}..".format(average_precision_mean),
                "ROC: {:.3f}..".format(roc_aoc_mean),
                "Batch Time: {:.1f}, avg: {:.3f}".format(batch_time, batch_time / steps)
            )
        else:
            print(
                "Epoch: {}/{} [----] Epoch Time={:.1f}..".format(e + 1, epochs, epoch_time),
                "Average Precision: {:.3f}..".format(average_precision_mean),
                "ROC: {:.3f}..".format(roc_aoc_mean),
                "Batch Time: {:.1f}, avg: {:.3f}".format(batch_time, batch_time / steps)
            )

    return model


def main():
    in_arg = get_input_args()

    training_start = time.time()
    checkpoint_prefix = os.path.join(in_arg.checkpoint_dir, '%s-%d' % (in_arg.arch, training_start))
    print('{} --> {}'.format(in_arg, checkpoint_prefix))
    batch_size = in_arg.batch

    image_dir = in_arg.image_dir
    train_csv_file = in_arg.dataset_dir + '/train.txt'
    test_csv_file = in_arg.dataset_dir + '/test.txt'
    valid_csv_file = in_arg.dataset_dir + '/valid.txt'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomOrder([
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15, resample=Image.BILINEAR),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(image_resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transforms = valid_transforms

    # TODO: Load the Datasets using Dataset Generator
    train_dataset = DatasetGenerator(train_csv_file, image_dir, transform=train_transforms)
    test_dataset = DatasetGenerator(test_csv_file, image_dir, transform=test_transforms)
    valid_dataset = DatasetGenerator(valid_csv_file, image_dir, transform=valid_transforms)

    # TODO: Using the image datasets and the transforms, define the dataloaders
    dataloaders = {"train": DataLoader(train_dataset, batch_size=batch_size, num_workers=24, shuffle=True),
                   "test": DataLoader(test_dataset, batch_size=batch_size, num_workers=24, shuffle=False),
                   "valid": DataLoader(valid_dataset, batch_size=batch_size, num_workers=24, shuffle=False)}

    device = 'cpu'
    if in_arg.gpu:
        cuda = torch.cuda.is_available()
        if cuda:
            # make sure if GPU is available, we can use it by model.to(device) in train
            device = 'cuda'

    pretrained_model = load_pretrained_model(in_arg.arch)
    pretrained_model.to(device)
    optimizer = optim.Adam(pretrained_model.parameters(),
                           lr=in_arg.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    criterion = nn.BCELoss(reduction='mean')
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')

    if in_arg.prior_checkpoint:
        checkpoint_path = os.path.join(in_arg.checkpoint_dir, in_arg.prior_checkpoint)
        checkpoint_status = load_checkpoint(checkpoint_path, pretrained_model, optimizer, scheduler, device,
                                            in_arg.arch)
        if not checkpoint_status:
            return

    trained_model = train(pretrained_model, criterion, optimizer, scheduler, dataloaders, device, in_arg.epoch,
                          checkpoint_prefix, print_every=300)
    metrics = trained_model.metrics
    print('%.2f seconds taken for model training' % (time.time() - training_start))
    print('Metrics: average_precision_score={}, roc_auc_score={}'.format(metrics['average_precision_score'],
                                                                         metrics['roc_auc_score']))


if __name__ == "__main__":
    main()
