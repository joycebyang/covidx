import os
import time

import torch
from dataset_generator import DatasetGenerator
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from train import (get_input_args, mean, std,
                   get_weighted_dataloader, get_balanced_dataloader,
                   load_pretrained_model, load_checkpoint, train, test_model)


def main():
    in_arg = get_input_args()

    training_start = time.time()
    checkpoint_prefix = os.path.join(in_arg.checkpoint_dir, '%s-%d' % (in_arg.arch, training_start))
    print('{} --> {}'.format(in_arg, checkpoint_prefix))

    train_csv_file = 'newtrain_split.txt'
    test_csv_file = 'test_split.txt'
    valid_csv_file = 'valid_split.txt'

    # TODO: Define your transforms for the training, validation, and testing sets

    valid_transforms = transforms.Compose([
        transforms.Resize(int(in_arg.image_size * (256 / 224)), interpolation=2),
        transforms.CenterCrop(in_arg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transforms = valid_transforms
    train_transforms = valid_transforms

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
