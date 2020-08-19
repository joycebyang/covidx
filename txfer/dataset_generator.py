import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DatasetGenerator(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        """

        Args:
            csv_file: Path to train_split.txt or test_split.txt file
            root_dir: image root directory (.jpeg)
            transform: transformations on image
        """
        self.csv_df = pd.read_csv(csv_file, header=None, delim_whitespace=True)
        # https://stackoverflow.com/questions/15772009/shuffling-permutating-a-dataframe-in-pandas/35784666
        # Sampling randomizes, so just sample the entire data frame.
        self.csv_df = self.csv_df.sample(frac=1)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, index):
        image_name = self.csv_df.iloc[index, 1]
        image_path = os.path.join(self.root_dir, image_name)

        image_data = Image.open(image_path).convert("RGB")
        if self.transform:
            image_data = self.transform(image_data)

        image_label = self.csv_df.iloc[index, 2]

        return image_data, image_label
