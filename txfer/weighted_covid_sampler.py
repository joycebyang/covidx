import numpy as np
from torch.utils.data.sampler import Sampler


class WeightedCovidSampler(Sampler):
    def __init__(self, train_noncovid_list, train_covid_list, batch_size, covid_percent):
        self.batch_size = batch_size
        self.train_noncovid_list = np.array(train_noncovid_list)
        self.train_covid_list = np.array(train_covid_list)
        self.covid_percent = covid_percent
        self.covid_size = max(int(batch_size * covid_percent), 1)
        self.size = len(self.train_noncovid_list) + len(self.train_covid_list)

    def __do_batch__(self):
        covid_inds = np.random.choice(self.train_covid_list, size=self.covid_size, replace=False)
        noncovid_inds = np.random.choice(self.train_noncovid_list, size=(self.batch_size - self.covid_size),
                                         replace=False)
        batch_inds = np.concatenate((covid_inds, noncovid_inds), axis=None)
        np.random.shuffle(batch_inds)

        return batch_inds

    def __iter__(self):
        for batch_idx in range(0, self.size, self.batch_size):
            yield self.__do_batch__()

    def __len__(self):
        return self.size
