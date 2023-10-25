# Create data modules
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from src.cancer_sim.data_utils import process_data, data_to_torch_tensor, intensity_and_treatment_scale


class CancerDataloader(pl.LightningDataModule):
    def __init__(self, pickle_map, test_set='f', # 'f', 'cf'
                 device='cpu',
                 batch_size=512, num_workers=0):
        super().__init__()
        self.test_set = test_set
        self.batch_size = batch_size
        self.pickle_map = pickle_map
        self.device = device
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def prepare_dataloader(self, data):
        data_X, data_tr, data_y, data_ae, data_tr_tau, _ = data_to_torch_tensor(data)

        data_concat = torch.cat((data_X, data_tr), dim=-1)
        ######### interpolate data in data-dependent CDE submodule
        dataset = torch.utils.data.TensorDataset(data_concat.to(dtype=torch.float32),
                                                 data_y.to(dtype=torch.float32),
                                                 data_tr_tau.to(dtype=torch.float32),
                                                 data_ae.to(dtype=torch.float32))

        return dataset

    def setup(self, stage):
        training_processed, validation_processed, test_f_processed, test_cf_processed, scale = process_data(self.pickle_map)

        training_processed = intensity_and_treatment_scale(training_processed, scale)
        validation_processed = intensity_and_treatment_scale(validation_processed, scale)
        test_f_processed = intensity_and_treatment_scale(test_f_processed, scale)
        test_cf_processed = intensity_and_treatment_scale(test_cf_processed, scale)

        self.train_data = self.prepare_dataloader(training_processed)
        self.val_data = self.prepare_dataloader(validation_processed)
        self.test_f_data = self.prepare_dataloader(test_f_processed)
        self.test_cf_data = self.prepare_dataloader(test_cf_processed)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=torch.Generator(device=self.device),
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            generator=torch.Generator(device=self.device),
        )
    def test_dataloader(self):
        if self.test_set == 'f':
            return DataLoader(
                self.test_f_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                generator=torch.Generator(device=self.device),
            )
        elif self.test_set == 'cf':
            return DataLoader(
                self.test_cf_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                generator=torch.Generator(device=self.device),
            )
        else:
            raise ValueError('Need to specify whether to test on factual or counterfactual test data')
    def predict_dataloader(self):
        if self.test_set == 'f':
            return DataLoader(
                self.test_f_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                generator=torch.Generator(device=self.device),
            )
        elif self.test_set == 'cf':
            return DataLoader(
                self.test_cf_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                generator=torch.Generator(device=self.device),
            )
        else:
            raise ValueError('Need to specify whether to test on factual or counterfactual test data')









