import numpy
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as f
from torch import nn


class ResNet(pl.LightningModule):
    def __init__(self, out_dim=11, num_of_layers_to_freeze=9):
        super(ResNet, self).__init__()
        self.model = self.create_model(num_of_layers_to_freeze)

        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, out_features=out_dim)
        self.lr = 1e-3
        self.example_input_array = torch.FloatTensor(numpy.ndarray(shape=(64, 3, 640, 640)))

    def create_model(self, num_of_layers_to_freeze: int) -> nn.Module:
        model = timm.create_model('resnet200d', pretrained=True)

        model.requires_grad = True
        counter = 0
        for child in model.children():
            if counter < num_of_layers_to_freeze:
                for param in child.parameters():
                    param.requires_grad = False
                child.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

            counter += 1

        return model

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = f.binary_cross_entropy_with_logits(y_hat, y)

        self.log('BCEWL', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = float(f.binary_cross_entropy_with_logits(y_hat, y))

        self.log('val_loss', loss)
        return loss
