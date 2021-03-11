import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as f
from torch import nn


class ResNet(pl.LightningModule):
    def __init__(self, out_dim=11, num_of_layers_to_freeze=8):
        super(ResNet, self).__init__()
        self.model = timm.create_model('resnet200d', pretrained=True)

        counter = 0
        for child in self.model.children():
            counter += 1
            if counter > num_of_layers_to_freeze:
                break

            for param in child.parameters():
                param.requires_grad = False

        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, out_features=out_dim)
        self.lr = 1e-3

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
