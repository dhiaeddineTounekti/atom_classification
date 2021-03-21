from typing import List, Any

import matplotlib.pyplot as plt
import numpy
import pytorch_lightning as pl
import pytorch_lightning.metrics as pl_met
import timm
import torch
import torch.nn.functional as f
from sklearn.metrics import confusion_matrix
from torch import nn


class ResNet(pl.LightningModule):
    def __init__(self, out_dim=12, num_of_layers_to_freeze=7, learning_rate=1e-3):
        super(ResNet, self).__init__()

        self.model = self.create_model(num_of_layers_to_freeze)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, out_features=out_dim)

        self.lr = learning_rate
        self.example_input_array = torch.FloatTensor(numpy.ndarray(shape=(64, 1, 100, 100)))

        self.classes = [(0, 1.0), (0, 3.0), (0, 6.0), (0, 10.0), (0, 20.0), (0, 30.0), (1, 1.0), (1, 3.0), (1, 6.0),
                        (1, 10.0), (1, 20.0), (1, 30.0)]
        self.class_weight = [5, 1, 10, 1, 10, 1, 1, 10, 1, 10, 1, 10]

    def create_model(self, num_of_layers_to_freeze: int) -> nn.Module:
        model = timm.create_model('resnet50', pretrained=False, in_chans=1)

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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = f.binary_cross_entropy_with_logits(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, idx = batch[0], batch[1], batch[2]
        y_hat = self(x)
        preds = f.sigmoid(y_hat)
        return {
            'predictions': preds,
            'targets': y,
            'image_index': idx
        }

    def test_epoch_end(self, outputs: List[Any]) -> None:
        predictions = torch.cat([x['predictions'] for x in outputs], dim=0)
        targets = torch.cat([x['targets'] for x in outputs], dim=0)
        image_indexes = numpy.concatenate([x['image_index'] for x in outputs])

        AUCROC = pl_met.classification.AUROC(num_classes=predictions.size()[1], average=None)
        AUCROC_mean = pl_met.classification.AUROC(num_classes=predictions.size()[1])

        auc = AUCROC(preds=predictions, target=torch.tensor(targets.clone().detach(), dtype=torch.int))
        auc_mean = AUCROC_mean(preds=predictions, target=torch.tensor(targets.clone().detach(), dtype=torch.int))
        auc_dict = {self.classes[idx]: auc[idx] for idx in range(len(self.classes))}
        auc_dict['mean_auc'] = auc_mean

        confusion_mat = confusion_matrix(targets, predictions, cmap=plt.cm.get_cmap('Blues'))
        self.logger.experiment.add_graph('Confusion matrix', confusion_mat, self.current_epoch)
        self.logger.experiment.add_scalars('AUCROC', auc_dict, self.current_epoch)

        submission = [[image_id] + pred for (image_id, pred) in zip(image_indexes, predictions.detach().cpu().tolist())]

    def training_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        self.log('val_loss', avg_loss, prog_bar=True)
