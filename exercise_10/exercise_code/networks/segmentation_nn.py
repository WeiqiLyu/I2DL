"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl



class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO - Train Your Model                                              #
        ########################################################################
        import torchvision.models as models
        
        self.mobilenet = models.mobilenet_v2(pretrained=True).features # last layer (1280, 1, 1)
        # self.alexnet = models.alexnet(pretrained=True) # last layer softmax for 1000
        # resnet = models.resnet34(pretrained=True)
        
        for parameter in self.mobilenet.parameters():
            parameter.requires_grad = False
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), # 1280 x 16 x 16
            nn.Conv2d(1280, 320, 1, stride=1), # 320 x 7 x 7
            nn.Upsample(scale_factor=3, mode='bilinear'), # 320 x 48 x 48
            nn.Conv2d(320, 128, 1, stride=1), # 128 x 48 x 48
            nn.Upsample(scale_factor=5, mode='bilinear'), # 128 x 240 x 240
            nn.Conv2d(128, num_classes, 1, stride=1), # 23 x 240 x 240
        )            

      
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x_encoded = self.mobilenet(x)
        x = self.decoder(x_encoded)
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
