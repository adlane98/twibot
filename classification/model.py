import torchvision
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, input_size=128, embedding_size=120, pretrained=False):
        nn.Module.__init__(self)
        self.model = torchvision.models.alexnet(pretrained=pretrained)
        self.model.classifier = nn.Linear(in_features=9216, out_features=embedding_size, bias=True)
        self.softmax = nn.Softmax()

    def forward(self, x):
        output = self.model(x)
        return self.softmax(output)