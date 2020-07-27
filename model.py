import torch.nn as nn
from torchvision.models import resnet50
import torch
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    resnet = resnet50(pretrained=True)
    module = list(resnet.children())[:-2]
    self.backbone = nn.Sequential(*module)
    self.avg = nn.AdaptiveAvgPool2d((1,1))
    '''
    self.classifier = nn.Sequential(nn.Dropout(),
                                    nn.Linear(2048, 512),
                                    nn.ReLU(True),
                                    nn.Linear(512, 10))
                                      '''
    self.drop = nn.Dropout()
    self.liner1 = nn.Linear(2048, 512)
    self.relu = nn.ReLU(True)
    self.liner2 = nn.Linear(512,10)

  def forward(self, x):
    x = self.backbone(x)
    x = x.view(x.size(0),-1)
    x = self.drop(x)
    x = self.liner1(x)
    x = self.relu(x)
    out = self.liner2(x)
    #out = self.classifier(x)
    return out

if __name__ == '__main__':
    x = torch.randn(3,3,224,224)
    model = Net()
    print(model)
    print(model(x).size())
