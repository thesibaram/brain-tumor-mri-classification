import torch.nn as nn

def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size= 3, padding= 1, bias= False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        nn.Conv2d(cout, cout, kernel_size= 3, padding= 1, bias= False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride= 2)
    )

class BTCCnn(nn.Module):
    def __init__(self, num_classes= 4, in_ch= 1):
        super(BTCCnn, self).__init__()
        self.features= nn.Sequential(
            conv_block(in_ch, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512)
        )

        self.pool= nn.AdaptiveAvgPool2d(1)

        self.classifier= nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x= self.features(x)
        x= self.pool(x)
        x= self.classifier(x)
        return x

model = BTCCnn(in_ch=1)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f"Parameters: {n_params}")
