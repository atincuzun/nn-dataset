

import torch
import torch.nn as nn
import torch.optim as optim

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}
class DPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DPNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

class DPN131(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, num_blocks=3, growth_rate=32):
        super(DPN131, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(growth_rate)
        self.relu = nn.ReLU(inplace=True)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(DPNBlock(growth_rate, growth_rate))
        
        # Adaptive average pooling to ensure a fixed feature size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Output size (1, 1)
        
        # Adjust fc layer's in_features to match the pooled feature size
        self.fc = nn.Linear(growth_rate, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prms, model_class=DPN131):
        super(Net, self).__init__()

        # Extract in_shape and out_shape values
        self.batch = in_shape[0]
        self.channel_number = in_shape[1]
        self.image_size = in_shape[2]  # Assuming square images (height == width)
        self.class_number = out_shape[0]  # Number of classes for classification

        # Use extracted values in the model
        self.model = model_class(self.channel_number, self.class_number, num_blocks=3, growth_rate=32)

        # Hyperparameters
        self.learning_rate = prms['lr']
        self.momentum = prms['momentum']
        self.dropout = prms['dropout']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, device, prms):
        self.device = device
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()