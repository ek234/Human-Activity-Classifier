import torch.nn as nn

class CNN3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 8, kernel_size=(10,1,1), stride=(3,1,1), padding=0)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(10,1,1), stride=(5,1,1), padding=0)
        # self.conv3 = nn.Conv3d(16, 32, kernel_size=(5,1,1), stride=(2,1,1), padding=0)
        self.conv3 = None
        self.conv4 = nn.Conv3d(16, 32, kernel_size=(5,25,2), stride=(2,1,1), padding=0)
        # self.conv5 = nn.Conv3d(64, 64, kernel_size=(1,1,1), stride=1, padding=0)
        self.conv5 = None
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(224, 90)
        self.bn5 = nn.BatchNorm1d(90)
        self.fc2 = nn.Linear(90, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.float()
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            if conv is not None:
                x = self.relu(conv(x))
            # print(f"shape: {x.shape}")
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.bn5(x)
        x = self.fc2(x)
        x = x.squeeze(1)
        x = self.softmax(x)
        return x
