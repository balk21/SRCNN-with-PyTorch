from torch import nn

class SRCNN(nn.Module):
    """
        This is the model that defined in SRCNN paper, Dong et al.

        Basic network settings (9-1-5) are applied.
    """

    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        
        # Layer 1: Feature Extraction
        # 9x9 kernel, no padding
        # 33 -> 25
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=0)
        self.relu1 = nn.ReLU(inplace=True)

        # Layer 2: Non-Linear Matching
        # 1x1 kernel, no padding
        # 25 -> 25
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)

        # Layer 3: Reconstruct
        # 5x5 kernel, no padding
        # 25 -> 21
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=0)
        
        self._initialize_weights()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

    def _initialize_weights(self):
        """
            Initial weights are indicated as zero bias, 
            zero mean, 0.001 standard deviation in reference paper. 
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                nn.init.zeros_(m.bias.data)