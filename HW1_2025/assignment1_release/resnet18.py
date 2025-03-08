'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
           self.shortcut = nn.Sequential(
               nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(planes)
           )

    def forward(self, x):
        # 1. Go through conv1, bn1, relu
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        # 2. Go through conv2, bn
        output = self.conv2(output)
        output = self.bn2(output)

        # 3. Combine with shortcut output, and go through relu
        output += self.shortcut(x)
        output = self.relu(output)

        return output


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, images):
        output = self.conv1(images)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.linear(output)
        return output

    def visualize(self, n_layer, logdir):
        """ Visualize the kernel in the desired directory """
        for name, module in self.named_modules():
            if not isinstance(module, torch.nn.Conv2d):
                continue

            # Get filters and bias from the module
            filters = module.weight.data
            bias = module.bias  # could be None

            print(f'Layer: {module}, Weights: {filters.shape}')

            # Normalize filter values to 0-1 using min-max normalization
            f_max, f_min = filters.max().item(), filters.min().item()
            normalised_filters = (filters - f_min) / (f_max - f_min)

            # Define the number of filters to visualize and number of channels per filter.
            n_filters = 6
            num_channels = filters.shape[1]  # e.g., 3 for RGB

            # Create a figure with rows for channels and columns for filters.
            plt.figure(figsize=(n_filters * 3, num_channels * 3))
            
            # Iterate over each channel (row) and each filter (column).
            for channel in range(num_channels):
                for i in range(n_filters):
                    ax = plt.subplot(num_channels, n_filters, channel * n_filters + i + 1)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Get the i-th filter; shape: (in_channels, kernel_height, kernel_width)
                    f = normalised_filters[i]
                    
                    # For the top row (channel 0), add descriptive text as the subplot title.
                    if channel == 0:
                        filter_shape = tuple(f.shape)
                        filter_bias = bias[i].item() if bias is not None else 'None'
                        ax.set_title(f'Filter {i}\nShape: {filter_shape}\nBias: {filter_bias}', fontsize=8)
                    
                    # Plot the corresponding channel of the filter.
                    plt.imshow(f[channel].cpu().numpy(), cmap='gray')
                    
            plt.tight_layout()
            plt.show()
            plt.close()
            break
