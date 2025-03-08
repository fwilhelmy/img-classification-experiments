from torch import nn
import torch
import math
import matplotlib.pyplot as plt
import os
import math
import torch
import matplotlib.pyplot as plt

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        :param x: image tensor of shape [batch, channels, img_size, img_size]
        :return out: [batch. num_patches, embed_dim]
        """
        _, _, H, W = x.shape
        assert H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size})."
        assert W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(
            self,
            in_features,
            hidden_features,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super(Mlp, self).__init__()
        out_features = in_features
        hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0),
            activation='gelu', drop=0., drop_path=0.):
        super(MixerBlock, self).__init__()
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]
        tokens_dim, channels_dim = int(mlp_ratio[0] * dim), int(mlp_ratio[1] * dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # norm1 used with mlp_tokens
        self.mlp_tokens = Mlp(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6) # norm2 used with mlp_channels
        self.mlp_channels = Mlp(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Token mixing
        residual = x
        output = self.norm1(x)
        output = output.transpose(1, 2) # [batch, embed_dim, num_patches]
        output = self.mlp_tokens(output) # Apply MLP on token dimension
        output = output.transpose(1, 2) # [batch, num_patches, embed_dim]
        output = output + residual # Skip-connection

        # Channel mixing
        residual = output
        output = self.norm2(output)
        output = self.mlp_channels(output) # Apply MLP on channel dimension
        output = output + residual # Skip-connection

        return output
    

class MLPMixer(nn.Module):
    def __init__(self, num_classes, img_size, patch_size, embed_dim, num_blocks, 
                 drop_rate=0., activation='gelu'):
        super(MLPMixer, self).__init__()
        self.patchemb = PatchEmbed(img_size=img_size, 
                                   patch_size=patch_size, 
                                   in_chans=3,
                                   embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[
            MixerBlock(
                dim=embed_dim, seq_len=self.patchemb.num_patches, 
                activation=activation, drop=drop_rate)
            for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        """ MLPMixer forward
        :param images: [batch, 3, img_size, img_size]
        """

        # Step 1 : Convert images to a sequence of patch embeddings
        output = self.patchemb(images) # [batch, num_patches, embed_dim]
        # Step 2 : Go through the mixer blocks
        output = self.blocks(output) # [batch, num_patches, embed_dim]
        # Step 3 : Layer norm
        output = self.norm(output) # [batch, num_patches, embed_dim]
        # Step 4 : Global average pooling over the patch (token) dimension
        output = output.mean(dim=1) # [batch, embed_dim]
        # Step 5 : Final classification head
        output = self.head(output) # [batch, num_classes]

        return output

    def visualize(self, block_index=0):
        """
        Visualize the weights (only the first linear layer) of the token-mixing MLP 
        in the specified MixerBlock (by index). The weights are reshaped into a 
        (sqrt(num_patches), sqrt(num_patches)) grid for visualization and displayed.
        
        :param block_index: Index of the MixerBlock in self.blocks to visualize.
        """
        # 1. Validate block_index
        if block_index < 0 or block_index >= len(self.blocks):
            raise IndexError(f"block_index={block_index} is out of range for {len(self.blocks)} MixerBlocks.")

        # 2. Get the specified MixerBlock
        mixer_block = self.blocks[block_index]

        # 3. Extract the token-mixing MLP from that block
        token_mlp = mixer_block.mlp_tokens

        # 4. Get the weights from the first linear layer (fc1) in the token-mixing MLP
        #    shape: (hidden_features, seq_len) => each row is one hidden unit, each column is one patch
        weight = token_mlp.fc1.weight.data.cpu().numpy()

        # 5. Figure out how many patches we have (seq_len) and verify shape
        seq_len = self.patchemb.num_patches  # e.g., 196 for a 14x14 patch grid
        if weight.shape[1] != seq_len:
            raise ValueError(
                f"Expected {seq_len} columns in fc1.weight, but got {weight.shape[1]}."
            )

        # 6. Compute the side of the patch grid (e.g., 14 for 14x14)
        side = int(math.sqrt(seq_len))
        if side * side != seq_len:
            raise ValueError(
                f"Number of patches {seq_len} is not a perfect square, cannot reshape to a square."
            )

        # 7. Determine how many hidden units we have and set up a grid of subplots
        tokens_dim = weight.shape[0]
        grid_size = int(math.ceil(math.sqrt(tokens_dim)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
        axes = axes.flatten()

        # 8. Plot each hidden unit as a heatmap
        for i in range(tokens_dim):
            row_weights = weight[i]  # shape: (seq_len,)
            patch_map = row_weights.reshape(side, side)
            ax = axes[i]
            ax.imshow(patch_map, cmap='bwr')  # 'bwr' highlights positive vs. negative
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide any unused subplots if tokens_dim is not a perfect square
        for i in range(tokens_dim, grid_size * grid_size):
            axes[i].axis('off')

        fig.suptitle(
            f"Token-mixing MLP (fc1) weights in MixerBlock {block_index}",
            fontsize=14
        )
        plt.tight_layout()

        # 9. Display the figure
        plt.show()
