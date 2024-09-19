import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, queries, key, values):
        batch_size, _, depth, height, width = queries.size()
        queries = queries.permute(2, 0, 1, 3, 4).reshape(depth, -1, queries.size(-1))
        key = key.permute(2, 0, 1, 3, 4).reshape(depth, -1, key.size(-1))
        values = values.permute(2, 0, 1, 3, 4).reshape(depth, -1, values.size(-1))
        output, _ = self.attention(queries, key, values)

        output = output.view(depth, batch_size, -1, height, width)
        #print('output shape: ', output.shape)
        output = output.permute(1, 2, 0, 3, 4)
        # print('output shape: ', output.shape)
        return output

class Attention3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention3d, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Conv3d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm3d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv3d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv3d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv3d(attention_channel, kernel_size * kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv3d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)
    

def conv_block(in_channels, out_channels, use_dropout=False, dropout_rate=0.1):
    layers = [
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    ]
    
    if use_dropout:
        layers.append(nn.Dropout3d(dropout_rate))

    return nn.Sequential(*layers)


class UNet3D_FeatureExtractor(nn.Module):
    def __init__(self, in_channels, use_dropout=False, dropout_rate=0.1):
        super(UNet3D_FeatureExtractor, self).__init__()
        self.encoder1 = conv_block(in_channels, 32, use_dropout, dropout_rate)
        self.encoder2 = conv_block(32, 64, use_dropout, dropout_rate)
        self.encoder3 = conv_block(64, 128, use_dropout, dropout_rate)
        self.middle = conv_block(128, 256, use_dropout, dropout_rate)

        self.pool = nn.MaxPool3d(2)
        
    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool(x1))
        x3 = self.encoder3(self.pool(x2))
        x4 = self.middle(self.pool(x3))

        return x4, x3, x2, x1

class Attn_UNet3D_FeatureExtractor(nn.Module):
    def __init__(self, in_channels, use_dropout=False, dropout_rate=0.1):
        super(Attn_UNet3D_FeatureExtractor, self).__init__()
        self.encoder1 = conv_block(in_channels, 32, use_dropout, dropout_rate)
        self.encoder = conv_block(1, 32, use_dropout, dropout_rate)
        self.encoder2 = conv_block(32, 64, use_dropout, dropout_rate)
        self.encoder3 = conv_block(64, 128, use_dropout, dropout_rate)
        self.middle = conv_block(128, 256, use_dropout, dropout_rate)

        self.pool = nn.MaxPool3d(2)
        
    def forward(self, ct, bld, ctv, rct):
        x1_ct = self.encoder1(ct)
        x2_ct = self.encoder2(self.pool(x1_ct))
        x3_ct = self.encoder3(self.pool(x2_ct))
        x4_ct = self.middle(self.pool(x3_ct))

        x1_rct = self.encoder(rct)
        
        x2_bld = self.encoder(bld)
        x2_bld = self.encoder2(self.pool(x2_bld))

        x3_ctv = self.encoder(ctv)
        x3_ctv = self.encoder2(self.pool(x3_ctv))
        x3_ctv = self.encoder3(self.pool(x3_ctv))

        return x4_ct, x3_ct, x2_ct, x1_ct, x3_ctv, x2_bld, x1_rct

class AttnLabelPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False, dropout_rate=0.1):
        super(AttnLabelPredictor, self).__init__()
        self.decoder1 = conv_block(256, 128, use_dropout, dropout_rate)
        self.decoder2 = conv_block(128, 64, use_dropout, dropout_rate)
        self.decoder3 = conv_block(64, 32, use_dropout, dropout_rate)
        self.final = nn.Conv3d(32, out_channels, kernel_size=1)

        self.upconv1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)

        self.cross_attention1 = CrossAttention(8, 4)
        self.cross_attention2 = CrossAttention(16, 4)
        self.cross_attention3 = CrossAttention(32, 4)

    def forward(self, x4_ct, x3_ct, x2_ct, x1_ct, x3_ctv, x2_bld, x1_rct):
        
        x = self.upconv1(x4_ct)           # [1, 256, 60, 60, 38]
        x3 = x3_ct + self.cross_attention1(x3_ct, x3_ctv, x3_ctv)
        x = torch.cat((x, x3), dim=1)  # [1, 256, 60, 60, 38]+[1, 128, 60, 60, 38]
        x = self.decoder1(x)           # [1, 128, 60, 60, 38]

        x = self.upconv2(x)
        x2 = x2_ct + self.cross_attention2(x2_ct, x2_bld, x2_bld)
        x = torch.cat((x, x2), dim=1)
        x = self.decoder2(x)

        x = self.upconv3(x)
        x1 = x1_ct + self.cross_attention3(x1_ct, x1_rct, x1_rct)
        x = torch.cat((x, x1), dim=1)
        x = self.decoder3(x)

        x = self.final(x)
        return x
