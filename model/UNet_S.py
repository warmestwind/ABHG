import torch.nn as nn
import torch.nn.functional as F
import torch
import  matplotlib.pyplot as plt
num_pt = 4

def make_model(args):
    return Unet(args)

class Unet(nn.Module):
    def __init__(self, config):
        super(Unet, self).__init__()
        filter_config = (32, 64, 128, 256)
        self.depth = len(filter_config)
        self.drop_rate = config.drop_rate
        in_channels = config.in_channels
        out_channels = num_pt

        self.encoders = nn.ModuleList()
        self.decoders_mean = nn.ModuleList()
        self.decoders_var = nn.ModuleList()

        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (in_channels,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        self.bottom_conv = nn.Sequential(*[nn.Conv2d(256, 512, 3, 1, 1),
                           nn.BatchNorm2d(512),
                           nn.ReLU(), nn.Conv2d(512, 256, 3, 1, 1),
                           nn.BatchNorm2d(256),
                           nn.ReLU()])  # nn.ConvTranspose2d(512, 256, 3, 1, 1)

        for i in range(0, self.depth):
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i]))

            # decoder architecture
            self.decoders_mean.append(_Decoder(decoder_filter_config[i],
                                               decoder_filter_config[i + 1],
                                               decoder_n_layers[i]))



        self.classifier_mean = nn.Conv2d(filter_config[0], 32, 3, 1, 1)
        self.classifier_mean1 = nn.Conv2d(32, 16, 3, 1, 1)
        self.classifier_mean2 = nn.Conv2d(16, out_channels, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x
        feat_encoders = []
        # encoder path, keep track of pooling indices and features size
        for i in range(0, self.depth):
            feat_ori, (feat, ind), size = self.encoders[i](feat)
            feat_encoders.append(feat_ori)
            # if i == 1:
            #     feat = F.dropout(feat, p=self.drop_rate, training=self.training)
            indices.append(ind)
            unpool_sizes.append(size)

        feat = self.bottom_conv(feat)
        feat_mean = feat
        feat_bottom = feat


        # decoder path, upsampling with corresponding indices and size
        for i in range(0, self.depth):
            feat_mean = self.decoders_mean[i](feat_mean, feat_encoders[self.depth-i-1], indices[self.depth -1 - i], unpool_sizes[self.depth -1 - i])

            # feat_var = self.decoders_var[i](feat_var, indices[self.depth -1 - i], unpool_sizes[self.depth -1 - i])
            # if i == 0:
            #     feat_mean = F.dropout(feat_mean, p=self.drop_rate, training=True)
            #     feat_var = F.dropout(feat_var, p=self.drop_rate, training=True)

        output_mean = self.classifier_mean(feat_mean)
        output_mean1 = self.classifier_mean1(output_mean)
        output_mean2 = self.classifier_mean2(output_mean1)

        b, c, h, w = output_mean2.shape
        output_mean2 = F.softmax(output_mean2.view(b, c, -1), -1).view(b, c, h, w)


        results = {'mean': output_mean2, 'var': 0, 'feat':   feat_encoders[0], 'bottom': feat_bottom}
        return results


class _Encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2):
        """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            n_in_feat (int): number of input features
            n_out_feat (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
        super(_Encoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU()]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU()]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return output, F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class _Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """

    def __init__(self, n_in_feat, n_out_feat, n_blocks=2):
        super(_Decoder, self).__init__()

        self.up_conv = nn.ConvTranspose2d(n_in_feat, n_in_feat, 3, 2, 1, 1)

        layers = [nn.Conv2d(2*n_in_feat, n_in_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_in_feat),
                  nn.ReLU()]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU()]

        self.features = nn.Sequential(*layers)

    def forward(self, x, x_e, indices, size):
        x = torch.cat([x_e, self.up_conv(x)], 1)
        return self.features(x)


if __name__ == '__main__':
    import torchinfo
    from config import get_config
    BUNET = Unet(get_config())

    batch_size = 2
    torchinfo.summary(BUNET, input_size=(batch_size, 1, 256, 256))
