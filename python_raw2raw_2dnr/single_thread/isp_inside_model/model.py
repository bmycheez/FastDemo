import torch
import torch.nn as nn
import torch.nn.functional as F

# raw2raw-torch-2dnr
class Unet4to4(nn.Module):
    def __init__(self, k=8, bias=False):
        super().__init__()
        # self.quant = torch.quantization.QuantStub()
        self.conv1_1 = nn.Conv2d(4, 1 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv1_2 = nn.Conv2d(1 * k, 1 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv2_1 = nn.Conv2d(1 * k, 2 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv2_2 = nn.Conv2d(2 * k, 2 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv3_1 = nn.Conv2d(2 * k, 4 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv3_2 = nn.Conv2d(4 * k, 4 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv4_1 = nn.Conv2d(4 * k, 8 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv4_2 = nn.Conv2d(8 * k, 8 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv5_1 = nn.Conv2d(8 * k, 16 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv5_2 = nn.Conv2d(16 * k, 16 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.upv6 = nn.ConvTranspose2d(16 * k, 8 * k, kernel_size=(2, 2), stride=(2, 2))
        self.conv6_1 = nn.Conv2d(16 * k, 8 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv6_2 = nn.Conv2d(8 * k, 8 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.upv7 = nn.ConvTranspose2d(8 * k, 4 * k, kernel_size=(2, 2), stride=(2, 2))
        self.conv7_1 = nn.Conv2d(8 * k, 4 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv7_2 = nn.Conv2d(4 * k, 4 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.upv8 = nn.ConvTranspose2d(4 * k, 2 * k, kernel_size=(2, 2), stride=(2, 2))
        self.conv8_1 = nn.Conv2d(4 * k, 2 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv8_2 = nn.Conv2d(2 * k, 2 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.upv9 = nn.ConvTranspose2d(2 * k, 1 * k, kernel_size=(2, 2), stride=(2, 2))
        self.conv9_1 = nn.Conv2d(2 * k, 1 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv9_2 = nn.Conv2d(1 * k, 1 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv10_1 = nn.Conv2d(1 * k, 4, kernel_size=(1, 1), stride=(1, 1), bias=bias)
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.l_relu = nn.LeakyReLU(0.2)
        # self.dequant = torch.quantization.DeQuantStub()

        self.sum_1 = torch.nn.quantized.FloatFunctional()
        self.sum_2 = torch.nn.quantized.FloatFunctional()
        self.sum_3 = torch.nn.quantized.FloatFunctional()
        self.sum_4 = torch.nn.quantized.FloatFunctional()
        self.sum_5 = torch.nn.quantized.FloatFunctional()
        self.sum_6 = torch.nn.quantized.FloatFunctional()

        self.cat_1 = torch.nn.quantized.FloatFunctional()
        self.cat_2 = torch.nn.quantized.FloatFunctional()
        self.cat_3 = torch.nn.quantized.FloatFunctional()
        self.cat_4 = torch.nn.quantized.FloatFunctional()

        self.alpha = 1.0
        self.demosaic = DemosaicMalvar()
    
    def _set_alpha(self, alpha):
        self.alpha = alpha
        
    def forward(self, x, alpha = 1.6):
        # x = self.quant(x)
        x = x[:, :, 12:-12, 54:-54]
        x = x.sub_(60).div_(2**10)
        x = x.clamp(0, 1)
        x = torch.cat((x[:, :, 0::2,1::2],
                       x[:, :, 0::2,0::2],
                       x[:, :, 1::2,1::2],
                       x[:, :, 1::2,0::2]), dim=1)
        
        # scale = 0.3/torch.mean(x)
        # x = x*scale
        input_x = x

        x = x - torch.mean(x, dim=[1, 2, 3], keepdim=True)

        conv1_1 = self.conv1_1(x, )
        # mul = torch.mul(0.2, conv1_1)
        max_1 = self.l_relu(conv1_1)
        conv1_2 = self.conv1_2(max_1, )
        # mul_1 = torch.mul(0.2, conv1_2)
        max_2 = self.l_relu(conv1_2)
        max_2 = self.sum_1.add(max_1, max_2)
        pool1 = self.pool1(max_2, )
        conv2_1 = self.conv2_1(pool1, )
        # mul_2 = torch.mul(0.2, conv2_1)
        max_3 = self.l_relu(conv2_1)
        conv2_2 = self.conv2_2(max_3, )
        # mul_3 = torch.mul(0.2, conv2_2)
        max_4 = self.l_relu(conv2_2)
        max_4 = self.sum_2.add(max_4, max_3)
        pool1_1 = self.pool1_1(max_4, )
        conv3_1 = self.conv3_1(pool1_1, )
        # mul_4 = torch.mul(0.2, conv3_1)
        max_5 = self.l_relu(conv3_1)
        conv3_2 = self.conv3_2(max_5, )
        # mul_5 = torch.mul(0.2, conv3_2)
        max_6 = self.l_relu(conv3_2)
        max_6 = self.sum_3.add(max_6, max_5)
        pool1_2 = self.pool1_2(max_6, )
        conv4_1 = self.conv4_1(pool1_2, )
        # mul_6 = torch.mul(0.2, conv4_1)
        max_7 = self.l_relu(conv4_1)
        conv4_2 = self.conv4_2(max_7, )
        # mul_7 = torch.mul(0.2, conv4_2)
        max_8 = self.l_relu(conv4_2)
        max_8 = self.sum_4.add(max_8, max_7)
        pool1_3 = self.pool1_3(max_8, )
        conv5_1 = self.conv5_1(pool1_3, )
        # mul_8 = torch.mul(0.2, conv5_1)
        max_9 = self.l_relu(conv5_1)
        conv5_2 = self.conv5_2(max_9, )
        # mul_9 = torch.mul(0.2, conv5_2)
        max_10 = self.l_relu(conv5_2)
        max_10 = self.sum_5.add(max_10, conv5_2)
        upv6 = self.upv6(max_10, )
        cat = self.cat_1.cat([upv6, max_8], 1)
        conv6_1 = self.conv6_1(cat, )
        # mul_10 = torch.mul(0.2, conv6_1)
        max_11 = self.l_relu(conv6_1)
        conv6_2 = self.conv6_2(max_11, )
        # mul_11 = torch.mul(0.2, conv6_2)
        max_12 = self.l_relu(conv6_2)
        upv7 = self.upv7(max_12, )
        cat_1 = self.cat_2.cat([upv7, max_6], 1)
        conv7_1 = self.conv7_1(cat_1, )
        # mul_12 = torch.mul(0.2, conv7_1)
        max_13 = self.l_relu( conv7_1)
        conv7_2 = self.conv7_2(max_13, )
        # mul_13 = torch.mul(0.2, conv7_2)
        max_14 = self.l_relu(conv7_2)
        upv8 = self.upv8(max_14, )
        cat_2 = self.cat_3.cat([upv8, max_4], 1)
        conv8_1 = self.conv8_1(cat_2, )
        # mul_14 = torch.mul(0.2, conv8_1)
        max_15 = self.l_relu(conv8_1)
        conv8_2 = self.conv8_2(max_15, )
        # mul_15 = torch.mul(0.2, conv8_2)
        max_16 = self.l_relu(conv8_2)
        upv9 = self.upv9(max_16, )
        cat_3 = self.cat_4.cat([upv9, max_2], 1)
        conv9_1 = self.conv9_1(cat_3, )
        # mul_16 = torch.mul(0.2, conv9_1)
        max_17 = self.l_relu(conv9_1)
        conv9_2 = self.conv9_2(max_17, )
        # mul_17 = torch.mul(0.2, conv9_2)
        max_18 = self.l_relu(conv9_2)
        conv10_1 = self.conv10_1(max_18, )

        noise = conv10_1 - torch.mean(conv10_1, dim=[2, 3], keepdim=True)

        # un_norm
        out = self.sum_6.add(noise*self.alpha, input_x)
        # """
        # 4ch gray world
        out = torch.stack([
            out[:, 1, :, :]*out[:, 0, :, :].mean()/out[:, 1, :, :].mean(),
            (out[:, 0, :, :] + out[:, 3, :, :]) / 2,
            out[:, 2, :, :]*out[:, 3, :, :].mean()/out[:, 2, :, :].mean()], dim=1)
        # """
        """
        # 3ch gray world
        out = self.demosaic(out)
        out = torch.stack([
            out[:, 0, :, :]*out[:, 1, :, :].mean()/out[:, 0, :, :].mean(),
            out[:, 1, :, :],
            out[:, 2, :, :]*out[:, 1, :, :].mean()/out[:, 2, :, :].mean()], dim=1)
        # """
        out = out.clamp(0, 1).pow(1/2.2)
        # out = torch.cat((out, input_x), dim=0)#/scale
        # out = self.isp(out)
        # r = r/g.mean()
        # out = out.permute(2, 0, 3, 1)#.mul(1.5).clamp(0, 1).pow(1/2.2).mul_(255.0)
        # out = out.reshape(out.shape[0], -1, out.shape[-1])
        return out
    
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for o_name, o_param in own_state.items():
            for s_name, s_param in state_dict.items():
                # if o_name.find('res_enc')>=0:
                #     o_name = o_name[4:]
                if o_name == s_name and o_param.shape == s_param.shape:
                    # print('{} -> {}'.format(o_name, s_name))
                    # print('{} -> {}'.format(o_param.shape, s_param.shape))
                    own_state[o_name].copy_(state_dict[s_name])
                    break
                    
class DemosaicMalvar(nn.Module):
    def __init__(self):
        super().__init__()
        # """ # malvar
        self.g_at_r = (
            torch.Tensor(
                [
                    [0, 0, -1, 0, 0],
                    [0, 0, 2, 0, 0],
                    [-1, 2, 4, 2, -1],
                    [0, 0, 2, 0, 0],
                    [0, 0, -1, 0, 0],
                ]
            )
            .float()
            .view(1, 1, 5, 5)
            / 8
        )

        self.g_at_b = self.g_at_r.clone()

        self.r_at_g1 = (
            torch.Tensor(
                [
                    [0, 0, 0.5, 0, 0],
                    [0, -1, 0, -1, 0],
                    [-1, 4, 5, 4, -1],
                    [0, -1, 0, -1, 0],
                    [0, 0, 0.5, 0, 0],
                ]
            )
            .float()
            .view(1, 1, 5, 5)
            / 8
        )

        self.r_at_g2 = (
            torch.Tensor(
                [
                    [0, 0, -1, 0, 0],
                    [0, -1, 4, -1, 0],
                    [0.5, 0, 5, 0, 0.5],
                    [0, -1, 4, -1, 0],
                    [0, 0, -1, 0, 0],
                ]
            )
            .float()
            .view(1, 1, 5, 5)
            / 8
        )

        self.r_at_b = (
            torch.Tensor(
                [
                    [0, 0, -1.5, 0, 0],
                    [0, 2, 0, 2, 0],
                    [-1.5, 0, 6, 0, -1.5],
                    [0, 2, 0, 2, 0],
                    [0, 0, -1.5, 0, 0],
                ]
            )
            .float()
            .view(1, 1, 5, 5)
            / 8
        )
        """
        # bilinear
        self.g_at_r = (
            torch.Tensor(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 2, 0, 2, 0],
                    [0, 0, 2, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            )
            .float()
            .view(1, 1, 5, 5)
            / 8
        )

        self.g_at_b = self.g_at_r.clone()

        self.r_at_g1 = (
            torch.Tensor(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 4, 0, 4, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            )
            .float()
            .view(1, 1, 5, 5)
            / 8
        )

        self.r_at_g2 = (
            torch.Tensor(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 4, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 4, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            )
            .float()
            .view(1, 1, 5, 5)
            / 8
        )

        self.r_at_b = (
            torch.Tensor(
                [
                    [0, 0, 0, 0, 0],
                    [0, 2, 0, 2, 0],
                    [0, 0, 0, 0, 0],
                    [0, 2, 0, 2, 0],
                    [0, 0, 0, 0, 0],
                ]
            )
            .float()
            .view(1, 1, 5, 5)
            / 8
        )
        """

        self.b_at_g1 = self.r_at_g2.clone()
        self.b_at_g2 = self.r_at_g1.clone()
        self.b_at_r = self.r_at_b.clone()

    def forward(self, x):
        self.g_at_r = self.g_at_r.to(x.device)
        self.g_at_b = self.g_at_b.to(x.device)
        self.r_at_g1 = self.r_at_g1.to(x.device)
        self.r_at_g2 = self.r_at_g2.to(x.device)
        self.r_at_b = self.r_at_b.to(x.device)
        self.b_at_g1 = self.b_at_g1.to(x.device)
        self.b_at_g2 = self.b_at_g2.to(x.device)
        self.b_at_r = self.b_at_r.to(x.device)

        x = F.pixel_shuffle(x, 2)
        g00 = F.conv2d(x, self.g_at_r, padding=2, stride=2)
        g01 = x[:, :, 0::2, 0::2]
        g10 = x[:, :, 1::2, 1::2]
        g11 = F.conv2d(x.flip(dims=(2, 3)), self.g_at_b, padding=2, stride=2)
        g11 = g11.flip(dims=(2, 3))
        g = F.pixel_shuffle(torch.cat([g00, g01, g10, g11], dim=1), 2)

        r00 = x[:, :, 0::2, 1::2]
        r01 = F.conv2d(x.flip(dims=(3,)), self.r_at_g1, padding=2, stride=2)
        r01 = r01.flip(dims=(3,))
        r10 = F.conv2d(x.flip(dims=(2,)), self.r_at_g2, padding=2, stride=2)
        r10 = r10.flip(dims=(2,))
        r11 = F.conv2d(x.flip(dims=(2, 3)), self.r_at_b, padding=2, stride=2)
        r11 = r11.flip(dims=(2, 3))
        r = F.pixel_shuffle(torch.cat([r00, r01, r10, r11], dim=1), 2)

        b00 = F.conv2d(x, self.b_at_r, padding=2, stride=2)
        b01 = F.conv2d(x.flip(dims=(3,)), self.b_at_g1, padding=2, stride=2)
        b01 = b01.flip(dims=(3,))
        b10 = F.conv2d(x.flip(dims=(2,)), self.b_at_g2, padding=2, stride=2)
        b10 = b10.flip(dims=(2,))
        b11 = x[:, :, 1::2, 0::2]
        b = F.pixel_shuffle(torch.cat([b00, b01, b10, b11], dim=1), 2)
        return torch.cat([b, g, r], dim=1)



