import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    # conv2d -> Relu -> Conv2d -> Relu (padding=1로 입력력,출력 size 유지지)
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super(UNet, self).__init__()

        # Encoder ( Downsampling, 각 단계에서 채널 수는 2배씩 증가, Spatial resolution은 1/2로 감소 )
        self.encoder1 = ConvBlock(in_channels, 64)    # (B, 64, 256, 256)
        self.pool1 = nn.MaxPool2d(2)                  # (B, 64, 128, 128)

        self.encoder2 = ConvBlock(64, 128)            # (B, 128, 128, 128)
        self.pool2 = nn.MaxPool2d(2)                  # (B, 128, 64, 64)

        self.encoder3 = ConvBlock(128, 256)           # (B, 256, 64, 64)
        self.pool3 = nn.MaxPool2d(2)                  # (B, 256, 32, 32)

        self.encoder4 = ConvBlock(256, 512)           # (B, 512, 32, 32)
        self.pool4 = nn.MaxPool2d(2)                  # (B, 512, 16, 16)

        self.bottleneck = ConvBlock(512, 1024)        # (B, 1024, 16, 16)

        # Decoder (Upsampling + Skip Connection, Deconvolution -> 해상도 증가)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # (B, 512, 32, 32) 
        self.decoder4 = ConvBlock(1024, 512)                                   # (B, 512, 32, 32)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   # (B, 256, 64, 64)
        self.decoder3 = ConvBlock(512, 256)                                    # (B, 256, 64, 64)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   # (B, 128, 128, 128)
        self.decoder2 = ConvBlock(256, 128)                                    # (B, 128, 128, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    # (B, 64, 256, 256)
        self.decoder1 = ConvBlock(128, 64)                                     # (B, 64, 256, 256)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)            # (B, num_classes, 256, 256)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder (torch.cat : 업샘플된 피처맵과 같은 레벨의 encoder 출력 연결 -> Skip connection)
        dec4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat((dec4,enc4), dim=1)) # (B, 1024, 32, 32)

        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))  # (B, 512, 64, 64)

        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))  # (B, 256, 128, 128)

        dec1 = self.upconv1(dec2)                        
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))  # (B, 128, 256, 256)

        return self.final_conv(dec1) 
