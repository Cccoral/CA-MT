from networks.unet import UNet

def net_with_ema(in_chns=3,class_num=5,ema=False):
    net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    if ema: 
        for param in net.parameters():
            param.detach_()
    return net
