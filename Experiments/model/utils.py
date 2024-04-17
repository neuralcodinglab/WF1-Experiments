import torch




def tensor2mask(pred, thres):
    """Puts opencv image (numpy BGR format) into a normalized Pytorch tensor on specified device"""
    x = img[:,:,[2,1,0]].swapaxes(0,2)/255
    x = torch.tensor(x,dtype=torch.float)
    x = F.normalize(x, mean, std)
    x = x.unsqueeze(0).to('cuda:0')
    return pred.cpu().numpy().squeeze()
