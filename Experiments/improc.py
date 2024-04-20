from model import resnet,sharpnet
import torch
import torchvision.transforms.functional as F
import numpy as np
import cv2
from threading import Thread


## Image Preprocessing Models and Phosphene simulator


class SharpNetFilter(object):
    """Loads SharpNet model with pre-trained weights.
    Can be called on opencv image format (numpy GBR array of type uint8)"""

    def __init__(self,checkpoint_path='./model/weights/final_checkpoint_NYU.pth', device='cuda:0', block=resnet.Bottleneck,
                layers_encoder=[3, 4, 6, 3],layers_decoders=[2, 2, 2, 2, 2],use_normals=True,
                use_depth=False,use_boundary=True,bias_decoder=False,smooth_output=1,threshold_b=94,threshold_c=80):

        # Post-processing parameters
        self.smooth_output  = smooth_output
        self.threshold_b    = threshold_b
        self.threshold_c    = threshold_c

        # Model output
        self.boundary       = None
        self.normals        = None
        self.canny          = None

        # initialize model
        torch.set_grad_enabled(False)
        self.checkpoint_path = checkpoint_path
        self.device          = device
        self.model           = sharpnet.SharpNet(block, layers_encoder, layers_decoders,use_normals,use_depth,
                                                use_boundary, bias_decoder)

        # Load model weights
        checkpoint   = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.weights = {k: v for k, v in checkpoint.items() if k in self.model.state_dict()}
        self.model.load_state_dict(self.weights)
        self.model.eval()
        self.model.to(device)

    def __call__(self, img, resize=None, store_raw=False):
        """Returns smoothed boundary prediction (binary mask)"""
        # Forward pass and unnormailzation
        x    = cvimg2tensor(img, device=self.device)
        normals, boundary = self.model(x)
        normals  = normals.cpu().numpy().squeeze().transpose([1,2,0]).swapaxes(0,1)
        normals  = ( (255*(normals-normals.min()))/(normals.max()-normals.min())).astype('uint8')
        boundary = (boundary.cpu().numpy().squeeze()*255).astype('uint8').swapaxes(0,1)

        # Post-processing
        if resize is not None:
            normals  = cv2.resize(normals, resize)
            boundary = cv2.resize(boundary, resize)
        boundary = cv2.GaussianBlur(boundary,(0,0),self.smooth_output)
        boundary = (boundary>self.threshold_b).astype('uint8')
        # normals  = cv2.GaussianBlur(normals,(0,0),self.smooth_output)
        canny    = cv2.Canny(normals, int(self.threshold_c//2),self.threshold_c)
        canny    = cv2.dilate(canny,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        pred = canny*boundary
        pred = cv2.erode(pred,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        canny = cv2.erode(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

        if store_raw:
            self.boundary = cv2.cvtColor(boundary*255,cv2.COLOR_GRAY2BGR)
            self.normals = normals
            self.canny = cv2.cvtColor(canny,cv2.COLOR_GRAY2BGR)
        return pred


class CannyFilter(object):
    def __init__(self,sigma=3,low=25,high=50):
        # Parameters
        self.sigma = sigma
        self.low = low
        self.high = high

    def __call__(self, img):
        # Gaussian blur and Canny edge detection
        pred = cv2.GaussianBlur(img,(0,0),self.sigma) # gaussian blur (to remove noise)
        pred = cv2.Canny(pred,self.low,self.high) # Canny edge detection
        return pred

class PhospheneSimulator(object):
    def __init__(self,phosphene_resolution=(50,50), size=(480,480), intensity=1, jitter=0.4, intensity_var=0.8, aperture=.66, sigma=2, custom_grid=None):
        """Phosphene simulator class to create gaussian-based phosphene simulations from activation mask
        on __init__, provide custom phosphene grid or use the grid parameters to create one
        - aperture: receptive field of each phosphene (uses dilation of the activation mask to achieve this)
        - sigma: the size parameter for the gaussian phosphene simulation """
        if custom_grid is None:
            self.phosphene_resolution = phosphene_resolution
            self.size = size
            self.phosphene_spacing = np.divide(size,phosphene_resolution)
            self.jitter = jitter
            self.intensity = intensity
            self.intensity_var = intensity_var
            self.grid = self.create_regular_grid(self.phosphene_resolution,self.size,self.jitter,self.intensity_var)
            self.aperture = np.round(aperture*self.phosphene_spacing[0]).astype(int) #relative aperture > dilation kernel size
        else:
            self.grid = custom_grid
            self.aperture = aperture
        self.sigma = sigma
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.aperture,self.aperture))
        self.k_size = np.round(4*sigma+1).astype(int) # rule of thumb: choose k_size>3*sigma

    def __call__(self,activation_mask):
        """ returns the phosphene simulation (image), given an activation mask"""
        assert self.grid.shape == activation_mask.shape
        self.mask = cv2.dilate(activation_mask, self.dilation_kernel, iterations=1)
        phosphenes = self.grid * self.mask
        phosphenes = (cv2.GaussianBlur(phosphenes,(self.k_size,self.k_size),self.sigma)*self.intensity).astype('uint8')
        # print(self.mask)
        phosphenes = cv2.cvtColor(phosphenes,cv2.COLOR_GRAY2BGR)
        return phosphenes

    def create_regular_grid(self, phosphene_resolution, size, jitter, intensity_var):
        """Returns regular eqiodistant phosphene grid of shape <size> with resolution <phosphene_resolution>
         for variable phosphene intensity with jitterred positions"""
        grid = np.zeros(size)
        phosphene_spacing = np.divide(size,phosphene_resolution)
        for x in np.linspace(0,size[0],num=phosphene_resolution[0],endpoint=False)+0.5*phosphene_spacing[0] :
            for y in np.linspace(0,size[1],num=phosphene_resolution[1],endpoint=False)+0.5*phosphene_spacing[0]:
                deviation = np.multiply(jitter*(2*np.random.rand(2)-1),phosphene_spacing)
                intensity = intensity_var*(np.random.rand()-0.5)+1
                rx = np.clip(np.round(x+deviation[0]),0,size[0]-1).astype(int)
                ry = np.clip(np.round(y+deviation[1]),0,size[1]-1).astype(int)
                grid[rx,ry]= intensity
        return grid

## Utils##

def crop_resize(img, zoom=1.9, width=480, height=480, ):
    """Center-crop and resize img to height x width (assumes w>h)"""
    margin = [int((height-(height/zoom))//2), int((width-(width/zoom)+(img.shape[1]-img.shape[0]))//2)]
    img = img[margin[0]:-margin[0],margin[1]:-margin[1],:]
    img = cv2.resize(img,(width,height))
    return img

def center_crop(img, zoom=1.9, resize=None):
    """Center-crop and resize img to height x width (assumes w>h)"""
    h,w,_ = img.shape
    x = min([w,h])
    # margin = [int((height-(height/zoom))//2), int((width-(width/zoom)+(img.shape[1]-img.shape[0]))//2)]
    margin = [int((x-(x/zoom))//2), int((x-(x/zoom)+(w-h))//2)]
    img = img[margin[0]:-margin[0],margin[1]:-margin[1],:]
    if resize is not None:
        img = cv2.resize(img,resize)
    return img

def cvimg2tensor(img, device='cuda:0',mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Puts opencv image (numpy BGR format) into a normalized Pytorch tensor on specified device"""
    x = img[:,:,[2,1,0]].swapaxes(0,2)/255
    x = torch.tensor(x,dtype=torch.float)
    x = F.normalize(x, mean, std)
    x = x.unsqueeze(0).to(device)
    return x


## Camera and VR display ##

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, cameradevice=0):
        self.stream = cv2.VideoCapture(cameradevice,cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
        self.stream.release()


class ShowVR:
    """
    Class that continuously shows a frame (in VR style) using a dedicated thread.
    """

    def __init__(self,windowname='VRSimulation', resolution=(1600,2880), imgsize=(480,480), ipd=1240, frame=None):
        self._resolution    = resolution
        self._imgsize       = imgsize
        self._ipd           = ipd
        self._windowname     = windowname
        self._screen         = np.zeros(resolution+(3,)).astype('uint8')
        self.frame          = frame if frame is not None else np.zeros(imgsize+(3,))
        self.stopped        = False
        self.keypress       = None


    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        cv2.imshow(self._windowname, self._screen)
        cv2.setWindowProperty(self._windowname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while not self.stopped:
            img = cv2.resize(self.frame, self._imgsize)
            pos=[int((self._resolution[i]-self._imgsize[i])//2) for i in range(2)]
            pos[1] = pos[1]-int(self._ipd//2) # shift leftward with 0.5 x ipd
            self._screen[pos[0]:pos[0]+self._imgsize[0], pos[1]:pos[1]+self._imgsize[1],:]=img
            pos[1] = pos[1] + self._ipd # shift rightward with 1 x ipd
            self._screen[pos[0]:pos[0]+self._imgsize[0], pos[1]:pos[1]+self._imgsize[1],:]=img
            cv2.imshow(self._windowname, self._screen)
            cv2.imshow('monitor', img)
            key = cv2.waitKey(1)
            if key != -1:
                self.keypress = key

    def getKey(self):
        key = self.keypress
        self.keypress=None
        return key

    def stop(self):
        self.stopped = True
