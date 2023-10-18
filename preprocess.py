import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

from timm.models.resnet import *

import cv2
import numpy as np


##############################################################
class PreprocessNet(nn.Module):

    def __init__(self,):
        super(PreprocessNet, self).__init__()
        self.register_buffer('mean', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))

        self.encoder = resnet34d(pretrained=True)
        self.encoder.maxpool = nn.Identity()
        dim = 512

        self.mask = nn.Sequential(
            nn.Conv2d(dim,4*16, kernel_size = 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(True),
            nn.Conv2d(16,4, kernel_size = 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.laterality = nn.Linear(dim,1)


    def forward(self, batch):
        x = batch['image']
        x = (x - self.mean) / self.std
        e = self.encoder.forward_features(x)
        mask = self.mask(e)
        mask = torch.sigmoid(mask)
        mask  = F.interpolate(mask,size=x.shape[-2:],mode='bilinear',align_corners=False)

        pool = F.adaptive_avg_pool2d(e,1)
        pool = torch.flatten(pool,1)
        laterality = self.laterality(pool).reshape(-1)
        laterality = torch.sigmoid(laterality)

        output={
            'mask':mask,
            'laterality':laterality,
        }
        return output

##############################################################

class PreprocessDataset(Dataset):
    def __init__(self, df, image_size=4096):
        self.df = df
        self.length = len(df)
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        m = cv2.imread(d.png_file, cv2.IMREAD_GRAYSCALE)
        h, w = m.shape
        image = np.zeros((4096, 4096), np.uint8)
        
        

        s = self.image_size /h
        h, w = int(s * h), int(s * w)
        m = cv2.resize(m, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        y = (self.image_size - h) // 2
        x = (self.image_size - w) // 2
        rect = (x, y, x + w, y + h)

        image[y:y + h, x:x + w] = m
        image = image.astype(np.float32) / 255

        r = {}
        r['index'] = index
        r['d'    ] = d
        r['rect' ] = rect
        r['image'] = torch.from_numpy(image).float()
        return r

def proprocess_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        v = [b[k] for b in batch]
        d[k] = v
    d['image'] = torch.stack(d['image'],0).unsqueeze(1)
    return d

#-----------------------------------------

def predict_to_mask(predict):

    #for image size 224
    erosion_size = 7
    min_diff = 224

    #----
    thresh = ((predict>0.5)*255).astype(np.uint8)
    contour, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_NONE     CHAIN_APPROX_SIMPLE
    c = max(contour, key=cv2.contourArea)
    m0 = np.zeros(shape=thresh.shape, dtype=np.uint8)
    cv2.drawContours(m0, [c], -1, 255, thickness=-1)

    element = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
    m1 = cv2.erode(m0, element)
    contour, hierarchy = cv2.findContours(m1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_NONE     CHAIN_APPROX_SIMPLE
    c = max(contour, key=cv2.contourArea)
    m1[...]=0
    cv2.drawContours(m1, [c], -1, 255, thickness=-1)

    m2 = cv2.dilate(m1, element)

    ##----------
    ss = (m0 == m2).sum()
    if ss < min_diff:
        mask = m0
    else:
        mask = m2

    mask = mask/255
    return mask

def mask_to_box(mask, pad=0.03):
    yy, xx = np.where(mask > 0)
    xmin = xx.min()
    xmax = xx.max()
    ymin = yy.min()
    ymax = yy.max()

    h,w = mask.shape
    pad = int(pad*h)
    xmin = max(0, xmin - pad)
    xmax = min(w, xmax + pad)
    ymin = max(0, ymin - pad)
    ymax = min(h, ymax + pad)
    return xmin, ymin, xmax, ymax


def post_process(batch, output):
    rect  = batch['rect']
    image = [m for m in batch['image'].float().data.cpu().numpy().squeeze(1)]
    mask  = [p for p in output['mask'].float().data.cpu().numpy().squeeze(1)]
    laterality = (output['laterality'].float().data.cpu().numpy() > 0.5).astype(int).tolist()
    box = []

    batch_size = len(batch['index'])
    for b in range(batch_size):
        x0, y0, x1, y1 = rect[b]
        m = image[b][y0:y1, x0:x1]
        p = mask[b][y0:y1, x0:x1]

        h = y1-y0
        p = predict_to_mask(p)
        xmin, ymin, xmax, ymax = mask_to_box(p)
        box.append((xmin/h, ymin/h, xmax/h, ymax/h))
        image[b] = m
        mask[b] = p

    output = {
        'image' : image,
        'mask' : mask,
        'laterality' : laterality,
        'box' : box,
    }
    return output


#-----------------------------------------
def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)


def image_show(name, image, type='bgr', resize=1):
    if type == 'rgb': image = np.ascontiguousarray(image[:, :, ::-1])
    H, W = image.shape[0:2]

    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)  # WINDOW_NORMAL
    # cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)  #WINDOW_GUI_EXPANDED
    cv2.imshow(name, image)  # .astype(np.uint8))
    cv2.resizeWindow(name, round(resize * W), round(resize * H))



def draw_preprocess_overlay(image, mask, box, laterality):
    h,w = image.shape
    xmin, ymin, xmax, ymax = box
    xmin = int(xmin*h)
    ymin = int(ymin*h)
    xmax = int(xmax*h)
    ymax = int(ymax*h)

    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = 1 - (1 - mask[..., np.newaxis] * [[[1, 0, 0]]]) * (1 - overlay)
    cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 0, 1), 2)

    if laterality == 0:
        draw_shadow_text(overlay, 'left', (5, 20), 0.7, (1, 1, 1), 2)
    if laterality == 1:
        draw_shadow_text(overlay, 'right', (overlay.shape[1] - 55, 20), 0.7, (1, 1, 1), 2)
    return overlay

