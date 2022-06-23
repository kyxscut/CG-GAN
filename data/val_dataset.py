import random
from torch.utils.data import Dataset
import os
from PIL import Image, ImageDraw, ImageFont
try:
    from lmdb_dataset import resizeKeepRatio
except:
    from data.lmdb_dataset import resizeKeepRatio

def make_dataset(dir):
    path_list = open(dir,'r').read().splitlines()
    images = []
    #import pdb;pdb.set_trace()
    for img_path in path_list:
        label =img_path.split('/')[-1].split('.')[0].split('_')[-1]
        id = int(img_path.split('/')[-2])
        item = (img_path,id,label)
        images.append(item)
        
    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def draw(font_path,label):
    font = ImageFont.truetype(font_path,80)
    label_w, label_h = font.getsize(label)
    img_target =Image.new('RGB', (label_w,label_h),(255,255,255))
    drawBrush = ImageDraw.Draw(img_target)
    drawBrush.text((0,0),label,fill=(0,0,0),font = font)
    return img_target

class ValDataset(Dataset):
    def __init__(self,root,ttfRoot,target_transform = resizeKeepRatio((128, 128))):
        #import pdb;pdb.set_trace()
        samples = make_dataset(root)
        #import pdb;pdb.set_trace()
        self.samples = samples
        self.ids = [s[1] for s in samples]
        self.target_transform = target_transform
        self.loader = pil_loader
        self.font_path = []
        ttf_dir = os.walk(ttfRoot)
        for path, d, filelist in ttf_dir:
            for filename in filelist:
                if filename.endswith('.ttf') or filename.endswith('.ttc'):
                    self.font_path.append(path+'/'+filename)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path,id,label = self.samples[index]
        img = self.loader(path)
        if self.target_transform is not None:
            img = self.target_transform(img)
        label_target = label
        font_path = self.font_path[random.randint(0,len(self.font_path)-1)]
        img_target = draw(font_path,label_target)
        img_target = self.target_transform(img_target)
        writerID = id               
        return {'A': img, 'B': img_target, 'A_paths': index, 'writerID': writerID,
            'A_label': label, 'B_label': label_target, 'val':True}


