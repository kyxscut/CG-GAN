import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import string


class ConcatLmdbDataset(Dataset):
    def __init__(self, dataset_list, batchsize_list, 
        ttfRoot=None, corpusRoot=None, transform_img=None,transform_target_img=None, alphabet=string.printable[:-6]):
        #import pdb;pdb.set_trace()
        assert len(dataset_list) == len(batchsize_list)

        if alphabet[-4:] == '.txt':
            alphabet_char = open(alphabet, 'r').read().splitlines()
        alphabet = ''.join(alphabet_char)

        self.corpus = open(corpusRoot, "r").read().splitlines()
        print('Totally %d strings in corpus.' % len(self.corpus))
        
        
        radical_dict = dict()
        total = open('data/IDS_dictionary.txt','r').read().splitlines()
        for line in total:
            char,radical = line.split(':')[0],line.split(':')[1]
            radical_dict[char] = radical
        
        if not os.path.isdir(ttfRoot):
            print('%s: the path to *.ttf is not a exist.' % (ttfRoot))
            sys.exit(0)        
        ttf = False
        self.font_path = []
        ttf_dir = os.walk(ttfRoot)
        for path, d, filelist in ttf_dir:
            for filename in filelist:
                if filename.endswith('.ttf') or filename.endswith('.ttc'):
                    self.font_path.append(path+'/'+filename)
                    ttf = True
        if not ttf:
            print('There is no ttf file in the dir.')
            sys.exit(0)
        else:
            print('Totally %d fonts for single character generation.' % len(self.font_path))
        
        self.datasets = []
        self.prob = [batchsize / sum(batchsize_list) for batchsize in batchsize_list]
        for i in range(len(dataset_list)):
            print('For every iter: %s samples from %s' % (batchsize_list[i], dataset_list[i]))
            self.datasets.append(lmdbDataset(dataset_list[i], self.font_path, self.corpus, transform_img,transform_target_img, alphabet,radical_dict))
        self.datasets_range = range(len(self.datasets))

    def __len__(self):
        return max([dataset.__len__() for dataset in self.datasets])

    def __getitem__(self, index):

        idx_dataset = np.random.choice(self.datasets_range, 1, p=self.prob).item()
        idx_sample = index % self.datasets[idx_dataset].__len__()
        #import pdb;pdb.set_trace()
        return self.datasets[idx_dataset][idx_sample]

class lmdbDataset(Dataset):

    def __init__(self, root=None, font_path=None, corpus=None,
        transform_img=None,transform_target_img=None, alphabet=string.printable[:-6], radical_dict = None):
        assert transform_img != None
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot open lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
        
        self.root = root
        self.transform_img = transform_img
        self.transform_target_img = transform_target_img
        self.font_path = font_path
        self.corpus = corpus
        self.alphabet = alphabet
        self.radical_dict = radical_dict
        

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            if label == '##':
                return self[index + 1]

            lexicon_Key = 'lexicon-%09d' % index
            lexicon = str(txn.get(lexicon_Key.encode()).decode('utf-8'))
            space_list = ['⿰','⿱','⿳','⿺','⿶','⿹','⿸','⿵','⿲','⿴','⿷','⿻']
            lexicon_list_old = lexicon.split()
            lexicon_list = []
            for i in lexicon_list_old:
                if i not in space_list:
                    lexicon_list.append(i)
            lexicon = ' '.join(lexicon_list)
         
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            
            writerID_key = 'writerID-%09d' % index
            writerID = int(txn.get(writerID_key.encode()))

            font = ImageFont.truetype(self.font_path[random.randint(0,len(self.font_path)-1)], 80)
            label_target = self.corpus[random.randint(0, len(self.corpus)-1)]                        
            lexicon_target = self.radical_dict[label_target]
            lexicon_target_list_old = lexicon_target.split()
            lexicon_target_list = []
            for i in lexicon_target_list_old:
                if i not in space_list:
                    lexicon_target_list.append(i)
            lexicon_target = ' '.join(lexicon_target_list)
            
            try:
                label_w, label_h = font.getsize(label_target)
                img_target = Image.new('RGB', (label_w, label_h), (255, 255, 255))
                drawBrush = ImageDraw.Draw(img_target)
                drawBrush.text((0, 0), label_target, fill=(0, 0, 0), font=font)
                
            except Exception as e:
                with open('failed_font.txt', 'a+') as f:
                    f.write(self.font_path[index % len(self.font_path)] + '\n')
                return self[index + 1]
            
            img_target = self.transform_target_img(img_target)
            img = self.transform_img(img)
            ###################### Target ######################

            return {'A': img, 'B': img_target, 'A_paths': (index-1) % len(self.corpus), 'writerID': writerID,
            'A_label': label, 'B_label': label_target,'root':self.root,'A_lexicon':lexicon,'B_lexicon':lexicon_target}
            

class resizeKeepRatio(object):

    def __init__(self, size, interpolation=Image.BILINEAR, 
        train=False):

        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.train = train

    def __call__(self, img):

        if img.mode == 'L':
            img_result = Image.new("L", self.size, (255))
        elif img.mode =='RGB':
            img_result = Image.new("RGB",self.size, (255, 255, 255))
        else:
            print("Unknow image mode!")

        img_w, img_h = img.size

        target_h = self.size[1]
        target_w = max(1, int(img_w * target_h / img_h))

        if target_w > self.size[0]:
            target_w = self.size[0]

        img = img.resize((target_w, target_h), self.interpolation)
        begin = random.randint(0, self.size[0]-target_w) if self.train else int((self.size[0]-target_w)/2)
        box = (begin, 0, begin+target_w, target_h)
        img_result.paste(img, box)

        img = self.toTensor(img_result)
        img.sub_(0.5).div_(0.5)
        return img


if __name__ =='__main__':
    dataset = ConcatLmdbDataset(
        dataset_list = ['data/FFG_lmdb_dataset_423fonts/test_399fonts_oov_seenstyles_addradical'],
        batchsize_list = [1],
        ttfRoot = 'data/font',
        corpusRoot = "data/char_seen_set.txt",
        transform_img= resizeKeepRatio((128,128)),
        transform_target_img=resizeKeepRatio((128,128)),
        alphabet = "data/char_seen_set.txt",      
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,batch_size =1,sampler =None,drop_last=True,num_workers = 0,shuffle=False
    )
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    for i, data in enumerate(train_loader):
        import pdb;pdb.set_trace()
        print(data)