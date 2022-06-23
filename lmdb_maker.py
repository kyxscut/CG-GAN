# -*- coding: utf-8 -*-  

import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import codecs
import random
import glob

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR)
        # if img == None:
        #     return False
        imgH, imgW, imgC = img.shape[0], img.shape[1], img.shape[2]
        if imgH * imgW * imgC == 0:
            return False
    except:
        return False
    # img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    # imgH, imgW = img.shape[0], img.shape[1]
    # if imgH * imgW == 0:
    #     return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)

def createDataset(outputPath, imagePathList, labelList, writerIDList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    #import pdb;pdb.set_trace()
    nSamples = len(imagePathList)
    
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    import pdb;pdb.set_trace()
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        writerID = writerIDList[i]

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        # import pdb; pdb.set_trace()
        
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        writerIDKey = 'writerID-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        cache[writerIDKey] = writerID
        #import pdb;pdb.set_trace()
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = lexiconList[i]
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

if __name__ == '__main__':


    dictionary_dir = '.txt'
    radical_dict= open(dictionary_dir,"r").read().splitlines()
    cache = dict()
    for line in radical_dict:
        label,radical = line.split(':')[0],line.split(':')[1]
        cache[label] = radical

    #import pdb;pdb.set_trace()
    img_path_list = []
    label_list = []
    ID_list = []
    lexicon_list = []
    
    img_file_list = open('./train_set.txt','r').read().splitlines()

    for img_path in img_file_list:        
        img_path_list.append(img_path)
        label = img_path.split('/')[-1].split('_')[-1].split('.')[0]
        label_list.append(label)
        lexicon = cache[label]
        lexicon_list.append(lexicon)
        writerID_str = img_path.split('/')[-2]
        if len(writerID_str) > 4:
            import pdb; pdb.set_trace()

        if writerID_str[0] == 'C':
            writerID_str = writerID_str[1:]

        try:
            writerID = int(writerID_str)
        except Exception as e:
            print(e)
            continue
        ID_list.append(writerID_str)
        

        # import pdb; pdb.set_trace()
    import pdb;pdb.set_trace()
    assert(len(img_path_list) == len(label_list))
    assert(len(img_path_list) == len(ID_list))
    assert(len(img_path_list) == len(lexicon_list))

    print('total sample: %d' % len(img_path_list))
    
    createDataset('./data/train_set', img_path_list, label_list, ID_list,lexicon_list)
