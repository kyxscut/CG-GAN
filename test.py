import torch
import os
from options.test_options import TestOptions
import data.lmdb_dataset as lmdb_dataset
from models import create_model
from util.visualizer import save_images
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.serial_batches = True
    dataset = lmdb_dataset.ConcatLmdbDataset(
        dataset_list=opt.dataroot, 
        batchsize_list=opt.batch_size, 
        ttfRoot=opt.ttfRoot,
        corpusRoot=opt.corpusRoot,
        transform_img=lmdb_dataset.resizeKeepRatio((opt.imgW, opt.imgH)),
        transform_target_img=lmdb_dataset.resizeKeepRatio((opt.imgW, opt.imgH)),
        alphabet=opt.alphabet
        )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=sum(opt.batch_size),shuffle =False,
        num_workers=int(opt.num_threads))
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of testing images = %d' % dataset_size)
    #import pdb;pdb.set_trace()
    model = create_model(opt)
    model.setup(opt)


    web_dir = os.path.join(opt.results_dir,opt.name,'{}_{}_{}'.format(opt.phase,opt.epoch, opt.state))
    if opt.load_iter>0:
        web_dir = '{:s}_iter{:d}'.format(web_dir,opt.load_iter)
    print('creating web directory',web_dir)
    webpage = html.HTML(web_dir,'Experiment =%s,Phase =%s,Epoch =%s' %(opt.name,opt.phase,opt.epoch))
    if opt.eval:
        model.eval()
    for i,data in enumerate(test_loader):
        #import pdb;pdb.set_trace()
        if i>=opt.num_test:
           break
        #import pdb;pdb.set_trace()
        model.set_input(data)
        model.test()
        visuals =model.get_current_visuals()
        img_path = model.get_image_paths()
        # import pdb;pdb.set_trace()
        if i % 5 ==0:
            print('preprocessing (）%04d-th image...%s' %(i,img_path))
        save_images(webpage,visuals,img_path,aspect_ratio=opt.aspect_ratio,width=opt.display_winsize)
    webpage.save()