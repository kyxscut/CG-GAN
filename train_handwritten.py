import time
from options.train_options import TrainOptions
import data_iam.lmdb_dataset_iam as lmdb_dataset
import data_iam.val_dataset_iam as val_dataset
from models import create_model
from util.visualizer import Visualizer
from validation import validateUN
import torch
import os

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    #import pdb;pdb.set_trace()
    dataset = lmdb_dataset.ConcatLmdbDataset(
        dataset_list=opt.dataroot, 
        batchsize_list=opt.batch_size, 
        ttfRoot=opt.ttfRoot,
        corpusRoot=opt.corpusRoot,
        transform_img=lmdb_dataset.resizeKeepRatio((opt.imgW, opt.imgH)),
        transform_target_img=lmdb_dataset.resizeKeepRatio((opt.imgW, opt.imgH)),
        alphabet=opt.alphabet)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=sum(opt.batch_size),
        shuffle=True, sampler=None, drop_last=True,
        num_workers=int(opt.num_threads))
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    valdataset = val_dataset.ValDataset(root = opt.val_seenstyleRoot, ttfRoot= opt.ttfRoot)
    valdataset_unseen = val_dataset.ValDataset(root = opt.val_unseenstyleRoot, ttfRoot= opt.ttfRoot)
    #import pdb;pdb.set_trace()
    validationFunc = validateUN
    val_dir = os.path.join(opt.checkpoints_dir, opt.name,'validation_seenstyle')
    val_dir_unseen = os.path.join(opt.checkpoints_dir, opt.name,'validation_unseenstyle')
    if not os.path.isdir(val_dir):
        os.mkdir(val_dir)
    if not os.path.isdir(val_dir_unseen):
        os.mkdir(val_dir_unseen)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
   
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0 
        model.train()    
        
        for i, data in enumerate(train_loader):  # inner loop within one epoch            
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += sum(opt.batch_size)
            epoch_iter += sum(opt.batch_size)
            #import pdb;pdb.set_trace()
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / sum(opt.batch_size)
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        
        #if epoch >=5:
        validationFunc(dataset=valdataset, model=model, epoch=epoch, val_dir=val_dir, val_num=opt.val_num)
        validationFunc(dataset=valdataset_unseen, model = model,epoch= epoch, val_dir=val_dir_unseen, val_num=opt.val_num)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
