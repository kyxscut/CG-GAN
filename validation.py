import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from data.lmdb_dataset import resizeKeepRatio
from util.visualizer import save_images_validation
from fid_folder import fid_score
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
from random import sample


def validateUN(dataset, model, epoch,val_dir,val_num):
    
    model = model
    model.eval()    
    val_dataset = dataset
    save_dir = os.path.join(val_dir,'epoch_%s' % epoch)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    with torch.no_grad():
        val_tot_tars = torch.tensor(val_dataset.ids)
        for cls_idx in range(0,500):
            tmp_cls_idx = (val_tot_tars == cls_idx).nonzero()[-val_num:]
            if cls_idx == 0:
                val_idx = tmp_cls_idx.clone()
            else:
                val_idx = torch.cat((val_idx,tmp_cls_idx))
        tmp_ds = torch.utils.data.Subset(val_dataset,val_idx)
        tmp_dl = torch.utils.data.DataLoader(tmp_ds,batch_size=1, shuffle=False,num_workers=8,pin_memory=True,drop_last=False)
        
        for i,data in tqdm(enumerate(tmp_dl)):            
            model.set_input(data)
            model.test()
            visuals =model.get_current_visuals()
            img_path = model.get_image_paths()
            save_images_validation(save_dir,visuals,img_path)
        
        print('done')
#         eval.main(save_dir)
        fid_score.calculate_fid_given_paths([save_dir],128,True,2048)

        

        


























            
            
            
        








    
        











    """
    x_each_cls = []
    with torch.no_grad():
        val_tot_tars = torch.tensor(val_dataset.targets)
        for cls_idx in range(len(args.att_to_use)):
            tmp_cls_set = (val_tot_tars == args.att_to_use[cls_idx]).nonzero()[-args.val_num:]
            tmp_ds = torch.utils.data.Subset(val_dataset, tmp_cls_set)
            tmp_dl = torch.utils.data.DataLoader(tmp_ds, batch_size=args.val_num, shuffle=False,
                                                 num_workers=0, pin_memory=True, drop_last=False)
            tmp_iter = iter(tmp_dl)
            tmp_sample = None
            for sample_idx in range(len(tmp_iter)):
                imgs, _ = next(tmp_iter)
                x_ = imgs
                if tmp_sample is None:
                    tmp_sample = x_.clone()
                else:
                    tmp_sample = torch.cat((tmp_sample, x_), 0)
            x_each_cls.append(tmp_sample)
    
    import pdb;pdb.set_trace()
    if epoch >= args.fid_start:
        # Reference guided
        with torch.no_grad():
           # Just a buffer image ( to make a grid )
            ones = torch.ones(1, x_each_cls[0].size(1), x_each_cls[0].size(2), x_each_cls[0].size(3)).cuda(args.gpu, non_blocking=True)
            for src_idx in range(len(args.att_to_use)):
                x_src = x_each_cls[src_idx][:args.val_batch, :, :, :].cuda(args.gpu, non_blocking=True)
                rnd_idx = torch.randperm(x_each_cls[src_idx].size(0))[:args.val_batch]
                x_src_rnd = x_each_cls[src_idx][rnd_idx].cuda(args.gpu, non_blocking=True)
                for ref_idx in range(len(args.att_to_use)):
                    x_res_ema = torch.cat((ones, x_src), 0)
                    x_rnd_ema = torch.cat((ones, x_src_rnd), 0)
                    x_ref = x_each_cls[ref_idx][:args.val_batch, :, :, :].cuda(args.gpu, non_blocking=True)
                    rnd_idx = torch.randperm(x_each_cls[ref_idx].size(0))[:args.val_batch]
                    x_ref_rnd = x_each_cls[ref_idx][rnd_idx].cuda(args.gpu, non_blocking=True)
                    for sample_idx in range(args.val_batch):
                        x_ref_tmp = x_ref[sample_idx: sample_idx + 1].repeat((args.val_batch, 1, 1, 1))
    
                        c_src, skip1, skip2 = G_EMA.cnt_encoder(x_src)
                        s_ref = C_EMA(x_ref_tmp, sty=True)
                        x_res_ema_tmp,_ = G_EMA.decode(c_src, s_ref, skip1, skip2)
    
                        x_ref_tmp = x_ref_rnd[sample_idx: sample_idx + 1].repeat((args.val_batch, 1, 1, 1))
    
                        c_src, skip1, skip2 = G_EMA.cnt_encoder(x_src_rnd)
                        s_ref = C_EMA(x_ref_tmp, sty=True)
                        x_rnd_ema_tmp,_ = G_EMA.decode(c_src, s_ref, skip1, skip2)
    
                        x_res_ema_tmp = torch.cat((x_ref[sample_idx: sample_idx + 1], x_res_ema_tmp), 0)
                        x_res_ema = torch.cat((x_res_ema, x_res_ema_tmp), 0)
    
                        x_rnd_ema_tmp = torch.cat((x_ref_rnd[sample_idx: sample_idx + 1], x_rnd_ema_tmp), 0)
                        x_rnd_ema = torch.cat((x_rnd_ema, x_rnd_ema_tmp), 0)
    
                    vutils.save_image(x_res_ema, os.path.join(args.res_dir, '{}_EMA_{}_{}{}.jpg'.format(args.gpu, epoch+1, src_idx, ref_idx)), normalize=True,
                                    nrow=(x_res_ema.size(0) // (x_src.size(0) + 2) + 1))
                    vutils.save_image(x_rnd_ema, os.path.join(args.res_dir, '{}_RNDEMA_{}_{}{}.jpg'.format(args.gpu, epoch+1, src_idx, ref_idx)), normalize=True,
                                    nrow=(x_res_ema.size(0) // (x_src.size(0) + 2) + 1))
    """
