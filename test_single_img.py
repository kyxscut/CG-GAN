import torch
import os
from options.test_options import TestOptions
import data.lmdb_dataset as lmdb_dataset
from models import create_model
from util.visualizer import save_single_image
from util import html
from PIL import Image,ImageDraw,ImageFont

def draw(font_path,label,save_dir):
    font = ImageFont.truetype(font_path,80)
    label_w, label_h = font.getsize(label)
    img_target = Image.new('RGB', (label_w,label_h),(255,255,255))
    drawBrush = ImageDraw.Draw(img_target)
    drawBrush.text((0,0),label,fill=(0,0,0),font = font)
    img_target.save(os.path.join(save_dir,'%s_img_cont_reference.png' %label))
    return img_target


if __name__ == '__main__':    
    opt = TestOptions().parse()  
    #import pdb;pdb.set_trace()  
    transform_img = lmdb_dataset.resizeKeepRatio((opt.imgW, opt.imgH))     
    img_content = draw(opt.ttfRoot,opt.label,opt.save_dir)
    #import pdb;pdb.set_trace()
    img_content = transform_img(img_content).unsqueeze(0)
    img_style = transform_img(Image.open(opt.sty_refRoot).convert('RGB')).unsqueeze(0)
    save_dir = os.path.dirname(opt.sty_refRoot)
    data = {'A': img_style, 'B': img_content}
    model = create_model(opt)
    model.setup(opt)

    if opt.eval:
        model.eval()          
    model.set_single_input(data)
    model.test()
    visuals =model.get_current_visuals()
    #img_path = model.get_image_paths()
    print('preprocessing target image...')
    save_single_image(save_dir, opt.label,visuals,aspect_ratio=opt.aspect_ratio,width=opt.display_winsize)