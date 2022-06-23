set -ex
CUDA_VISIBLE_DEVICES=0 python test_single_img.py \
--ttfRoot data_iam/fonts_iam/CascadiaCodePL-SemiLight.ttf \
--save_dir images_iam \
--sty_refRoot images_iam/img_sty_reference.png \
--label huawei \
--name exp_handwritten \
--model handwritten \
--no_dropout \
--batch_size 1 \
--imgH 64 \
--imgW 384 \
--gpu_ids 0 \
--epoch latest \
--G_ch 64 \
