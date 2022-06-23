set -ex
CUDA_VISIBLE_DEVICES=1 python test_handwritten.py \
--dataroot data_iam/train_372writers_addradical \
--ttfRoot data_iam/fonts_iam \
--corpusRoot data_iam/unseen_char.txt \
--alphabet data_iam/unseen_char.txt \
--name exp_handwritten \
--state seenstyle_oov \
--model handwritten \
--no_dropout \
--batch_size 1 \
--imgH 64 \
--imgW 384 \
--gpu_ids 0 \
--G_ch 64 \
--num_test 10 \
--epoch latest \





