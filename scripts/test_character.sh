set -ex
CUDA_VISIBLE_DEVICES=0 python test.py \
--dataroot data/train_set \
--ttfRoot data/font \
--corpusRoot data/char_unseen_set.txt \
--alphabet data/char_unseen_set.txt \
--name exp \
--state unseenstyle_oov \
--model character \
--no_dropout \
--batch_size 1 \
--imgH 128 \
--imgW 128 \
--gpu_ids 0 \
--num_test 10 \
--epoch latest \
--G_ch 64 \
