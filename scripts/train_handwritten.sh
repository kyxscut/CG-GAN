set -ex
CUDA_VISIBLE_DEVICES=0 python train_handwritten.py \
--dataroot data_iam/train_372writers_addradical \
--ttfRoot data_iam/fonts_iam \
--corpusRoot data_iam/seen_char.txt \
--alphabet data_iam/seen_char.txt \
--dictionaryRoot data_iam/iam_dictionary.txt \
--name exp_handwritten \
--model handwritten \
--no_dropout \
--batch_size 16 \
--imgH 64 \
--imgW 384 \
--num_writer 372 \
--num_writer_emb 256 \
--gpu_ids 0 \
--lr 0.0001 \
--lr_decay_iters 30 \
--niter 15 \
--niter_decay 30 \
--G_ch 64 \
--D_ch 64 \
--max_length 96 \
--hidden_size 256 \
--val_num 30 \
--val_seenstyleRoot data_iam/.txt \
--val_unseenstyleRoot data_iam/.txt \



