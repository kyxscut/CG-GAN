训练:
训练中文字体合成：
    unetD:sh scripts/train_399fonts.sh
    normD:sh scripts/train_399fonts_normD.sh
训练英文手写：
    unetD:sh scripts/train_iam.sh
    normD:sh scripts/train_iam_normD.sh

批量测试：
测试中文字体合成：sh scripts/test_399fonts.sh
测试英文手写：sh scripts/test_iam.sh

训练&测试数据集路径：
中文：data
英文（iam）：data_iam

预训练模型路径：checkpoints
