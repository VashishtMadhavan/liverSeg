import os

work_dir = "../runs/weight2/"
if not os.path.exists(work_dir):
    os.mkdir(work_dir)

train_list = "/x/vashishtm/data/medical/train_data.txt"
train_label = "/x/vashishtm/data/medical/train_labels.txt"

test_list = "/x/vashishtm/data/medical/val_data.txt"
test_label = "/x/vashishtm/data/medical/val_labels.txt"

classes = 2
lr = 0.0005
gpu = 0

weights = "/x/vashishtm/caffemodels/vgg_conv.caffemodel"
caffe_dir = "/home/vashishtm/caffe-dilation/build_master/tools/caffe"
ratio_file = "/home/vashishtm/liverSeg/src/ratio.txt"
log_file = work_dir + "train.log"

runstring = """python ../dilation/train.py frontend \
--work_dir %s \
--train_image %s \
--train_label %s \
--test_image %s \
--test_label %s \
--train_batch 14 \
--test_batch 2 \
--caffe %s \
--weights %s \
--crop_size 500 \
--classes %s \
--lr %s \
--gpu %s \
--ratio_file %s \
--momentum 0.9 2>&1 | tee -a %s""" % (work_dir, train_list, train_label, test_list, test_label,caffe_dir, weights, str(classes), str(lr), str(gpu), ratio_file, log_file)

os.system(runstring)
