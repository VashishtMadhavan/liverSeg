import os

work_dir = "../runs/changeWeight_test/"
if not os.path.exists(work_dir):
    os.mkdir(work_dir)

images = "/x/vashishtm/data/medical/test_data.txt"
weights = "../runs/changeWeight_bin/snapshots/frontend_vgg_iter_25000.caffemodel"
classes = 2
gpu = 0

runstring = """python ../dilation/test.py frontend \
--work_dir %s \
--image_list %s \
--gpu %s \
--bin \
--weights %s \
--classes %s
""" %(work_dir, images, str(gpu), weights, classes)

os.system(runstring)
