import os

work_dir = "../runs/context_test/"
if not os.path.exists(work_dir):
    os.mkdir(work_dir)

images = "/x/vashishtm/data/medical/test_data.txt"
bin_file = '~/liverSeg/runs/baseline_test/test_bin.txt'
weights = "../runs/baseline_context/snapshots/context_iter_24000.caffemodel"
classes = 2
gpu = 1

runstring = """python ../dilation/test.py context \
--work_dir %s \
--image_list %s \
--bin_list %s \
--gpu %s \
--weights %s \
--classes %s
""" %(work_dir, images, bin_file, str(gpu), weights, classes)

os.system(runstring)
