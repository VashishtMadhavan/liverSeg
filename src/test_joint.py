import os

work_dir = "../runs/joint_combined_test/"
if not os.path.exists(work_dir):
    os.mkdir(work_dir)

images = "/x/vashishtm/data/medical/test_data.txt"
weights = "../runs/joint_combined/snapshots/joint_iter_25000.caffemodel"
classes = 2
gpu = 0

runstring = """python ../dilation/test.py joint \
--work_dir %s \
--image_list %s \
--gpu %s \
--weights %s \
--classes %s
""" %(work_dir, images, str(gpu), weights, classes)

os.system(runstring)
