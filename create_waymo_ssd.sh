cd /localdata_ssd/
mkdir waymo_ssd
cd waymo_ssd
ln -s  /public/MARS/datasets/waymo_v1.3.1_untar/waymo_dev1x/waymo_format/
mkdir kitti_format
cd kitti_format
ln -s  /public/MARS/datasets/waymo_v1.3.1_untar/waymo_dev1x/kitti_format/testing
cp  /public/MARS/datasets/waymo_v1.3.1_untar/waymo_dev1x/kitti_format/*.pkl ./
cp -r  /public/MARS/datasets/waymo_v1.3.1_untar/waymo_dev1x/kitti_format/ImageSets/ ./
echo copying training data....
cp -r  /public/MARS/datasets/waymo_v1.3.1_untar/waymo_dev1x/kitti_format/training/ ./
