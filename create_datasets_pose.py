import sys
import pykitti
import h5py
import argparse
import numpy as np
import transformation
import vispy.scene
from vispy.scene import visuals

parser = argparse.ArgumentParser()
parser.add_argument('--basedir', default='.', help='kitti basedir [default: .]')
parser.add_argument('--date', default='2011_09_30', help='kitti date [default: 2011_09_30]')
parser.add_argument('--drive', default='0028', help='kitti drive [default: 0028]')
parser.add_argument('--output_filename', default='kitti_velo_pose.h5', help='output file name[default: kitti_velo_pose.h5]')
FLAGS = parser.parse_args()

f = h5py.File(FLAGS.output_filename, "w")
basedir = FLAGS.basedir
date = FLAGS.date
drive = FLAGS.drive

kitti = pykitti.raw(basedir, date, drive, frames=range(0, 5001, 10))

batch_count = 0
n_pointcloud = 10000
last_velo = None
last_oxts = None

def write_batch(h5_batch, point_cloud_moving, point_cloud_fixed, relative_pose):
    h5_batch.create_dataset('point_cloud_moving', data = point_cloud_moving)
    h5_batch.create_dataset('point_cloud_fixed', data = point_cloud_fixed)
    ax, ay, az = transformation.euler_from_matrix(relative_pose[0:3,0:3], axes='sxyz')
    pose = np.zeros((6))
    pose[0:3] = relative_pose[0:3,3].flatten()
    pose[3:6] = [ax, ay, az]
    h5_batch.create_dataset('pose', data = pose)

# https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion#answer-4116899
def cart2sph(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.arctan2(xyz[:,1], xyz[:,0])
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.sqrt(xy + xyz[:,2]**2)
    return ptsnew


canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# create scatter object and fill in the data
scatter = visuals.Markers()


# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)

for velo_raw, oxts in zip(kitti.velo, kitti.oxts):
    # to (alpha, theta, range)
    sph = cart2sph(velo_raw[:,0:3])
    # drop all the lower channels
    theta_min = min(sph[:,1])
    theta_max = max(sph[:,1])
    theta_start = (theta_min + 2.0 * theta_max) / 3.0
    range_max = 10.0
    idx = np.where((sph[:,2] < range_max) & (sph[:,1] < theta_start))[0]
    # from IPython import embed; embed()

    idx_idx = np.random.permutation(len(idx))
    if len(idx) < n_pointcloud:
        print('number of points = %d, too sparse point_cloud' % len(idx))
        continue
    velo = velo_raw[idx[idx_idx[0:n_pointcloud]],0:3]
    # scatter.set_data(velo, size=5)
    # view.add(scatter)
    # view.camera = 'turntable'  # or try 'arcball'

    # vispy.app.run()
    if last_velo is None:
        last_velo = velo
        last_oxts = oxts
        continue

    T_w_velo = oxts.T_w_imu.dot(np.linalg.inv(kitti.calib.T_velo_imu))
    T_w_last_velo = last_oxts.T_w_imu.dot(np.linalg.inv(kitti.calib.T_velo_imu))

    point_cloud_moving = velo
    point_cloud_fixed = last_velo
    relative_pose = np.linalg.inv(T_w_last_velo).dot(T_w_velo) # T_fixed_moving
    print('create batch%d'%batch_count)
    h5_batch = f.create_group('batch%d'%batch_count)
    write_batch(h5_batch, point_cloud_moving, point_cloud_fixed, relative_pose)
    batch_count += 1

    point_cloud_moving = last_velo
    point_cloud_fixed = velo
    relative_pose = np.linalg.inv(T_w_velo).dot(T_w_last_velo) # T_fixed_moving
    print('create batch%d'%batch_count)
    h5_batch = f.create_group('batch%d'%batch_count)
    write_batch(h5_batch, point_cloud_moving, point_cloud_fixed, relative_pose)
    batch_count += 1

    last_velo = velo
    last_oxts = oxts

f.close()
