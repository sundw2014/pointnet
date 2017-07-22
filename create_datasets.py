import pykitti
import h5py
import argparse
import numpy as np

# https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion#answer-4116899
def cart2sph(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.arctan2(xyz[:,1], xyz[:,0])
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.sqrt(xy + xyz[:,2]**2)
    return ptsnew

parser = argparse.ArgumentParser()
parser.add_argument('--basedir', default='.', help='kitti basedir [default: .]')
parser.add_argument('--date', default='2011_09_30', help='kitti date [default: 2011_09_30]')
parser.add_argument('--drive', default='0028', help='kitti drive [default: 0028]')
parser.add_argument('--output_filename', default='kitti_velo_patches.h5', help='output file name[default: kitti_velo_patches.h5]')
FLAGS = parser.parse_args()

f = h5py.File(FLAGS.output_filename, "w")
basedir = FLAGS.basedir
date = FLAGS.date
drive = FLAGS.drive

data = pykitti.raw(basedir, date, drive, frames=range(0, 5001, 10))

velo = data.velo

frame_count = 0
n_patch = 30 # extract 10 patches from every frame
alpha_window_stride = 2.0 * np.pi / n_patch
# alpha_window = alpha_window_stride = 2 * np.pi / n_patch # patch window size
n_pointcloud = 2048
n_querypoints = 128

for v in velo:
    # to (alpha, theta, range)
    sph = cart2sph(v[:,0:3])
    # drop all the lower channels
    theta_min = min(sph[:,1])
    theta_max = max(sph[:,1])
    theta_start = (theta_min + theta_max) / 2.0

    h5_frame = f.create_group('frame%d'%frame_count)
    print('create frame%d'%frame_count)
    frame_count += 1

    for i in range(n_patch):
        alpha_min = -1.0 * np.pi + alpha_window_stride * i
        alpha_max = alpha_min + alpha_window_stride
        idx = np.where(
            ((sph[:,0] < alpha_max) &
            (sph[:,0] > alpha_min) &
            (sph[:,1] < theta_start)))
        # from IPython import embed; embed()
        patch = sph[idx, :]
        patch=patch.squeeze()
        idx = np.random.permutation(patch.shape[0])
        print('downsample ratio: %f'%(float(n_pointcloud)/idx.shape[0]))
        if idx.shape[0] < (n_pointcloud + n_querypoints):
            print('idx = %d, too sparse point_cloud' % idx.shape[0])
            continue
        pl = patch[idx[0:n_pointcloud],:]
        query = patch[idx[n_pointcloud:n_pointcloud+n_querypoints],:]
        h5_patch = h5_frame.create_group('patch%d'%i)
        h5_patch.create_dataset('pl', data = pl)
        h5_patch.create_dataset('query', data = query)
f.close()
