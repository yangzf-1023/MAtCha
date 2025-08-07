import os
import argparse
import glob

import numpy as np

import math
import shutil
import struct

# returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
def closest_point_2_lines(oa, da, ob, db): 
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

if __name__ == '__main__':
    parser = argparse.ArgumentParser() # TODO: refine it.
    parser.add_argument("path", default="", help="input path to image")
    parser.add_argument("--valid_views", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20", 
                        help="list of training views, all others will be used for testing")
    args = parser.parse_args()
    args.valid_views = [int(v) for v in args.valid_views.split(',')]
    print(f'[INFO] valid views: {args.valid_views}')

    # path must end with / to make sure image path is relative
    if args.path[-1] != '/':
        args.path += '/'
        
    images_path = os.path.join(args.path, 'images')
    os.makedirs(images_path, exist_ok=True)
        
    # load data
    images = [f[len(args.path):] for f in sorted(glob.glob(os.path.join(images_path, "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
    images = [image for image in images if int(image.split('cam')[-1].split('_')[0]) in args.valid_views]
    print(images)
    cams = sorted(set([im[7:12] for im in images]))
    
    poses_bounds = np.load(os.path.join(args.path, 'poses_bounds.npy'))
    poses_bounds = poses_bounds[args.valid_views] # 假设没有缺少视角
    N = poses_bounds.shape[0]

    print(f'[INFO] loaded {len(images)} images from {len(cams)} videos, {N} poses_bounds as {poses_bounds.shape}')

    assert N == len(cams)

    poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N, 3, 5)
    bounds = poses_bounds[:, -2:] # (N, 2)

    H, W, fl = poses[0, :, -1] 

    print(f'[INFO] H = {H}, W = {W}, fl = {fl}')

    # inversion of this: https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py#L51
    poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]], -1) # (N, 3, 4)

    # to homogeneous 
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N, 1, 4)
    poses = np.concatenate([poses, last_row], axis=1) # (N, 4, 4) 

    # the following stuff are from colmap2nerf... 
    poses[:, 0:3, 1] *= -1
    poses[:, 0:3, 2] *= -1
    poses = poses[:, [1, 0, 2, 3], :] # swap y and z
    poses[:, 2, :] *= -1 # flip whole world upside down

    up = poses[:, 0:3, 1].sum(0)
    up = up / np.linalg.norm(up)
    R = rotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    poses = R @ poses

    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for i in range(N):
        mf = poses[i, :3, :]
        for j in range(i + 1, N):
            mg = poses[j, :3, :]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            #print(i, j, p, w)
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    print(f'[INFO] totp = {totp}')
    poses[:, :3, 3] -= totp

    avglen = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()

    poses[:, :3, 3] *= 4.0 / avglen

    print(f'[INFO] average radius = {avglen}')
    
    train_frames = []
    for i in range(N):
        cam_frames = [{'file_path': im.lstrip("/").split('.')[0], 
                       'transform_matrix': poses[i].tolist(),
                       'time': int(im.lstrip("/").split('.')[0][-4:]) / 30.} for im in images if cams[i] in im]
        
        train_frames += cam_frames


    train_transforms = {
        'w': W,
        'h': H,
        'fl_x': fl,
        'fl_y': fl,
        'cx': W // 2,
        'cy': H // 2,
        'frames': train_frames,
    }
    
    colmap_workspace = os.path.join(args.path, 'sparse')
    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    W, H, cx, cy, fx, fy = int(W), int(H), train_transforms['cx'], train_transforms['cy'], train_transforms['fl_x'], train_transforms['fl_y']
    os.makedirs(os.path.join(colmap_workspace, '0'), exist_ok=True)
    
    fname2pose = {}
    with open(os.path.join(colmap_workspace, '0/cameras.txt'), 'w') as file:
        file.write(f'1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}')
    
    with open(os.path.join(colmap_workspace, '0/cameras.bin'), 'wb') as f:
        f.write(struct.pack('Q', 1))  # uint64_t
        camera_id = 1
        model = 1  # PINHOLE model
        f.write(struct.pack('i', camera_id))  # int32_t
        f.write(struct.pack('i', model))      # int32_t
        f.write(struct.pack('Q', W))          # uint64_t
        f.write(struct.pack('Q', H))          # uint64_t
        f.write(struct.pack('d', fx))         # double
        f.write(struct.pack('d', fy))         # double
        f.write(struct.pack('d', cx))         # double
        f.write(struct.pack('d', cy))         # double
        
    for frame in train_frames:
        if frame['time'] == 0:
            fname = frame['file_path'].split('/')[-1] + '.png'
            pose = np.array(frame['transform_matrix']) @ blender2opencv
            fname2pose.update({fname: pose})
                
    with open(os.path.join(colmap_workspace, '0/images.txt'), 'w') as f:
        idx = 1
        for fname in fname2pose.keys():
            pose = fname2pose[fname]
            R = np.linalg.inv(pose[:3, :3])
            T = -np.matmul(R, pose[:3, 3])
            q0 = 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
            q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
            q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
            q3 = (R[1, 0] - R[0, 1]) / (4 * q0)

            f.write(f'{idx} {q0} {q1} {q2} {q3} {T[0]} {T[1]} {T[2]} 1 {fname}\n\n')
            idx += 1
    
    # 写入 images.bin
    with open(os.path.join(colmap_workspace, '0/images.bin'), 'wb') as f:
        # 写入图像数量
        f.write(struct.pack('Q', len(fname2pose)))  # uint64_t
        idx = 1
        for fname in fname2pose.keys():
            pose = fname2pose[fname]
            R = np.linalg.inv(pose[:3, :3])
            T = -np.matmul(R, pose[:3, 3])
            q0 = 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
            q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
            q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
            q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
            # 写入图像数据: image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name
            f.write(struct.pack('I', idx))         # uint32_t
            f.write(struct.pack('d', q0))          # double
            f.write(struct.pack('d', q1))          # double
            f.write(struct.pack('d', q2))          # double
            f.write(struct.pack('d', q3))          # double
            f.write(struct.pack('d', T[0]))        # double
            f.write(struct.pack('d', T[1]))        # double
            f.write(struct.pack('d', T[2]))        # double
            f.write(struct.pack('I', 1))           # uint32_t (camera_id)
            # 写入图像名称 (以 null 结尾)
            f.write(fname.encode('utf-8') + b'\0')
            # 写入空的关键点数据 (points2D)
            f.write(struct.pack('Q', 0))           # uint64_t (关键点数量为 0)
            idx += 1
    