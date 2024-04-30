import os, sys
import cv2, imageio
import mayavi.mlab as mlab
import numpy as np

colors = np.array(
    [
        [0, 0, 0, 255],
        [255, 120, 50, 255],  # barrier              orangey
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green
        [0, 255, 127, 255],  # ego car              dark cyan
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ]
).astype(np.uint8)

is_wild = True
voxel_size = 0.5
pc_range = [-50, -50,  -5, 50, 50, 3]

scene_token = sys.argv[2] 
visual_path = sys.argv[1]
print("scene_token: ", scene_token)
print("visual_path: ", visual_path)

file_list = os.listdir(visual_path)
print("fil_list: ", file_list)
if 'result.mp4' in file_list:
    file_num = len(file_list) - 1
else:
    file_num = len(file_list)
file_num = 0
for file_name in file_list:
    if len(file_name.split('_')[1]) < 4:
        file_num += 1
print('!!!!!!', scene_token, file_num)
mlab.options.offscreen = True
f = 0.0055



for i in range(file_num):
    #print(i)
    fov_voxels = np.load(os.path.join(visual_path, scene_token + str(i), 'pred.npy'))
    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
    fov_voxels[:, 0] += pc_range[0]
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]

    pose = np.load(os.path.join(visual_path, scene_token + '_' + str(i), 'pose.npy'))
    os.makedirs(os.path.join(visual_path, scene_token + '_' + str(i), 'mayavi'), exist_ok=True)
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # pdb.set_trace()
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        # colormap=colors,
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        # mode="point",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )
    #plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    #plt_plot_fov.module_manager.scalar_lut_manager.lut.number_of_colors = 17

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    scene = figure.scene
   
    view_angles = [50, 35, 35, 60, 35, 35]
    
    for j in range(6):
        cam_position = (np.linalg.inv(pose[j]) @ np.array([0., 0., 0., 1.]).reshape([4, 1])).flatten()[:3]
        focal_position = (np.linalg.inv(pose[j]) @ np.array([0., 0., f, 1.]).reshape([4, 1])).flatten()[:3]

        if not is_wild:
            scene.camera.position = cam_position - np.array([0, 0, -1])
            scene.camera.focal_point = focal_position - np.array([0, 0, -1])
            scene.camera.view_angle = view_angles[j]
            scene.camera.view_up = [0.0, 0.0, 1]
            scene.camera.clipping_range = [0.01, 300.]
            scene.camera.compute_view_plane_normal()
            scene.render()
            #mlab.show()

        mlab.savefig(os.path.join(visual_path, scene_token + '_' + str(i), 'mayavi', str(j) + '.png'))
    mlab.close(all=True)
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    x = np.arange(-1, 1.1, voxel_size)
    y = np.arange(-1, 1.1, voxel_size)
    z = np.arange(0, 1.1, voxel_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    ego_car = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    ego_car = np.concatenate([ego_car, np.zeros((ego_car.shape[0], 1))], axis=-1)
    fov_voxels = np.concatenate([fov_voxels[..., :4], ego_car], axis=0)
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        # colormap=colors,
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        # mode="point",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    scene = figure.scene


    scene.camera.position = [0.75131739, -35.08337438, 16.71378558]
    scene.camera.focal_point = [0.75131739, -34.21734897, 16.21378558]
    scene.camera.view_angle = 45.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [0.01, 300.]
    scene.camera.compute_view_plane_normal()
    scene.render()
    mlab.savefig(os.path.join(visual_path, scene_token + '_' + str(i), 'mayavi', 'front.png'))

    scene.camera.position = [0.75131739, 0.78265103, 93.21378558]
    scene.camera.focal_point = [0.75131739, 0.78265103, 92.21378558]
    scene.camera.view_angle = 55.0
    scene.camera.view_up = [1., 0., 0.]

    scene.camera.clipping_range = [0.01, 400.]
    scene.camera.compute_view_plane_normal()
    scene.render()
    mlab.savefig(os.path.join(visual_path, scene_token + '_' + str(i), 'mayavi', 'top.png'))


    # scene = figure.scene
    # scene.camera.position = [0.75131739, -35.08337438, 16.71378558]
    # scene.camera.focal_point = [0.75131739, -34.21734897, 16.21378558]
    # scene.camera.view_angle = 40.0
    # scene.camera.view_up = [0.0, 0.0, 1.0]
    # scene.camera.clipping_range = [0.01, 300.]
    # scene.camera.compute_view_plane_normal()
    # scene.render()

    mlab.close(all=True)


cam_H, cam_W = 270, 480
space = 10
front_H, front_W = 1080, 1920
top_H, top_W = 1080, 1000

imgs = np.zeros((file_num, 820, 1465, 3))

for i in range(file_num):
    rgb_all = np.zeros((1640, 2930, 3))

    rgbs = []
    for j in range(6):
        rgb = cv2.imread(os.path.join(visual_path, scene_token + '_' + str(i), '{}.jpg'.format(j)))
        rgb = cv2.resize(rgb, (cam_W, cam_H))
        rgbs.append(rgb)
    rgb_all[:cam_H, :cam_W] = rgbs[2]
    rgb_all[:cam_H, cam_W + space:2*cam_W + space] = rgbs[0]
    rgb_all[:cam_H, 2*cam_W + 2*space:3*cam_W + 2*space] = rgbs[1]
    rgb_all[cam_H + space:2*cam_H + space, :cam_W] = rgbs[4][:, ::-1]
    
    rgb_all[cam_H + space:2*cam_H + space, cam_W + space:2*cam_W + space] = rgbs[3][:, ::-1]
    rgb_all[cam_H + space:2*cam_H + space, 2*cam_W + 2*space:3*cam_W + 2*space] = rgbs[5][:, ::-1]

    rgbs = []
    for j in range(6):
        rgb = cv2.imread(os.path.join(visual_path, scene_token + '_' + str(i), 'mayavi', '{}.png'.format(j)))
        rgb = cv2.resize(rgb, (cam_W, cam_H))
        rgbs.append(rgb)
    rgb_all[:cam_H, 3*cam_W + 3*space:4*cam_W + 3*space] = rgbs[2]
    rgb_all[:cam_H, 4*cam_W + 4*space:5*cam_W + 4*space] = rgbs[0]
    
    rgb_all[:cam_H, 5*cam_W + 5*space:6*cam_W + 5*space] = rgbs[1]
    rgb_all[cam_H + space:2*cam_H + space, 3*cam_W + 3*space:4*cam_W + 3*space] = rgbs[4][:, ::-1]
    
    rgb_all[cam_H + space:2*cam_H + space, 4*cam_W + 4*space:5*cam_W + 4*space] = rgbs[3][:, ::-1]
    rgb_all[cam_H + space:2*cam_H + space, 5*cam_W + 5*space:6*cam_W + 5*space] = rgbs[5][:, ::-1]

    front = cv2.imread(os.path.join(visual_path, scene_token + '_' + str(i), 'mayavi', 'front.png'))
    front = cv2.resize(front, (front_W, front_H))
    rgb_all[2*cam_H + 2*space:2*cam_H + 2*space + front_H, :front_W] = front

    top = cv2.imread(os.path.join(visual_path, scene_token + '_' + str(i), 'mayavi', 'top.png'))
    top = top[:, 550:2560-550]
    top = cv2.resize(top, (top_W, top_H))
    rgb_all[2*cam_H + 2*space:2*cam_H + 2*space + front_H, front_W + space:] = top

    rgb_all = cv2.resize(rgb_all, (1465, 820))
    imgs[i] = rgb_all


imageio.mimwrite(os.path.join(visual_path, 'result.mp4'), imgs[..., ::-1].astype(np.uint8), fps=10, quality=4)

