def plot_icvl():
    import cPickle
    import numpy as np
    from PIL import Image

    # plot input depth map with estimated joints
    test_id = 0 # id of test frame to plot
    eval_prefix = 'ICVL_COM_AUGMENT'

    keypoints_test = cPickle.load( open( "/home/mahdi/HVR/git_repos/deep-prior-pp/src/eval/ICVL_COM_AUGMENT/result_main_icvl_com_refine.py_ICVL_COM_AUGMENT.pkl", "rb" ) )
    keypoints_test = np.asarray(keypoints_test).squeeze()


    def pixel2world(x, y, z, img_width, img_height, _fx, _fy, _cx, _cy):
        w_x = (x - _cx) * z / _fx
        w_y = (y - _cy) * z / _fy
        w_z = z
        return w_x, w_y, w_z


    def depthmap2points(image, _fx, _fy, _cx, _cy):
        h, w = image.shape
        y, x = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
        points = np.zeros((h, w, 3), dtype=np.float32)
        points[:, :, 0], points[:, :, 1], points[:, :, 2] = pixel2world(x, y, image, w, h, _fx, _fy, _cx, _cy)
        return points

    depth_raw = np.asarray(Image.open('/home/mahdi/HVR/git_repos/deep-prior-pp/data/ICVL/Depth/test_seq_1/image_0000.png'))
    # ICVL calibration
    h = np.asarray(depth_raw).shape[0]  # 240
    w = np.asarray(depth_raw).shape[1]  # 320
    iw = w
    ih = h
    xscale = h / ih
    yscale = w / iw
    _fx = 241.42 * xscale
    _fy = 241.42 * yscale
    _cx = 160. * xscale
    _cy = 120. * yscale

    input_points_xyz = depthmap2points(depth_raw, _fx, _fy, _cx, _cy)
    input_points_xyz = input_points_xyz.reshape(input_points_xyz.shape[0] * input_points_xyz.shape[1], 3)
    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')

    ax.scatter(input_points_xyz[:, 0], input_points_xyz[:, 1], input_points_xyz[:, 2], marker="o", s=.01, label='depth map')
    # keypoints_test = np.squeeze(keypoints_test)
    ax.scatter(keypoints_test[test_id, 0], keypoints_test[test_id, 1], keypoints_test[test_id, 2], marker="x", c='red', s=10, label='estimated hand joints')
    # ax.scatter(input_center[0], input_center[1], input_center[2], marker="X", c='black', s=10, label='refined hand center')
    ax.view_init(elev=90, azim=0)
    ax.set_title('V2V hand joint estimator: MSRA15')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig('./eval/{}/pcl_joints_{}'.format(eval_prefix, test_id))
    ax.view_init(elev=0, azim=0)
    plt.show()

    plt.close()
    print('All done ..')
    return 1

def plot_nyu():
    # # ####################################################################################################################
    # plot
    import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('Agg')  # plot to file
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from PIL import Image
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    fig2, ax2 = plt.subplots(figsize=(12, 12))


    filename = '/home/mahdi/HVR/git_repos/deep-prior-pp/data/NYU_fake/test_1/depth_1_0000001.png'  # open image
    def loadDepthMap(filename):
        """
        Read a depth-map from png raw data of NYU
        :param filename: file name to load
        :return: image data of depth image
        """
        img = Image.open(filename)
        # top 8 bits of depth are packed into green channel and lower 8 bits into blue
        assert len(img.getbands()) == 3
        r, g, b = img.split()
        r = np.asarray(r, np.int32)
        g = np.asarray(g, np.int32)
        b = np.asarray(b, np.int32)
        dpt = np.bitwise_or(np.left_shift(g, 8), b)
        imgdata = np.asarray(dpt, np.float32)

        return imgdata
    iD = loadDepthMap(filename)


    def pixel2world(x, y, z, fx, fy, ux, uy):
        w_x = (x - ux) * z / fx
        w_y = (-y + uy) * z / fy
        w_z = z
        return w_x, w_y, w_z
    def depthmap2points(image, fx, fy, ux, uy):
        h, w = image.shape
        x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
        points = np.zeros((h, w, 3), dtype=np.float32)
        points[:, :, 0], points[:, :, 1], points[:, :, 2] = pixel2world(x, y, image, fx, fy, ux, uy)
        return points

    iD_xyz = depthmap2points(iD, fx=588.03, fy=587.07, ux=320., uy=240.)
    _input_points_xyz = iD_xyz.reshape(iD_xyz.shape[0]*iD_xyz.shape[1],3)
    input_points_xyz = _input_points_xyz[np.logical_and(_input_points_xyz[:, 2] < 850, 700 < _input_points_xyz[:, 2]), :]
    ax.scatter(input_points_xyz[:, 0], input_points_xyz[:, 1], input_points_xyz[:, 2], marker="o", s=.05,
               label='depth map')
    ax2.scatter(input_points_xyz[:, 0], input_points_xyz[:, 1], marker="o", s=.05,
               label='depth map')



    test_id = 0
    joints3D = np.load('/home/mahdi/HVR/git_repos/deep-prior-pp/src/cache/joint_{}.npy'.format(test_id))
    # def world2pixel(x, y, z, fx, fy, ux, uy):
    #     p_x = x * fx / z + ux
    #     p_y = y * fy / z + uy
    #     return p_x, p_y
    # def transform_resize_crop(M, xc, yc):
    #     '''
    #     transform from cropped-resized frame to the original 320by240 frame
    #     '''
    #     v = np.array([xc, yc, 1])
    #     invM = np.linalg.inv(M)
    #     cdd = np.inner(invM[2, :], v)
    #     x = np.inner(invM[0, :], v) * cdd
    #     y = np.inner(invM[1, :], v) * cdd
    #     return x, y
    # T = np.load('/home/mahdi/HVR/git_repos/deep-prior-pp/src/cache/T_{}.npy'.format(test_id))


    keypoints_test = joints3D
    ax.scatter(keypoints_test[test_id, :, 0], keypoints_test[test_id, :, 1], keypoints_test[test_id, :, 2], marker="o",
               c='red', s=50, label='estimated hand joints')
    ax2.scatter(keypoints_test[test_id, :, 0], keypoints_test[test_id, :, 1], marker="o",
               c='red', s=50, label='estimated hand joints')


    # ax.scatter(input_center[0], input_center[1], input_center[2], marker="X", c='black', s=10, label='refined hand center')
    # ax.view_init(elev=-90, azim=0)
    ax.set_title('DP++ hand joint estimator: NYU')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    ax2.set_title('DP++ hand joint estimator: NYU')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    # plt.savefig('/home/mahdi/HVR/git_repos/deep-prior-pp/src/cache/NYU_3Djoints_{}'.format(test_id))
    plt.show()
    plt.close()
    ####################################################################################################################
    return 1

if __name__=='__main__':
    plot_nyu()