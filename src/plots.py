import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image

def plot_icvl():
    _fx = 241.42
    _fy = 241.42
    _ux = 160.
    _uy = 120.
    # # ####################################################################################################################
    # plot
    eval_prefix = 'ICVL_posereg'

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    fig2, ax2 = plt.subplots(figsize=(12, 12))


    filename = '/home/mahdi/HVR/git_repos/deep-prior-pp/data/ICVL_fake/Depth/test_seq_1/image_0000.png'  # open image
    def loadDepthMap(filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        img = Image.open(filename)  # open image
        assert len(img.getbands()) == 1  # ensure depth image
        imgdata = np.asarray(img, np.float32)

        return imgdata
    iD = loadDepthMap(filename)


    def pixel2world(x, y, z, fx, fy, ux, uy):
        w_x = (x - ux) * z / fx
        w_y = (y - uy) * z / fy
        w_z = z
        return w_x, w_y, w_z
    def depthmap2points(image, fx, fy, ux, uy):
        h, w = image.shape
        x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
        points = np.zeros((h, w, 3), dtype=np.float32)
        points[:, :, 0], points[:, :, 1], points[:, :, 2] = pixel2world(x, y, image, fx, fy, ux, uy)
        return points

    iD_xyz = depthmap2points(iD, fx=_fx, fy=_fy, ux=_ux, uy=_uy)
    _input_points_xyz = iD_xyz.reshape(iD_xyz.shape[0]*iD_xyz.shape[1],3)
    input_points_xyz = _input_points_xyz[np.logical_and(_input_points_xyz[:, 2] < 500, 0 < _input_points_xyz[:, 2]), :]
    ax.scatter(input_points_xyz[:, 0], input_points_xyz[:, 1], input_points_xyz[:, 2], marker="o", s=.05,
               label='depth map')
    ax2.scatter(input_points_xyz[:, 0], input_points_xyz[:, 1], marker="o", s=.05,
               label='depth map')

    test_id = 0
    joints3D = np.load('/home/mahdi/HVR/git_repos/deep-prior-pp/src/eval/{}/joint_{}_{}.npy'.format(eval_prefix, test_id, eval_prefix))
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
    ax.set_title('DP++ hand joint estimator: ICVL')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    ax2.set_title('DP++ hand joint estimator: ICVL')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    # plt.savefig('/home/mahdi/HVR/git_repos/deep-prior-pp/src/cache/NYU_3Djoints_{}'.format(test_id))
    plt.show()
    plt.close()
    ####################################################################################################################
    return 1


def plot_nyu():
    _fx = 588.03
    _fy = 587.07
    _ux = 320.
    _uy = 240.
    # # ####################################################################################################################
    # plot
    eval_prefix = 'NYU_posereg'

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

    iD_xyz = depthmap2points(iD, fx=_fx, fy=_fy, ux=_ux, uy=_uy)
    _input_points_xyz = iD_xyz.reshape(iD_xyz.shape[0]*iD_xyz.shape[1],3)
    input_points_xyz = _input_points_xyz[np.logical_and(_input_points_xyz[:, 2] < 850, 700 < _input_points_xyz[:, 2]), :]
    ax.scatter(input_points_xyz[:, 0], input_points_xyz[:, 1], input_points_xyz[:, 2], marker="o", s=.05,
               label='depth map')
    ax2.scatter(input_points_xyz[:, 0], input_points_xyz[:, 1], marker="o", s=.05,
               label='depth map')



    test_id = 0
    joints3D = np.load('/home/mahdi/HVR/git_repos/deep-prior-pp/src/eval/{}/joint_{}_{}.npy'.format(eval_prefix, test_id, eval_prefix))

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
    plot_icvl()