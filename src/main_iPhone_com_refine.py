"""This is the main file for training hand joint classifier on MSRA dataset

Copyright 2015 Markus Oberweger, ICG,
Graz University of Technology <oberweger@icg.tugraz.at>

This file is part of DeepPrior.

DeepPrior is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DeepPrior is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with DeepPrior.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy
import gc
import matplotlib


import os
os.environ["THEANO_FLAGS"] = "device=cuda,floatX=float32"

# matplotlib.use('Agg')  # plot to file
# import matplotlib.pyplot as plt
from net.scalenet import ScaleNetParams, ScaleNet
from trainer.scalenettrainer import ScaleNetTrainerParams, ScaleNetTrainer
from util.handdetector import HandDetector
import os
import cPickle
from data.importers import iPhoneImporter
from data.dataset import iPhoneDataset
from util.handpose_evaluation import MSRAHandposeEvaluation
from util.helpers import shuffle_many_inplace
from PIL import Image
import open3d as o3d


if __name__ == '__main__':

    eval_prefix = 'iPhone_COM_AUGMENT'
    if not os.path.exists('./eval/'+eval_prefix+'/'):
        os.makedirs('./eval/'+eval_prefix+'/')

    rng = numpy.random.RandomState(23455)

    print("create data")
    aug_modes = ['com', 'rot', 'none']  # 'sc',

    di = iPhoneImporter('../data/iPhone/')
    # Seq0_1 = di.loadSequence('P0', shuffle=True, rng=rng, docom=False)
    # Seq0_1 = Seq0_1._replace(name='P0_gt')
    # Seq0_2 = di.loadSequence('P0', shuffle=True, rng=rng, docom=True)
    # Seq0_2 = Seq0_2._replace(name='P0_com')
    # Seq1_1 = di.loadSequence('P1', shuffle=True, rng=rng, docom=False)
    # Seq1_1 = Seq1_1._replace(name='P1_gt')
    # Seq1_2 = di.loadSequence('P1', shuffle=True, rng=rng, docom=True)
    # Seq1_2 = Seq1_2._replace(name='P1_com')
    # Seq2_1 = di.loadSequence('P2', shuffle=True, rng=rng, docom=False)
    # Seq2_1 = Seq2_1._replace(name='P2_gt')
    # Seq2_2 = di.loadSequence('P2', shuffle=True, rng=rng, docom=True)
    # Seq2_2 = Seq2_2._replace(name='P2_com')
    # Seq3_1 = di.loadSequence('P3', shuffle=True, rng=rng, docom=False)
    # Seq3_1 = Seq3_1._replace(name='P3_gt')
    # Seq3_2 = di.loadSequence('P3', shuffle=True, rng=rng, docom=True)
    # Seq3_2 = Seq3_2._replace(name='P3_com')
    # Seq4_1 = di.loadSequence('P4', shuffle=True, rng=rng, docom=False)
    # Seq4_1 = Seq4_1._replace(name='P4_gt')
    # Seq4_2 = di.loadSequence('P4', shuffle=True, rng=rng, docom=True)
    # Seq4_2 = Seq4_2._replace(name='P4_com')
    # Seq5_1 = di.loadSequence('P5', shuffle=True, rng=rng, docom=False)
    # Seq5_1 = Seq5_1._replace(name='P5_gt')
    # Seq5_2 = di.loadSequence('P5', shuffle=True, rng=rng, docom=True)
    # Seq5_2 = Seq5_2._replace(name='P5_com')
    # Seq6_1 = di.loadSequence('P6', shuffle=True, rng=rng, docom=False)
    # Seq6_1 = Seq6_1._replace(name='P6_gt')
    # Seq6_2 = di.loadSequence('P6', shuffle=True, rng=rng, docom=True)
    # Seq6_2 = Seq6_2._replace(name='P6_com')
    # Seq7_1 = di.loadSequence('P7', shuffle=True, rng=rng, docom=False)
    # Seq7_1 = Seq7_1._replace(name='P7_gt')
    # Seq7_2 = di.loadSequence('P7', shuffle=True, rng=rng, docom=True)
    # Seq7_2 = Seq7_2._replace(name='P7_com')
    # Seq8_1 = di.loadSequence('P8', shuffle=True, rng=rng, docom=False)
    # Seq8_1 = Seq8_1._replace(name='P8_gt')
    # Seq8_2 = di.loadSequence('P8', shuffle=True, rng=rng, docom=True)
    # Seq8_2 = Seq8_2._replace(name='P8_com')
    # trainSeqs = [Seq0_1, Seq0_2, Seq1_1, Seq1_2, Seq2_1, Seq2_2, Seq3_1, Seq3_2,
    #              Seq4_1, Seq4_2, Seq5_1, Seq5_2, Seq6_1, Seq6_2, Seq7_1, Seq7_2,
    #              Seq8_1, Seq8_2]

    # trainSeqs = [Seq0_1]
    Seq_0 = di.loadSequence('P0', docom=True, cube=(200,200,200)) # we crop data centered in center of mass of data between maxdepth and mindepth and cube size cube in mm^3 and use it to get the resultant in UVD (do resize and ...) and feed it into the either train or test COM augmented refine net
    testSeqs = [Seq_0]

    # # create training data
    # trainDataSet = MSRA15Dataset(trainSeqs, localCache=False)
    # nSamp = numpy.sum([len(s.data) for s in trainSeqs])
    # d1, g1 = trainDataSet.imgStackDepthOnly(trainSeqs[0].name)
    # train_data = numpy.ones((nSamp, d1.shape[1], d1.shape[2], d1.shape[3]), dtype='float32')
    # train_gt3D = numpy.ones((nSamp, g1.shape[1], g1.shape[2]), dtype='float32')
    # train_data_com = numpy.ones((nSamp, 3), dtype='float32')
    # train_data_M = numpy.ones((nSamp, 3, 3), dtype='float32')
    # train_data_cube = numpy.ones((nSamp, 3), dtype='float32')
    # del d1, g1
    # gc.collect()
    # gc.collect()
    # gc.collect()
    # oldIdx = 0
    # for seq in trainSeqs:
    #     d, g = trainDataSet.imgStackDepthOnly(seq.name)
    #     train_data[oldIdx:oldIdx+d.shape[0]] = d
    #     train_gt3D[oldIdx:oldIdx+d.shape[0]] = g
    #     train_data_com[oldIdx:oldIdx+d.shape[0]] = numpy.asarray([da.com for da in seq.data])
    #     train_data_M[oldIdx:oldIdx+d.shape[0]] = numpy.asarray([da.T for da in seq.data])
    #     train_data_cube[oldIdx:oldIdx+d.shape[0]] = numpy.asarray([seq.config['cube']]*d.shape[0])
    #     oldIdx += d.shape[0]
    #     del d, g
    #     gc.collect()
    #     gc.collect()
    #     gc.collect()
    # shuffle_many_inplace([train_data, train_gt3D, train_data_com, train_data_cube, train_data_M], random_state=rng)
    #
    # mb = (train_data.nbytes) / (1024 * 1024)
    # print("data size: {}Mb".format(mb))

    testDataSet = iPhoneDataset(testSeqs)
    test_data, test_gt3D = testDataSet.imgStackDepthOnly(testSeqs[0].name, mode='iPad')

    val_data = test_data
    val_gt3D = test_gt3D

    ####################################
    # # resize data
    # dsize = (int(train_data.shape[2]//2), int(train_data.shape[3]//2))
    # xstart = int(train_data.shape[2]/2-dsize[0]/2)
    # xend = xstart + dsize[0]
    # ystart = int(train_data.shape[3]/2-dsize[1]/2)
    # yend = ystart + dsize[1]
    # train_data2 = train_data[:, :, ystart:yend, xstart:xend]
    #
    # dsize = (int(train_data.shape[2]//4), int(train_data.shape[3]//4))
    # xstart = int(train_data.shape[2]/2-dsize[0]/2)
    # xend = xstart + dsize[0]
    # ystart = int(train_data.shape[3]/2-dsize[1]/2)
    # yend = ystart + dsize[1]
    # train_data4 = train_data[:, :, ystart:yend, xstart:xend]
    #
    # dsize = (int(train_data.shape[2]//2), int(train_data.shape[3]//2))
    # xstart = int(train_data.shape[2]/2-dsize[0]/2)
    # xend = xstart + dsize[0]
    # ystart = int(train_data.shape[3]/2-dsize[1]/2)
    # yend = ystart + dsize[1]
    # val_data2 = val_data[:, :, ystart:yend, xstart:xend]
    #
    # dsize = (int(train_data.shape[2]//4), int(train_data.shape[3]//4))
    # xstart = int(train_data.shape[2]/2-dsize[0]/2)
    # xend = xstart + dsize[0]
    # ystart = int(train_data.shape[3]/2-dsize[1]/2)
    # yend = ystart + dsize[1]
    # val_data4 = val_data[:, :, ystart:yend, xstart:xend]
    #
    # dsize = (int(train_data.shape[2]//2), int(train_data.shape[3]//2))
    # xstart = int(train_data.shape[2]/2-dsize[0]/2)
    # xend = xstart + dsize[0]
    # ystart = int(train_data.shape[3]/2-dsize[1]/2)
    # yend = ystart + dsize[1]
    # test_data2 = test_data[:, :, ystart:yend, xstart:xend]
    #
    # dsize = (int(train_data.shape[2]//4), int(train_data.shape[3]//4))
    # xstart = int(train_data.shape[2]/2-dsize[0]/2)
    # xend = xstart + dsize[0]
    # ystart = int(train_data.shape[3]/2-dsize[1]/2)
    # yend = ystart + dsize[1]
    # test_data4 = test_data[:, :, ystart:yend, xstart:xend]
    #
    # print train_gt3D.max(), test_gt3D.max(), train_gt3D.min(), test_gt3D.min()
    # print train_data.max(), test_data.max(), train_data.min(), test_data.min()
    #
    # imgSizeW = train_data.shape[3]
    # imgSizeH = train_data.shape[2]
    # nChannels = train_data.shape[1]

    #############################################################################
    print("create network")
    batchSize = 64
    # poseNetParams = ScaleNetParams(type=1, nChan=nChannels, wIn=imgSizeW, hIn=imgSizeH, batchSize=batchSize,
    #                                resizeFactor=2, numJoints=1, nDims=3)
    nChannels = 1
    imgSizeW = 128
    imgSizeH = 128
    poseNetParams = ScaleNetParams(type=1, nChan=nChannels, wIn=imgSizeW, hIn=imgSizeH, batchSize=batchSize,
                                   resizeFactor=2, numJoints=1, nDims=3)
    poseNet = ScaleNet(rng, cfgParams=poseNetParams)

    # poseNetTrainerParams = ScaleNetTrainerParams()
    # poseNetTrainerParams.use_early_stopping = False
    # poseNetTrainerParams.batch_size = batchSize
    # poseNetTrainerParams.learning_rate = 0.0005
    # poseNetTrainerParams.weightreg_factor = 0.0001
    # poseNetTrainerParams.force_macrobatch_reload = True
    # poseNetTrainerParams.para_augment = True
    # poseNetTrainerParams.augment_fun_params = {'fun': 'augment_poses', 'args': {'normZeroOne': False,
    #                                                                             'di': di,
    #                                                                             'aug_modes': aug_modes,
    #                                                                             'hd': HandDetector(train_data[0, 0].copy(), abs(di.fx), abs(di.fy), importer=di)}}
    #
    # print("setup trainer")
    # poseNetTrainer = ScaleNetTrainer(poseNet, poseNetTrainerParams, rng, './eval/'+eval_prefix)
    # poseNetTrainer.setData(train_data, train_gt3D[:, di.crop_joint_idx, :], val_data, val_gt3D[:, di.crop_joint_idx, :])
    # poseNetTrainer.addStaticData({'val_data_x1': val_data2, 'val_data_x2': val_data4})
    # poseNetTrainer.addManagedData({'train_data_x1': train_data2, 'train_data_x2': train_data4})
    # poseNetTrainer.addManagedData({'train_data_com': train_data_com,
    #                                'train_data_cube': train_data_cube,
    #                                'train_data_M': train_data_M,
    #                                'train_gt3D': train_gt3D})
    # poseNetTrainer.compileFunctions()

    # ###################################################################
    # # TRAIN
    # train_res = poseNetTrainer.train(n_epochs=100)
    # train_costs = train_res[0]
    # val_errs = train_res[2]
    #
    # # plot cost
    # fig = plt.figure()
    # plt.semilogy(train_costs)
    # plt.show(block=False)
    # fig.savefig('./eval/'+eval_prefix+'/'+eval_prefix+'_cost.png')
    #
    # fig = plt.figure()
    # plt.semilogy(val_errs)
    # plt.show(block=False)
    # fig.savefig('./eval/'+eval_prefix+'/'+eval_prefix+'_errs.png')

    # # save results
    # poseNet.save("./eval/{}/net_{}.pkl".format(eval_prefix, eval_prefix))
    # poseNet.load("./eval/{}/net_{}.pkl".format(eval_prefix,eval_prefix))
    # poseNet.load("./eval/{}/net_ICVL_COM_AUGMENT.pkl".format(eval_prefix))
    poseNet.load("./eval/{}/net_MSRA15_COM_AUGMENT.pkl".format(eval_prefix))

    ####################################################
    # TEST
    print("Testing ...")
    gt3D = [j.gt3Dorig[di.crop_joint_idx].reshape(1, 3) for j in testSeqs[0].data]
    # jts = poseNet.computeOutput([test_data, test_data2, test_data4])
    jts = poseNet.computeOutput([test_data, test_data[:, :, 32:96, 32:96], test_data[:, :, 48:80, 48:80]])
    joints = []
    for i in xrange(test_data.shape[0]):
        joints.append(jts[i].reshape(1, 3)*(testSeqs[0].config['cube'][2]/2.) + testSeqs[0].data[i].com)
    print("jts = {}".format(jts))
    # 3D coordinates of the refined center = joints
    print("joints = {}".format(joints))
########################################################################################################################
    # plot
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    fig, ax = plt.subplots()
    # ax.imshow(Seq_0.data[0].dpt, cmap=matplotlib.cm.jet)
    ax.imshow(test_data.squeeze(), cmap=matplotlib.cm.jet)

    icom = np.empty((2, 1))
    icom[0] = Seq_0.data[0].gtcrop[5][0]
    icom[1] = Seq_0.data[0].gtcrop[5][1]

    ax.scatter(icom[0], icom[1], marker='+', c='yellow', s=100, label='initial center: Center of Mass to train refine net')  # initial hand com in UVD
    plt.show()
    ax.legend()
########################################################################################################################
    # iPhone calibration
    h = 240.
    w = 320.
    iw = 3088.0
    ih = 2316.0
    xscale = h / ih
    yscale = w / iw
    _fx = 2880.0796 * xscale
    _fy = 2880.0796 * yscale
    _ux = 1153.2035 * xscale
    _uy = 1546.5824 * yscale

    # temporary: must be changed ###########################################################################################
    color_raw = o3d.io.read_image('/home/mahdi/HVR/hvr/hand_pcl_iPhone/Tom_set_2/iPhone/hand30wall50_color.png')
    depth_raw = o3d.io.read_image('/home/mahdi/HVR/git_repos/deep-prior-pp/data/iPhone/P0/5/hand30wall50_depth.png')
    color_raw = o3d.Image(np.asarray(color_raw))
    rgbd_image = o3d.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=0.529, depth_trunc=30.0, convert_rgb_to_intensity=False)
    # iPhone calibration
    h = np.asarray(color_raw).shape[0]  # 480
    w = np.asarray(color_raw).shape[1]  # 640
    iw = 3088.0
    ih = 2316.0
    xscale = h / ih
    yscale = w / iw
    __fx = 2880.0796 * xscale
    __fy = 2880.0796 * yscale
    # _cx = 1546.5824 * xscale
    # _cy = 1153.2035 * yscale
    __cx = 1153.2035 * xscale
    __cy = 1546.5824 * yscale
    setIntrinsic = o3d.camera.PinholeCameraIntrinsic()
    setIntrinsic.set_intrinsics(width=w, height=h, fx=__fx, fy=__fy, cx=__cx, cy=__cy)
    pcd = o3d.PointCloud.create_from_rgbd_image(
        rgbd_image,
        setIntrinsic)
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    z_values = (-np.asarray(pcd.points)[:, 2] * 1000)  # in mm
    depth_map = np.reshape(z_values, (480, 640))
    imgdata = np.asarray(Image.fromarray(depth_map).resize((320, 240)))
# temporary: must be changed ###########################################################################################
    fig = plt.figure(figsize=(12, 12))
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.gca(projection='3d')
    def pixel2world(x, y, z, fx, fy, ux, uy):
        w_x = (x - ux) * z / fx
        w_y = -(-y + uy) * z / fy
        w_z = z
        return w_x, w_y, w_z
    def depthmap2points(image, fx, fy, ux, uy):
        h, w = image.shape
        x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
        points = np.zeros((h, w, 3), dtype=np.float32)
        points[:, :, 0], points[:, :, 1], points[:, :, 2] = pixel2world(x, y, image, fx, fy, ux, uy)
        return points
    iD_xyz = depthmap2points(imgdata, fx=_fx, fy=_fy, ux=_ux, uy=_uy)
    _input_points_xyz = iD_xyz.reshape(iD_xyz.shape[0]*iD_xyz.shape[1],3)
    input_points_xyz = _input_points_xyz[np.logical_and(_input_points_xyz[:, 2] < 340, -np.inf < _input_points_xyz[:, 2]), :]
    ax.scatter(input_points_xyz[:, 0], input_points_xyz[:, 1], input_points_xyz[:, 2], marker="o", s=.05,
               label='depth map')
    # Seq_0.data[0].com in 3D is calculated hand center of mass to get cropped cube to feed into the refinenet to let it predict in UVD of 320by240 pixels
    # gt_com = np.empty((2, 1))
    gt_com3D = Seq_0.data[0].com
    # gt_com[0] = gt_com3D[0] / gt_com3D[2] * _fx + _ux
    # gt_com[1] = gt_com3D[1] / gt_com3D[2] * _fy + _uy
    # # transformPoints2D(gt_com, M=Seq_0.data[0].T)
    ax.scatter(gt_com3D[0], gt_com3D[1], gt_com3D[2], marker='x', c='m', s=150, label='calculated hand center of mass')  # 'calculated hand center of mass to get cropped cube to >feed into the refinenet to let it predict' ; initial hand com in IMG

    # refined_com = np.empty((2, 1))
    refined_com3D = joints[0][0] # estimated joints in 3D
    # refined_com[0] = refined_com3D[0] / refined_com3D[2] * _fx + _ux
    # refined_com[1] = refined_com3D[1] / refined_com3D[2] * _fy + _uy
    ax.scatter(refined_com3D[0], refined_com3D[1], refined_com3D[2], marker='*', c='lime', s=150, label='refined hand center refinenet estimation')  # initial hand com in IMG
    # ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    # plt.savefig('/home/mahdi/HVR/git_repos/deep-prior-pp/src/cache/iPhone_30hand50wall.png')
    ax.legend()
    plt.show()
    plt.close()

# start save 3D joint data for 320*240 frame ###########################################################################
    # iPhone calibration
    _h = 128.
    _w = 128.
    _iw = 3088.0
    _ih = 2316.0
    xscale = _h / _ih
    yscale = _w / _iw
    _fx = 2880.0796 * xscale
    _fy = 2880.0796 * yscale
    _ux = 1153.2035 * xscale
    _uy = 1546.5824 * yscale
    _refined_com = np.empty((2, 1))
    _refined_com3D = joints[0][0] # joints in 3D for cropped(200 by 200) then resized (128by128) frame of DPpp
    _refined_com[0] = _refined_com3D[0] / _refined_com3D[2] * _fx + _ux
    _refined_com[1] = _refined_com3D[1] / _refined_com3D[2] * _fy + _uy
    def transform_resize_crop(M, xc, yc):
        '''
        transform from cropped-resized frame to the original 320by240 frame
        '''
        v = np.array([xc, yc, 1])
        invM = np.linalg.inv(M)
        cdd = np.inner(invM[2, :], v)
        x = np.inner(invM[0, :], v) * cdd
        y = np.inner(invM[1, :], v) * cdd
        return x, y

    refined_com_x, refined_com_y = transform_resize_crop(M=Seq_0.data[0].T, xc=_refined_com[0], yc=_refined_com[1])
    # iPhone calibration
    _h = 240.
    _w = 320.
    _iw = 3088.0
    _ih = 2316.0
    xscale = _h / _ih
    yscale = _w / _iw
    _fx = 2880.0796 * xscale
    _fy = 2880.0796 * yscale
    _ux = 1153.2035 * xscale
    _uy = 1546.5824 * yscale
    refined_com3D = np.empty((2, 1))
    refined_com3D[0] = (refined_com_x - _ux) * joints[0][0][2] / _fx
    refined_com3D[1] = (refined_com_y - _uy) * joints[0][0][2] / _fy
    refined_center_3D_corrected = np.append(refined_com3D, joints[0][0][2])
    np.savetxt('/home/mahdi/HVR/git_repos/deep-prior-pp/src/cache/{}_3Drefinedcom.txt'.format('iPhone_30hand50wall'), refined_center_3D_corrected, fmt='%4.12f', newline=' ')
# end save 3D joint data for 320*240 frame #############################################################################

    np.save('/home/mahdi/HVR/git_repos/deep-prior-pp/src/cache/T.npy', Seq_0.data[0].T)


    # hpe = MSRAHandposeEvaluation(gt3D, joints)
    # hpe.subfolder += '/'+eval_prefix+'/'
    # print("Mean error: {}mm, max error: {}mm".format(hpe.getMeanError(), hpe.getMaxError()))
    #
    # # save results
    # cPickle.dump(joints, open("./eval/{}/result_{}_{}.pkl".format(eval_prefix,os.path.split(__file__)[1],eval_prefix), "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
    #
    # print "Testing baseline"
    #
    # #################################
    # # BASELINE
    # com = [j.com for j in testSeqs[0].data]
    # hpe_com = MSRAHandposeEvaluation(gt3D, numpy.asarray(com).reshape((len(gt3D), 1, 3)))
    # hpe_com.subfolder += '/'+eval_prefix+'/'
    # print("Mean error: {}mm".format(hpe_com.getMeanError()))
    #
    # hpe.plotEvaluation(eval_prefix, methodName='Our regr', baseline=[('CoM', hpe_com)])
    print('ended')
