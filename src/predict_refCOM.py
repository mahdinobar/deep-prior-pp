"""
"""
import os
os.environ["THEANO_FLAGS"] = "device=cuda,floatX=float32"
# matplotlib.use('Agg')  # plot to file

import numpy
import numpy as np
import matplotlib.pyplot as plt
from net.scalenet import ScaleNetParams, ScaleNet
import os
from data.importers import iPadImporter
from data.dataset import iPadDataset
from PIL import Image


def refCOM(depth, args=None, refineNet='/home/mahdi/HVR/git_repos/deep-prior-pp/src/eval/iPad_COM_AUGMENT/net_ICVL_COM_AUGMENT.pkl', device='iPad', minDepth=None, maxDepth=None):
    if device =='iPad':
        eval_prefix = 'iPad_COM_AUGMENT'
        if not os.path.exists('./eval/'+eval_prefix+'/'):
            os.makedirs('./eval/'+eval_prefix+'/')

        rng = numpy.random.RandomState(23455)

        print("create data")
        di = iPadImporter(depth, args, useCache=False)
        Seq_0 = di.loadSequence_predict(docom=True, cube=(250,250,250), minDepth=minDepth, maxDepth=maxDepth) # we crop data centered in center of mass of data between maxdepth and mindepth and cube size cube in mm^3 and use it to get the resultant in UVD (do resize and ...) and feed it into the either train or test COM augmented refine net
        testSeqs = [Seq_0]
        testDataSet = iPadDataset(testSeqs)
        test_data = testDataSet.imgStackDepthOnly(seqName=testSeqs, mode='iPad', vis=args.vis)

        #############################################################################
        print("create network")
        batchSize = 64

        nChannels = 1
        imgSizeW = 128
        imgSizeH = 128
        poseNetParams = ScaleNetParams(type=1, nChan=nChannels, wIn=imgSizeW, hIn=imgSizeH, batchSize=batchSize,
                                       resizeFactor=2, numJoints=1, nDims=3)
        poseNet = ScaleNet(rng, cfgParams=poseNetParams)

        poseNet.load(refineNet)

        ####################################################
        # TEST
        print("Testing ...")
        jts = poseNet.computeOutput([test_data, test_data[:, :, 32:96, 32:96], test_data[:, :, 48:80, 48:80]])
        joints = []
        for i in range(test_data.shape[0]):
            joints.append(jts[i].reshape(1, 3)*(testSeqs[0].config['cube'][2]/2.) + testSeqs[0].data[i].com)

        if args.vis == True:
            # iPad calibration
            _h = 240.
            _w = 320.
            iw = 3088.0
            ih = 2316.0
            xscale = _h / ih
            yscale = _w / iw
            _fx = 2883.24 * xscale
            _fy = 2883.24 * yscale
            _ux = 1154.66 * xscale
            _uy = 1536.17 * yscale

        # temporary: must be changed ##########################################################################################
            iDepth = np.loadtxt(args.iD, dtype=np.float32) * 1000  # mm
            iDepth = np.asarray(Image.fromarray(iDepth).resize((320, 240)))
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
            iD_xyz = depthmap2points(iDepth, fx=_fx, fy=_fy, ux=_ux, uy=_uy)
            _input_points_xyz = iD_xyz.reshape(iD_xyz.shape[0]*iD_xyz.shape[1],3)
            input_points_xyz = _input_points_xyz[np.logical_and(_input_points_xyz[:, 2] < 1000, -np.inf < _input_points_xyz[:, 2]), :]
            ax.scatter(input_points_xyz[:, 0], input_points_xyz[:, 1], input_points_xyz[:, 2], marker="o", s=.3,
                       label='depth map')
            # Seq_0.data[0].com in 3D is calculated hand center of mass to get cropped cube to feed into the refinenet to let it predict in UVD of 320by240 pixels
            gt_com3D = Seq_0.data[0].com
            ax.scatter(gt_com3D[0], gt_com3D[1], gt_com3D[2], marker='x', c='m', s=150, label='calculated hand center of mass')  # 'calculated hand center of mass to get cropped cube to >feed into the refinenet to let it predict' ; initial hand com in IMG
            # refined_com = np.empty((2, 1))
            refined_com3D = joints[0][0] # estimated joints in 3D
            ax.scatter(refined_com3D[0], refined_com3D[1], refined_com3D[2], marker='*', c='lime', s=150, label='refined hand center refinenet estimation')  # initial hand com in IMG
            ax.legend()
            plt.show()
            # np.save("./eval/{}/refined_com3D.npy".format(eval_prefix), refined_com3D)
        return joints[0][0]


def refCOM_multi(depth, args=None, refineNet='/home/mahdi/HVR/git_repos/deep-prior-pp/src/eval/iPad_COM_AUGMENT/net_ICVL_COM_AUGMENT.pkl', device='iPad', minDepth=None, maxDepth=None):
    if device =='iPad':
        eval_prefix = 'iPad_COM_AUGMENT'
        if not os.path.exists('./eval/'+eval_prefix+'/'):
            os.makedirs('./eval/'+eval_prefix+'/')

        rng = numpy.random.RandomState(23455)

        S = 1
        D = 15
        O = 1
        N = 2
        repetition = 1
        finger_width_est = np.zeros((S, D, O, N, repetition))
        Time = np.zeros((S, D, O, N, repetition))
        number_hand_points = np.zeros((S, D, O, N, repetition))
        number_points_green_region = np.zeros((S, D, O, N, repetition))
        mean_hand_depth = np.zeros((S, D, O, N, repetition))
        counter = 1
        test_data = []
        com_data = []
        for s in range(1, S + 1):
            for d in range(15, D + 5, 5):
                for o in range(1, O + 1):
                    for n in range(1, N + 1):
                        print("create data")
                        minDepth = max(d*10 - 200, 10)
                        maxDepth = d*10 + 200
                        args.minDepth = max(d*10 - 200, 10)
                        args.maxDepth = d*10 + 200
                        depth = np.loadtxt('/home/mahdi/HVR/hvr/data/iPad/set_{}/Depth_S_{}_D_{}_O_{}_N_{}.npy'.format(5, s, d, o, n))*1000
                        di = iPadImporter(depth, args, useCache=False)
                        Seq_0 = di.loadSequence_predict(docom=True, cube=(250,250,250), minDepth=minDepth, maxDepth=maxDepth) # we crop data centered in center of mass of data between maxdepth and mindepth and cube size cube in mm^3 and use it to get the resultant in UVD (do resize and ...) and feed it into the either train or test COM augmented refine net
                        testSeqs = [Seq_0]
                        testDataSet = iPadDataset(testSeqs)
                        test_data.append(testDataSet.imgStackDepthOnly(seqName=testSeqs, mode='iPad', vis=True))
                        com_data.append(testSeqs[0].data[0].com)

        #############################################################################
        test_data = np.asarray(test_data).squeeze(axis=1)
        print("create network")
        batchSize = 64

        nChannels = 1
        imgSizeW = 128
        imgSizeH = 128
        poseNetParams = ScaleNetParams(type=1, nChan=nChannels, wIn=imgSizeW, hIn=imgSizeH, batchSize=batchSize,
                                       resizeFactor=2, numJoints=1, nDims=3)
        poseNet = ScaleNet(rng, cfgParams=poseNetParams)

        poseNet.load(refineNet)

        ####################################################
        # TEST
        print("Testing ...")
        jts = poseNet.computeOutput([test_data, test_data[:, :, 32:96, 32:96], test_data[:, :, 48:80, 48:80]])
        joints = []
        for i in range(test_data.shape[0]):
            joints.append(jts[i].reshape(1, 3)*(testSeqs[0].config['cube'][2]/2.) + com_data[i])
        np.save('/home/mahdi/HVR/hvr/tmp/preprocesssed_depths/set_5_net_ICVL_COM_AUGMENT_refcoms.npy', np.asarray(joints).squeeze())

        return 1
#
# if __name__ == '__main__':
#     idepth = np.loadtxt('/home/mahdi/HVR/hvr/data/iPad/set_1/iPad_1_Depth_1.txt') * 1000  # mm
#     refined_com3D = refCOM(idepth, refineNet='/home/mahdi/HVR/git_repos/deep-prior-pp/src/eval/iPad_COM_AUGMENT/net_ICVL_COM_AUGMENT.pkl', device='iPad')
#     print "refined_com3D={}".format(refined_com3D)
