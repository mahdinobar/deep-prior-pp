"""Provides importer classes for importing data from different datasets.

DepthImporter provides interface for loading the data from a dataset, esp depth images.
ICVLImporter, NYUImporter, MSRAImporter are specific instances of different importers.

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

import scipy.io
import numpy as np
from PIL import Image
import os
import progressbar as pb
import struct
from data.basetypes import DepthFrame, NamedImgSequence
from util.handdetector import HandDetector
from data.transformations import transformPoints2D
import cPickle
import open3d as o3d
from PIL import Image

__author__ = "Mahdi Nobar <mahdi.nobar@epfl.ch>"
__copyright__ = "Copyright 2020, EPFL"
__credits__ = ["Mahdi Nobar"]
__license__ = "confidential"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "mahdi.nobar@epfl.ch"
__status__ = "Development"


class DepthImporter(object):
    """
    provide basic functionality to load depth data
    """

    def __init__(self, fx, fy, ux, uy, hand=None):
        """
        Initialize object
        :param fx: focal length in x direction
        :param fy: focal length in y direction
        :param ux: principal point in x direction
        :param uy: principal point in y direction
        """

        self.fx = fx
        self.fy = fy
        self.ux = ux
        self.uy = uy
        self.depth_map_size = (320, 240)
        self.refineNet = None
        self.crop_joint_idx = 0
        self.hand = hand

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        ret[0] = (sample[0]-self.ux)*sample[2]/self.fx
        ret[1] = (sample[1]-self.uy)*sample[2]/self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = sample[1]/sample[2]*self.fy+self.uy
        ret[2] = sample[2]
        return ret

    def getCameraProjection(self):
        """
        Get homogenous camera projection matrix
        :return: 4x4 camera projection matrix
        """
        ret = np.zeros((4, 4), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        ret[3, 2] = 1.
        return ret

    def getCameraIntrinsics(self):
        """
        Get intrinsic camera matrix
        :return: 3x3 intrinsic camera matrix
        """
        ret = np.zeros((3, 3), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        return ret

    def showAnnotatedDepth(self, frame):
        """
        Show the depth image
        :param frame: image to show
        :return:
        """
        raise NotImplementedError("Must be overloaded by base!")

    @staticmethod
    def depthToPCL(dpt, T, background_val=0.):

        # get valid points and transform
        pts = np.asarray(np.where(~np.isclose(dpt, background_val))).transpose()
        pts = np.concatenate([pts[:, [1, 0]] + 0.5, np.ones((pts.shape[0], 1), dtype='float32')], axis=1)
        pts = np.dot(np.linalg.inv(np.asarray(T)), pts.T).T
        pts = (pts[:, 0:2] / pts[:, 2][:, None]).reshape((pts.shape[0], 2))

        # replace the invalid data
        depth = dpt[(~np.isclose(dpt, background_val))]

        # get x and y data in a vectorized way
        row = (pts[:, 0] - 160.) / 241.42 * depth
        col = (pts[:, 1] - 120.) / 241.42 * depth

        # combine x,y,depth
        return np.column_stack((row, col, depth))

    def loadRefineNetLazy(self, net):
        if isinstance(net, basestring):
            if os.path.exists(net):
                from net.scalenet import ScaleNet, ScaleNetParams
                comrefNetParams = ScaleNetParams(type=5, nChan=1, wIn=128, hIn=128, batchSize=1, resizeFactor=2,
                                                 numJoints=1, nDims=3)
                self.refineNet = ScaleNet(np.random.RandomState(23455), cfgParams=comrefNetParams)
                self.refineNet.load(net)
            else:
                raise EnvironmentError("File not found: {}".format(net))


class ICVLImporter(DepthImporter):
    """
    provide functionality to load data from the ICVL dataset
    """

    def __init__(self, basepath, useCache=True, cacheDir='./cache/', refineNet=None, hand=None):
        """
        Constructor
        :param basepath: base path of the ICVL dataset
        :return:
        """

        super(ICVLImporter, self).__init__(241.42, 241.42, 160., 120., hand)  # see Qian et.al.

        self.depth_map_size = (320, 240)
        self.basepath = basepath
        self.useCache = useCache
        self.cacheDir = cacheDir
        self.numJoints = 16
        self.crop_joint_idx = 0
        self.refineNet = refineNet
        self.default_cubes = {'train': (250, 250, 250),
                              'test_seq_1': (250, 250, 250),
                              'test_seq_2': (250, 250, 250)}
        self.sides = {'train': 'right', 'test_seq1': 'right', 'test_seq_2': 'right'}

    def loadDepthMap(self, filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        img = Image.open(filename)  # open image
        assert len(img.getbands()) == 1  # ensure depth image
        imgdata = np.asarray(img, np.float32)

        return imgdata

    def getDepthMapNV(self):
        """
        Get the value of invalid depth values in the depth map
        :return: value
        """
        return 32001
        
    def loadSequence(self, seqName, subSeq=None, Nmax=float('inf'), shuffle=False, rng=None, docom=False, cube=None):
        """
        Load an image sequence from the dataset
        :param seqName: sequence name, e.g. train
        :param subSeq: list of subsequence names, e.g. 0, 45, 122-5
        :param Nmax: maximum number of samples to load
        :return: returns named image sequence
        """

        if (subSeq is not None) and (not isinstance(subSeq, list)):
            raise TypeError("subSeq must be None or list")

        if cube is None:
            config = {'cube': self.default_cubes[seqName]}
        else:
            assert isinstance(cube, tuple)
            assert len(cube) == 3
            config = {'cube': cube}

        if subSeq is None:
            pickleCache = '{}/{}_{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, self.hand,
                                                               HandDetector.detectionModeToString(docom, self.refineNet is not None), config['cube'][0])
        else:
            pickleCache = '{}/{}_{}_{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName,
                                                                  ''.join(subSeq), self.hand,
                                                                  HandDetector.detectionModeToString(docom, self.refineNet is not None), config['cube'][0])
        if self.useCache:
            if os.path.isfile(pickleCache):
                print("Loading cache data from {}".format(pickleCache))
                f = open(pickleCache, 'rb')
                (seqName, data, config) = cPickle.load(f)
                f.close()

                # shuffle data
                if shuffle and rng is not None:
                    print("Shuffling")
                    rng.shuffle(data)
                if not(np.isinf(Nmax)):
                    return NamedImgSequence(seqName, data[0:Nmax], config)
                else:
                    return NamedImgSequence(seqName, data, config)

            # check for multiple subsequences
            if subSeq is not None:
                if len(subSeq) > 1:
                    missing = False
                    for i in range(len(subSeq)):
                        if not os.path.isfile('{}/{}_{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__,
                                                                                   seqName, self.hand, subSeq[i],
                                                                                   HandDetector.detectionModeToString(docom, self.refineNet is not None))):
                            missing = True
                            print("missing: {}".format(subSeq[i]))
                            break

                    if not missing:
                        # load first data
                        pickleCache = '{}/{}_{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__,
                                                                           seqName, self.hand, subSeq[0],
                                                                           HandDetector.detectionModeToString(docom, self.refineNet is not None))
                        print("Loading cache data from {}".format(pickleCache))
                        f = open(pickleCache, 'rb')
                        (seqName, fullData, config) = cPickle.load(f)
                        f.close()
                        # load rest of data
                        for i in range(1, len(subSeq)):
                            pickleCache = '{}/{}_{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__,
                                                                               seqName, self.hand, subSeq[i],
                                                                               HandDetector.detectionModeToString(docom, self.refineNet is not None))
                            print("Loading cache data from {}".format(pickleCache))
                            f = open(pickleCache, 'rb')
                            (seqName, data, config) = cPickle.load(f)
                            fullData.extend(data)
                            f.close()

                        # shuffle data
                        if shuffle and rng is not None:
                            print("Shuffling")
                            rng.shuffle(fullData)
                        if not(np.isinf(Nmax)):
                            return NamedImgSequence(seqName, fullData[0:Nmax], config)
                        else:
                            return NamedImgSequence(seqName, fullData, config)

        self.loadRefineNetLazy(self.refineNet)

        # Load the dataset
        objdir = '{}/Depth/'.format(self.basepath)
        trainlabels = '{}/{}.txt'.format(self.basepath, seqName)

        inputfile = open(trainlabels)
        
        txt = 'Loading {}'.format(seqName)
        pbar = pb.ProgressBar(maxval=len(inputfile.readlines()), widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()
        inputfile.seek(0)
        
        data = []
        i = 0
        for line in inputfile:
            # early stop
            if len(data) >= Nmax:
                break

            part = line.split(' ')
            # check for subsequences and skip them if necessary
            subSeqName = ''
            if subSeq is not None:
                p = part[0].split('/')
                # handle original data (unrotated '0') separately
                if ('0' in subSeq) and len(p[0]) > 6:
                    pass
                elif not('0' in subSeq) and len(p[0]) > 6:
                    i += 1
                    continue
                elif (p[0] in subSeq) and len(p[0]) <= 6:
                    pass
                elif not(p[0] in subSeq) and len(p[0]) <= 6:
                    i += 1
                    continue

                if len(p[0]) <= 6:
                    subSeqName = p[0]
                else:
                    subSeqName = '0'

            dptFileName = '{}/{}'.format(objdir, part[0])

            if not os.path.isfile(dptFileName):
                print("File {} does not exist!".format(dptFileName))
                i += 1
                continue
            dpt = self.loadDepthMap(dptFileName)
            if self.hand is not None:
                if self.hand != self.sides[seqName]:
                    raise NotImplementedError()
                    dpt = dpt[:, ::-1]

            # joints in image coordinates
            gtorig = np.zeros((self.numJoints, 3), np.float32)
            for joint in range(self.numJoints):
                for xyz in range(0, 3):
                    gtorig[joint, xyz] = part[joint*3+xyz+1]

            # normalized joints in 3D coordinates
            gt3Dorig = self.jointsImgTo3D(gtorig)
            # print gt3D
            # self.showAnnotatedDepth(DepthFrame(dpt,gtorig,gtorig,0,gt3Dorig,gt3Dcrop,0,dptFileName,subSeqName,''))

            # Detect hand
            hd = HandDetector(dpt, self.fx, self.fy, refineNet=self.refineNet, importer=self)
            if not hd.checkImage(1):
                print("Skipping image {}, no content".format(dptFileName))
                i += 1
                continue
            try:
                dpt, M, com = hd.cropArea3D(com=gtorig[self.crop_joint_idx], size=config['cube'], docom=docom)
            except UserWarning:
                print("Skipping image {}, no hand detected".format(dptFileName))
                i += 1
                continue

            com3D = self.jointImgTo3D(com)
            gt3Dcrop = gt3Dorig - com3D  # normalize to com
            gtcrop = transformPoints2D(gtorig, M)

            # print("{}".format(gt3Dorig))
            # self.showAnnotatedDepth(DepthFrame(dpt,gtorig,gtcrop,M,gt3Dorig,gt3Dcrop,com3D,dptFileName,subSeqName,''))

            data.append(DepthFrame(dpt.astype(np.float32), gtorig, gtcrop, M, gt3Dorig, gt3Dcrop, com3D, dptFileName,
                                   subSeqName, 'left', {}))
            pbar.update(i)
            i += 1
                   
        inputfile.close()
        pbar.finish()
        print("Loaded {} samples.".format(len(data)))

        if self.useCache:
            print("Save cache data to {}".format(pickleCache))
            f = open(pickleCache, 'wb')
            cPickle.dump((seqName, data, config), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        # shuffle data
        if shuffle and rng is not None:
            print("Shuffling")
            rng.shuffle(data)
        return NamedImgSequence(seqName, data, config)

    def loadBaseline(self, filename, firstName=False):
        """
        Load baseline data
        :param filename: file name of data
        :return: list with joint coordinates
        """

        def nonblank_lines(f):
            for l in f:
                line = l.rstrip()
                if line:
                    yield line

        inputfile = open(filename)
        inputfile.seek(0)

        if firstName == True:
            off = 1
        else:
            off = 0

        data = []
        for line in nonblank_lines(inputfile):
            part = line.strip().split(' ')
            # joints in image coordinates
            ev = np.zeros((self.numJoints, 3), np.float32)
            for joint in range(ev.shape[0]):
                for xyz in range(0, 3):
                    ev[joint, xyz] = part[joint*3+xyz+off]

            gt3Dworld = self.jointsImgTo3D(ev)

            data.append(gt3Dworld)

        return data

    def loadBaseline2D(self, filename, firstName=False):
        """
        Load baseline data
        :param filename: file name of data
        :return: list with joint coordinates
        """

        inputfile = open(filename)
        inputfile.seek(0)

        if firstName is True:
            off = 1
        else:
            off = 0

        data = []
        for line in inputfile:
            part = line.split(' ')
            # joints in image coordinates
            ev = np.zeros((self.numJoints,2),np.float32)
            for joint in range(ev.shape[0]):
                for xyz in range(0, 2):
                    ev[joint,xyz] = part[joint*3+xyz+off]

            data.append(ev)

        return data

    def showAnnotatedDepth(self, frame):
        """
        Show the depth image
        :param frame: image to show
        :return:
        """
        import matplotlib
        import matplotlib.pyplot as plt

        print("img min {}, max {}".format(frame.dpt.min(), frame.dpt.max()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(frame.dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
        ax.scatter(frame.gtcrop[:, 0], frame.gtcrop[:, 1])

        ax.plot(frame.gtcrop[0:4, 0], frame.gtcrop[0:4, 1], c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[4:7, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[4:7, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[7:10, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[7:10, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[10:13, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[10:13, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[13:16, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[13:16, 1])), c='r')

        def format_coord(x, y):
            numrows, numcols = frame.dpt.shape
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                z = frame.dpt[row, col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)

        ax.format_coord = format_coord

        for i in range(frame.gtcrop.shape[0]):
            ax.annotate(str(i), (int(frame.gtcrop[i, 0]), int(frame.gtcrop[i, 1])))

        plt.show()

class iPhoneImporter(DepthImporter):
    """
    provide functionality to load data from iPhone Truedepth

    """

    def __init__(self, basepath, useCache=True, cacheDir='./cache/', refineNet=None, detectorNet=None, derotNet=None, hand=None):
        """
        Constructor
        :param basepath: base path of the iPhone Truedepth
        :return:
        """
        # iPhone calibration
        _h = 240.
        _w = 320.
        iw = 3088.0
        ih = 2316.0
        xscale = _h / ih
        yscale = _w / iw
        _fx = 2880.0796 * xscale
        _fy = 2880.0796 * yscale
        # _cx = 1546.5824 * xscale
        # _cy = 1153.2035 * yscale
        _cx = 1153.2035 * xscale
        _cy = 1546.5824 * yscale
        super(iPhoneImporter, self).__init__(_fx, _fy, _cy, _cx, hand)  # see Sun et.al.

        self.depth_map_size = (320, 240)
        self.basepath = basepath
        self.useCache = useCache
        self.cacheDir = cacheDir
        self.refineNet = refineNet
        self.derotNet = derotNet
        self.detectorNet = detectorNet
        self.numJoints = 21
        self.crop_joint_idx = 5
        self.default_cubes = {'P0': (200, 200, 200)}
        self.sides = {'P0': 'right'}

    def loadDepthMap(self, filename):
        """
        Read a depth-map in bin format of msra15
        :param filename: file name to load
        :return: image data of depth image in mm for msra15 and image shape(240by320)
        """
# temporary: must be changed ###########################################################################################
        color_raw = o3d.io.read_image('/home/mahdi/HVR/hvr/hand_pcl_iPhone/Tom_set_2/iPhone/hand30wall50_color.png')
        depth_raw = o3d.io.read_image(filename)
        color_raw = o3d.geometry.Image(np.asarray(color_raw))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_scale=0.529, depth_trunc=30.0, convert_rgb_to_intensity=False)
        # iPhone calibration
        h = np.asarray(color_raw).shape[0]  # 480
        w = np.asarray(color_raw).shape[1]  # 640
        iw = 3088.0
        ih = 2316.0
        xscale = h / ih
        yscale = w / iw
        _fx = 2880.0796 * xscale
        _fy = 2880.0796 * yscale
        # _cx = 1546.5824 * xscale
        # _cy = 1153.2035 * yscale
        _cx = 1153.2035 * xscale
        _cy = 1546.5824 * yscale
        setIntrinsic = o3d.camera.PinholeCameraIntrinsic()
        setIntrinsic.set_intrinsics(width=w, height=h, fx=_fx, fy=_fy, cx=_cx, cy=_cy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            setIntrinsic)
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        z_values = (-np.asarray(pcd.points)[:, 2] * 1000)  # in mm
        depth_map = np.reshape(z_values, (480, 640))
        imgdata = np.asarray(Image.fromarray(depth_map).resize((320, 240)))
# temporary: must be changed ###########################################################################################
        return np.copy(imgdata)

    def getDepthMapNV(self):
        """
        Get the value of invalid depth values in the depth map
        :return: value
        """
        return 32001

    def loadSequence(self, seqName, subSeq=None, Nmax=float('inf'), shuffle=False, rng=None, docom=False, cube=None):
        """
        Load an image sequence from the dataset
        :param seqName: sequence name, e.g. subject1
        :param Nmax: maximum number of samples to load
        :return: returns named image sequence
        """

        if (subSeq is not None) and (not isinstance(subSeq, list)):
            raise TypeError("subSeq must be None or list")

        if cube is None:
            config = {'cube': self.default_cubes[seqName]}
        else:
            assert isinstance(cube, tuple)
            assert len(cube) == 3
            config = {'cube': cube}

        if subSeq is None:
            pickleCache = '{}/{}_{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, self.hand,
                                                               HandDetector.detectionModeToString(docom, self.refineNet is not None), config['cube'][0])
        else:
            pickleCache = '{}/{}_{}_{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, self.hand,
                                                               ''.join(subSeq), HandDetector.detectionModeToString(docom, self.refineNet is not None), config['cube'][0])
        if self.useCache & os.path.isfile(pickleCache):
            print("Loading cache data from {}".format(pickleCache))
            f = open(pickleCache, 'rb')
            (seqName, data, config) = cPickle.load(f)
            f.close()
            # shuffle data
            if shuffle and rng is not None:
                print("Shuffling")
                rng.shuffle(data)
            if not(np.isinf(Nmax)):
                return NamedImgSequence(seqName, data[0:Nmax], config)
            else:
                return NamedImgSequence(seqName, data, config)

        self.loadRefineNetLazy(self.refineNet)

        # Load the dataset
        objdir = '{}/{}/'.format(self.basepath, seqName)
        subdirs = sorted([name for name in os.listdir(objdir) if os.path.isdir(os.path.join(objdir, name))])

        txt = 'Loading {}'.format(seqName)
        nImgs = sum([len(files) for r, d, files in os.walk(objdir)]) // 2
        pbar = pb.ProgressBar(maxval=nImgs, widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()

        data = []
        pi = 0
        for subdir in subdirs:
            # check for subsequences and skip them if necessary
            subSeqName = ''
            if subSeq is not None:
                if subdir not in subSeq:
                    continue

                subSeqName = subdir

            # iterate all subdirectories
            trainlabels = '{}/{}/joint.txt'.format(objdir, subdir)

            inputfile = open(trainlabels)
            # read number of samples
            nImgs = int(inputfile.readline())

            for i in range(nImgs):
                # early stop
                if len(data) >= Nmax:
                    break

                line = inputfile.readline()
                part = line.split(' ')

                # dptFileName = '{}/{}/{}_depth.bin'.format(objdir, subdir, str(i).zfill(6))
                dptFileName = '{}/{}/{}_depth.png'.format(objdir, subdir, 'hand30wall50')

                if not os.path.isfile(dptFileName):
                    print("File {} does not exist!".format(dptFileName))
                    continue
                dpt = self.loadDepthMap(dptFileName)

                # joints in image coordinates
                gt3Dorig = np.zeros((self.numJoints, 3), np.float32)
                for joint in range(gt3Dorig.shape[0]):
                    for xyz in range(0, 3):
                        gt3Dorig[joint, xyz] = part[joint*3+xyz]

                # invert axis
                # gt3Dorig[:, 0] *= (-1.)
                gt3Dorig[:, 1] *= (-1.)
                gt3Dorig[:, 2] *= (-1.)

                # joints in image coordinates (x,y) in pxls and z in mm with resolution 320by240
                gtorig = self.joints3DToImg_iPhone(gt3Dorig)
                # gtorig = o3d_gtorig[:, [0, 1, 2]]
                # gtorig[:,[0]] = 320-gtorig[:,[0]]
########################################################################################################################
                import matplotlib.pyplot as plt
                import matplotlib
                fig, ax = plt.subplots()
                ax.imshow(dpt, cmap=matplotlib.cm.jet)
                ax.plot(gtorig[:, 0], gtorig[:,1], marker='+', markersize=30)
                plt.savefig('/home/mahdi/HVR/git_repos/deep-prior-pp/src/cache/test.png')
                plt.close()
########################################################################################################################
                if self.hand is not None:
                    if self.hand != self.sides[seqName]:
                        gtorig[:, 0] -= dpt.shape[1] / 2.
                        gtorig[:, 0] *= (-1)
                        gtorig[:, 0] += dpt.shape[1] / 2.
                        gt3Dorig = self.jointsImgTo3D(gtorig)
                        dpt = dpt[:, ::-1]

                # print gt3D
                # self.showAnnotatedDepth(DepthFrame(dpt,gtorig,gtorig,0,gt3Dorig,gt3Dcrop,com3D,dptFileName,'',''))
                # Detect hand
                hd = HandDetector(dpt, self.fx, self.fy, refineNet=self.refineNet, importer=self)
                if not hd.checkImage(1.):
                    print("Skipping image {}, no content".format(dptFileName))
                    continue

                try: #here we initialize the com with ground truth mcp middle finger of msra15 dataset [z in mm, (x,y) in pxls]
                    dpt, M, com = hd.cropArea3D(com=gtorig[self.crop_joint_idx], size=config['cube'], docom=docom) #dpt resolution changes to 128by128
                except UserWarning:
                    print("Skipping image {}, no hand detected".format(dptFileName))
                    continue

                com3D = self.jointImgTo3D_iPhone(com)
                gt3Dcrop = gt3Dorig - com3D  # normalize to com

                # gtcrop=ground truth joints.txt in CROPPED image coordinate(pxl, pxl, mm)
                gtcrop = transformPoints2D(gtorig, M)

                # print("{}".format(gt3Dorig))
                # self.showAnnotatedDepth(DepthFrame(dpt,gtorig,gtcrop,M,gt3Dorig,gt3Dcrop,com3D,dptFileName,'','',{}))

                data.append(DepthFrame(dpt.astype(np.float32), gtorig, gtcrop, M, gt3Dorig, gt3Dcrop, com3D,
                                       dptFileName, subSeqName, self.sides[seqName], {}))
                pbar.update(pi)
                # pi += 1

            inputfile.close()

        pbar.finish()
        print("Loaded {} samples.".format(len(data)))

        if self.useCache:
            print("Save cache data to {}".format(pickleCache))
            f = open(pickleCache, 'wb')
            cPickle.dump((seqName, data, config), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        # shuffle data
        if shuffle and rng is not None:
            print("Shuffling")
            rng.shuffle(data)
        return NamedImgSequence(seqName, data, config)


    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in xrange(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        ret[0] = (sample[0] - self.ux) * sample[2] / self.fx
        ret[1] = (self.uy - sample[1]) * sample[2] / self.fy
        ret[2] = sample[2]
        return ret


    def jointsImgTo3D_iPhone(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in xrange(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D_iPhone(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        ret[0] = (sample[0] - self.ux) * sample[2] / self.fx
        ret[1] = (sample[1] - self.uy) * sample[2] / self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg_iPhone(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in xrange(sample.shape[0]):
            ret[i] = self.joint3DToImg_iPhone(sample[i])
        return ret

    def joint3DToImg_iPhone(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3, ), np.float32)
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.uy
        ret[1] = sample[1]/sample[2]*self.fy + self.ux
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in xrange(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3, ), np.float32)
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = self.uy-sample[1]/sample[2]*self.fy
        ret[2] = sample[2]
        return ret

    def getCameraIntrinsics(self):
        """
        Get intrinsic camera matrix
        :return: 3x3 intrinsic camera matrix
        """
        ret = np.zeros((3, 3), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = -self.fy
        ret[2, 2] = 1
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        return ret

    def getCameraProjection(self):
        """
        Get homogenous camera projection matrix
        :return: 4x4 camera projection matrix
        """
        ret = np.zeros((4, 4), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = -self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        ret[3, 2] = 1.
        return ret

    def showAnnotatedDepth(self, frame):
        """
        Show the depth image
        :param frame: image to show
        :return:
        """
        import matplotlib
        import matplotlib.pyplot as plt

        print("img min {}, max {}".format(frame.dpt.min(),frame.dpt.max()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(frame.dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
        ax.scatter(frame.gtcrop[:, 0], frame.gtcrop[:, 1])

        ax.plot(frame.gtcrop[0:5, 0], frame.gtcrop[0:5, 1], c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[5:9, 0])), np.hstack((frame.gtcrop[0, 1], frame.gtcrop[5:9, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[9:13, 0])), np.hstack((frame.gtcrop[0, 1], frame.gtcrop[9:13, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[13:17, 0])), np.hstack((frame.gtcrop[0, 1], frame.gtcrop[13:17, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[17:21, 0])), np.hstack((frame.gtcrop[0, 1], frame.gtcrop[17:21, 1])), c='r')

        def format_coord(x, y):
            numrows, numcols = frame.dpt.shape
            col = int(x+0.5)
            row = int(y+0.5)
            if 0 <= col < numcols and 0 <= row < numrows:
                z = frame.dpt[row, col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f'%(x, y)
        ax.format_coord = format_coord

        for i in range(frame.gtcrop.shape[0]):
            ax.annotate(str(i), (int(frame.gtcrop[i, 0]), int(frame.gtcrop[i, 1])))

        plt.show()

    @staticmethod
    def depthToPCL(dpt, T, background_val=0.):

        # get valid points and transform
        pts = np.asarray(np.where(~np.isclose(dpt, background_val))).transpose()
        pts = np.concatenate([pts[:, [1, 0]] + 0.5, np.ones((pts.shape[0], 1), dtype='float32')], axis=1)
        pts = np.dot(np.linalg.inv(np.asarray(T)), pts.T).T
        pts = (pts[:, 0:2] / pts[:, 2][:, None]).reshape((pts.shape[0], 2))

        # replace the invalid data
        depth = dpt[(~np.isclose(dpt, background_val))]

        # get x and y data in a vectorized way
        row = (pts[:, 0] - 160.) / 241.42 * depth
        col = (120. - pts[:, 1]) / 241.42 * depth

        # combine x,y,depth
        return np.column_stack((row, col, depth))


class MSRA15Importer(DepthImporter):
    """
    provide functionality to load data from the MSRA 2015 dataset

    faulty images:
    - P2/TIP: 172, 173,174
    - P2/MP: 173, 174, 175, 345-354, 356, 359, 360
    - P3/T: 120, 489
    - P8/4: 168
    """

    def __init__(self, basepath, useCache=True, cacheDir='./cache/', refineNet=None, detectorNet=None, derotNet=None, hand=None):
        """
        Constructor
        :param basepath: base path of the MSRA dataset
        :return:
        """

        super(MSRA15Importer, self).__init__(241.42, 241.42, 160., 120., hand)  # see Sun et.al.

        self.depth_map_size = (320, 240)
        self.basepath = basepath
        self.useCache = useCache
        self.cacheDir = cacheDir
        self.refineNet = refineNet
        self.derotNet = derotNet
        self.detectorNet = detectorNet
        self.numJoints = 21
        self.crop_joint_idx = 5
        self.default_cubes = {'P0': (200, 200, 200)}
        self.sides = {'P0': 'right'}

    def loadDepthMap(self, filename):
        """
        Read a depth-map in bin format of msra15
        :param filename: file name to load
        :return: image data of depth image in mm for msra15 and image shape(240by320)
        """
        with open(filename, 'rb') as f:
            # first 6 uint define the full image
            width = struct.unpack('i', f.read(4))[0]
            height = struct.unpack('i', f.read(4))[0]
            left = struct.unpack('i', f.read(4))[0]
            top = struct.unpack('i', f.read(4))[0]
            right = struct.unpack('i', f.read(4))[0]
            bottom = struct.unpack('i', f.read(4))[0]
            patch = np.fromfile(f, dtype='float32', sep="")
            imgdata = np.zeros((height, width), dtype='float32')
            imgdata[top:bottom, left:right] = patch.reshape([bottom-top, right-left])

        return imgdata

    def getDepthMapNV(self):
        """
        Get the value of invalid depth values in the depth map
        :return: value
        """
        return 32001

    def loadSequence(self, seqName, subSeq=None, Nmax=float('inf'), shuffle=False, rng=None, docom=False, cube=None):
        """
        Load an image sequence from the dataset
        :param seqName: sequence name, e.g. subject1
        :param Nmax: maximum number of samples to load
        :return: returns named image sequence
        """

        if (subSeq is not None) and (not isinstance(subSeq, list)):
            raise TypeError("subSeq must be None or list")

        if cube is None:
            config = {'cube': self.default_cubes[seqName]}
        else:
            assert isinstance(cube, tuple)
            assert len(cube) == 3
            config = {'cube': cube}

        if subSeq is None:
            pickleCache = '{}/{}_{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, self.hand,
                                                               HandDetector.detectionModeToString(docom, self.refineNet is not None), config['cube'][0])
        else:
            pickleCache = '{}/{}_{}_{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, self.hand,
                                                               ''.join(subSeq), HandDetector.detectionModeToString(docom, self.refineNet is not None), config['cube'][0])
        if self.useCache & os.path.isfile(pickleCache):
            print("Loading cache data from {}".format(pickleCache))
            f = open(pickleCache, 'rb')
            (seqName, data, config) = cPickle.load(f)
            f.close()
            # shuffle data
            if shuffle and rng is not None:
                print("Shuffling")
                rng.shuffle(data)
            if not(np.isinf(Nmax)):
                return NamedImgSequence(seqName, data[0:Nmax], config)
            else:
                return NamedImgSequence(seqName, data, config)

        self.loadRefineNetLazy(self.refineNet)

        # Load the dataset
        objdir = '{}/{}/'.format(self.basepath, seqName)
        subdirs = sorted([name for name in os.listdir(objdir) if os.path.isdir(os.path.join(objdir, name))])

        txt = 'Loading {}'.format(seqName)
        nImgs = sum([len(files) for r, d, files in os.walk(objdir)]) // 2
        pbar = pb.ProgressBar(maxval=nImgs, widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()

        data = []
        pi = 0
        for subdir in subdirs:
            # check for subsequences and skip them if necessary
            subSeqName = ''
            if subSeq is not None:
                if subdir not in subSeq:
                    continue

                subSeqName = subdir

            # iterate all subdirectories
            trainlabels = '{}/{}/joint.txt'.format(objdir, subdir)

            inputfile = open(trainlabels)
            # read number of samples
            nImgs = int(inputfile.readline())

            for i in range(nImgs):
                # early stop
                if len(data) >= Nmax:
                    break

                line = inputfile.readline()
                part = line.split(' ')

                dptFileName = '{}/{}/{}_depth.bin'.format(objdir, subdir, str(i).zfill(6))

                if not os.path.isfile(dptFileName):
                    print("File {} does not exist!".format(dptFileName))
                    continue
                dpt = self.loadDepthMap(dptFileName)

                # joints in image coordinates
                gt3Dorig = np.zeros((self.numJoints, 3), np.float32)
                for joint in range(gt3Dorig.shape[0]):
                    for xyz in range(0, 3):
                        gt3Dorig[joint, xyz] = part[joint*3+xyz]

                # invert axis
                # gt3Dorig[:, 0] *= (-1.)
                # gt3Dorig[:, 1] *= (-1.)
                gt3Dorig[:, 2] *= (-1.)

                # normalized joints in 3D coordinates
                gtorig = self.joints3DToImg(gt3Dorig)

                if self.hand is not None:
                    if self.hand != self.sides[seqName]:
                        gtorig[:, 0] -= dpt.shape[1] / 2.
                        gtorig[:, 0] *= (-1)
                        gtorig[:, 0] += dpt.shape[1] / 2.
                        gt3Dorig = self.jointsImgTo3D(gtorig)
                        dpt = dpt[:, ::-1]

                # print gt3D
                # self.showAnnotatedDepth(DepthFrame(dpt,gtorig,gtorig,0,gt3Dorig,gt3Dcrop,com3D,dptFileName,'',''))
                # Detect hand
                hd = HandDetector(dpt, self.fx, self.fy, refineNet=self.refineNet, importer=self)
                if not hd.checkImage(1.):
                    print("Skipping image {}, no content".format(dptFileName))
                    continue

                try: #here we initialize the com with ground truth mcp middle finger of msra15 dataset [z in mm, (x,y) in pxls]
                    dpt, M, com = hd.cropArea3D(com=gtorig[self.crop_joint_idx], size=config['cube'], docom=docom)
                except UserWarning:
                    print("Skipping image {}, no hand detected".format(dptFileName))
                    continue

                com3D = self.jointImgTo3D(com)
                gt3Dcrop = gt3Dorig - com3D  # normalize to com

                gtcrop = transformPoints2D(gtorig, M)

                # print("{}".format(gt3Dorig))
                # self.showAnnotatedDepth(DepthFrame(dpt,gtorig,gtcrop,M,gt3Dorig,gt3Dcrop,com3D,dptFileName,'','',{}))

                data.append(DepthFrame(dpt.astype(np.float32), gtorig, gtcrop, M, gt3Dorig, gt3Dcrop, com3D,
                                       dptFileName, subSeqName, self.sides[seqName], {}))
                pbar.update(pi)
                # pi += 1

            inputfile.close()

        pbar.finish()
        print("Loaded {} samples.".format(len(data)))

        if self.useCache:
            print("Save cache data to {}".format(pickleCache))
            f = open(pickleCache, 'wb')
            cPickle.dump((seqName, data, config), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        # shuffle data
        if shuffle and rng is not None:
            print("Shuffling")
            rng.shuffle(data)
        return NamedImgSequence(seqName, data, config)

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in xrange(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        ret[0] = (sample[0] - self.ux) * sample[2] / self.fx
        ret[1] = (self.uy - sample[1]) * sample[2] / self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in xrange(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3, ), np.float32)
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = self.uy-sample[1]/sample[2]*self.fy
        ret[2] = sample[2]
        return ret

    def getCameraIntrinsics(self):
        """
        Get intrinsic camera matrix
        :return: 3x3 intrinsic camera matrix
        """
        ret = np.zeros((3, 3), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = -self.fy
        ret[2, 2] = 1
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        return ret

    def getCameraProjection(self):
        """
        Get homogenous camera projection matrix
        :return: 4x4 camera projection matrix
        """
        ret = np.zeros((4, 4), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = -self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        ret[3, 2] = 1.
        return ret

    def showAnnotatedDepth(self, frame):
        """
        Show the depth image
        :param frame: image to show
        :return:
        """
        import matplotlib
        import matplotlib.pyplot as plt

        print("img min {}, max {}".format(frame.dpt.min(),frame.dpt.max()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(frame.dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
        ax.scatter(frame.gtcrop[:, 0], frame.gtcrop[:, 1])

        ax.plot(frame.gtcrop[0:5, 0], frame.gtcrop[0:5, 1], c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[5:9, 0])), np.hstack((frame.gtcrop[0, 1], frame.gtcrop[5:9, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[9:13, 0])), np.hstack((frame.gtcrop[0, 1], frame.gtcrop[9:13, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[13:17, 0])), np.hstack((frame.gtcrop[0, 1], frame.gtcrop[13:17, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[17:21, 0])), np.hstack((frame.gtcrop[0, 1], frame.gtcrop[17:21, 1])), c='r')

        def format_coord(x, y):
            numrows, numcols = frame.dpt.shape
            col = int(x+0.5)
            row = int(y+0.5)
            if 0 <= col < numcols and 0 <= row < numrows:
                z = frame.dpt[row, col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f'%(x, y)
        ax.format_coord = format_coord

        for i in range(frame.gtcrop.shape[0]):
            ax.annotate(str(i), (int(frame.gtcrop[i, 0]), int(frame.gtcrop[i, 1])))

        plt.show()

    @staticmethod
    def depthToPCL(dpt, T, background_val=0.):

        # get valid points and transform
        pts = np.asarray(np.where(~np.isclose(dpt, background_val))).transpose()
        pts = np.concatenate([pts[:, [1, 0]] + 0.5, np.ones((pts.shape[0], 1), dtype='float32')], axis=1)
        pts = np.dot(np.linalg.inv(np.asarray(T)), pts.T).T
        pts = (pts[:, 0:2] / pts[:, 2][:, None]).reshape((pts.shape[0], 2))

        # replace the invalid data
        depth = dpt[(~np.isclose(dpt, background_val))]

        # get x and y data in a vectorized way
        row = (pts[:, 0] - 160.) / 241.42 * depth
        col = (120. - pts[:, 1]) / 241.42 * depth

        # combine x,y,depth
        return np.column_stack((row, col, depth))

class NYUImporter(DepthImporter):
    """
    provide functionality to load data from the NYU hand dataset
    """

    def __init__(self, basepath, useCache=True, cacheDir='./cache/', refineNet=None, 
                 allJoints=False, hand=None):
        """
        Constructor
        :param basepath: base path of the ICVL dataset
        :return:
        """

        super(NYUImporter, self).__init__(588.03, 587.07, 320., 240., hand)

        self.depth_map_size = (640, 480)
        self.basepath = basepath
        self.useCache = useCache
        self.cacheDir = cacheDir
        self.allJoints = allJoints
        self.numJoints = 36
        if self.allJoints:
            self.crop_joint_idx = 32
        else:
            self.crop_joint_idx = 13
        self.default_cubes = {'train': (300, 300, 300),
                              'test_1': (300, 300, 300),
                              'test_2': (250, 250, 250),
                              'test': (300, 300, 300),
                              'train_synth': (300, 300, 300),
                              'test_synth_1': (300, 300, 300),
                              'test_synth_2': (250, 250, 250),
                              'test_synth': (300, 300, 300)}
        self.sides = {'train': 'right', 'test_1': 'right', 'test_2': 'right', 'test': 'right', 'train_synth': 'right',
                      'test_synth_1': 'right', 'test_synth_2': 'right', 'test_synth': 'right'}
        # joint indices used for evaluation of Tompson et al.
        self.restrictedJointsEval = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
        self.refineNet = refineNet

    def loadDepthMap(self, filename):
        """
        Read a depth-map
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

    def getDepthMapNV(self):
        """
        Get the value of invalid depth values in the depth map
        :return: value
        """
        return 32001

    def loadSequence(self, seqName, Nmax=float('inf'), shuffle=False, rng=None, docom=False, cube=None):
        """
        Load an image sequence from the dataset
        :param seqName: sequence name, e.g. train
        :param Nmax: maximum number of samples to load
        :return: returns named image sequence
        """
        if cube is None:
            config = {'cube': self.default_cubes[seqName]}
        else:
            assert isinstance(cube, tuple)
            assert len(cube) == 3
            config = {'cube': cube}

        pickleCache = '{}/{}_{}_{}_{}_{}_{}__cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, 
                                                                 self.hand, self.allJoints,
                                                                 HandDetector.detectionModeToString(docom, self.refineNet is not None), config['cube'][0])
        if self.useCache:
            if os.path.isfile(pickleCache):
                print("Loading cache data from {}".format(pickleCache))
                f = open(pickleCache, 'rb')
                (seqName, data, config) = cPickle.load(f)
                f.close()

                # shuffle data
                if shuffle and rng is not None:
                    print("Shuffling")
                    rng.shuffle(data)
                if not(np.isinf(Nmax)):
                    return NamedImgSequence(seqName, data[0:Nmax], config)
                else:
                    return NamedImgSequence(seqName, data, config)

        self.loadRefineNetLazy(self.refineNet)

        # Load the dataset
        objdir = '{}/{}/'.format(self.basepath, seqName)
        trainlabels = '{}/{}/joint_data.mat'.format(self.basepath, seqName)

        mat = scipy.io.loadmat(trainlabels)
        joints3D = mat['joint_xyz'][0]
        joints2D = mat['joint_uvd'][0]
        if self.allJoints:
            eval_idxs = np.arange(36)
        else:
            eval_idxs = self.restrictedJointsEval

        self.numJoints = len(eval_idxs)

        txt = 'Loading {}'.format(seqName)
        pbar = pb.ProgressBar(maxval=joints3D.shape[0], widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()

        data = []
        i = 0
        for line in range(joints3D.shape[0]):
            dptFileName = '{0:s}/depth_1_{1:07d}.png'.format(objdir, line+1)

            if not os.path.isfile(dptFileName):
                print("File {} does not exist!".format(dptFileName))
                i += 1
                continue
            dpt = self.loadDepthMap(dptFileName)
            if self.hand is not None:
                if self.hand != self.sides[seqName]:
                    raise NotImplementedError()
                    dpt = dpt[:, ::-1]

            # joints in image coordinates
            gtorig = np.zeros((self.numJoints, 3), np.float32)
            jt = 0
            for ii in range(joints2D.shape[1]):
                if ii not in eval_idxs:
                    continue
                gtorig[jt, 0] = joints2D[line, ii, 0]
                gtorig[jt, 1] = joints2D[line, ii, 1]
                gtorig[jt, 2] = joints2D[line, ii, 2]
                jt += 1

            # normalized joints in 3D coordinates
            gt3Dorig = np.zeros((self.numJoints, 3), np.float32)
            jt = 0
            for jj in range(joints3D.shape[1]):
                if jj not in eval_idxs:
                    continue
                gt3Dorig[jt, 0] = joints3D[line, jj, 0]
                gt3Dorig[jt, 1] = joints3D[line, jj, 1]
                gt3Dorig[jt, 2] = joints3D[line, jj, 2]
                jt += 1
            # print gt3D
            # self.showAnnotatedDepth(DepthFrame(dpt,gtorig,gtorig,0,gt3Dorig,gt3Dorig,0,dptFileName,'',''))

            # Detect hand
            hd = HandDetector(dpt, self.fx, self.fy, refineNet=self.refineNet, importer=self)
            if not hd.checkImage(1):
                print("Skipping image {}, no content".format(dptFileName))
                i += 1
                continue
            try:
                dpt, M, com = hd.cropArea3D(com=gtorig[self.crop_joint_idx], size=config['cube'], docom=docom)
            except UserWarning:
                print("Skipping image {}, no hand detected".format(dptFileName))
                i += 1
                continue

            com3D = self.jointImgTo3D(com)
            gt3Dcrop = gt3Dorig - com3D  # normalize to com
            gtcrop = transformPoints2D(gtorig, M)

            # print("{}".format(gt3Dorig))
            # self.showAnnotatedDepth(DepthFrame(dpt,gtorig,gtcrop,M,gt3Dorig,gt3Dcrop,com3D,dptFileName,'','',{}))

            data.append(DepthFrame(dpt.astype(np.float32), gtorig, gtcrop, M, gt3Dorig, gt3Dcrop, com3D, dptFileName,
                                   '', self.sides[seqName], {}))
            pbar.update(i)
            i += 1

            # early stop
            if len(data) >= Nmax:
                break

        pbar.finish()
        print("Loaded {} samples.".format(len(data)))

        if self.useCache:
            print("Save cache data to {}".format(pickleCache))
            f = open(pickleCache, 'wb')
            cPickle.dump((seqName, data, config), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        # shuffle data
        if shuffle and rng is not None:
            print("Shuffling")
            rng.shuffle(data)
        return NamedImgSequence(seqName, data, config)

    def loadBaseline(self, filename, gt=None):
        """
        Load baseline data
        :param filename: file name of data
        :return: list with joint coordinates
        """

        if gt is not None:
            mat = scipy.io.loadmat(filename)
            names = mat['conv_joint_names'][0]
            joints = mat['pred_joint_uvconf'][0]

            self.numJoints = names.shape[0]

            data = []
            for dat in range(min(joints.shape[0], gt.shape[0])):
                fname = '{0:s}/depth_1_{1:07d}.png'.format(os.path.split(filename)[0], dat+1)
                if not os.path.isfile(fname):
                    continue
                dm = self.loadDepthMap(fname)
                # joints in image coordinates
                ev = np.zeros((self.numJoints, 3), np.float32)
                jt = 0
                for i in range(joints.shape[1]):
                    if np.count_nonzero(joints[dat, i, :]) == 0:
                        continue
                    ev[jt, 0] = joints[dat, i, 0]
                    ev[jt, 1] = joints[dat, i, 1]
                    ev[jt, 2] = dm[int(ev[jt, 1]), int(ev[jt, 0])]
                    jt += 1

                for jt in range(ev.shape[0]):
                    #if ev[jt,2] == 2001. or ev[jt,2] == 0.:
                    if abs(ev[jt, 2] - gt[dat, 13, 2]) > 150.:
                        ev[jt, 2] = gt[dat, jt, 2]#np.clip(ev[jt,2],gt[dat,13,2]-150.,gt[dat,13,2]+150.) # set to groundtruth if unknown

                ev3D = self.jointsImgTo3D(ev)
                data.append(ev3D)

            return data
        else:

            def nonblank_lines(f):
                for l in f:
                    line = l.rstrip()
                    if line:
                        yield line

            inputfile = open(filename)
            # first line specifies the number of 3D joints
            self.numJoints = len(inputfile.readline().split(' ')) / 3
            inputfile.seek(0)

            data = []
            for line in nonblank_lines(inputfile):
                part = line.split(' ')
                # joints in image coordinates
                ev = np.zeros((self.numJoints, 3), np.float32)
                for joint in range(ev.shape[0]):
                    for xyz in range(0, 3):
                        ev[joint, xyz] = part[joint*3+xyz]

                gt3Dworld = self.jointsImgTo3D(ev)

                data.append(gt3Dworld)

            return data

    def loadBaseline2D(self, filename):
        """
        Load baseline data
        :param filename: file name of data
        :return: list with joint coordinates
        """

        mat = scipy.io.loadmat(filename)
        names = mat['conv_joint_names'][0]
        joints = mat['pred_joint_uvconf'][0]

        self.numJoints = names.shape[0]

        data = []
        for dat in range(joints.shape[0]):
            # joints in image coordinates
            ev = np.zeros((self.numJoints, 2), np.float32)
            jt = 0
            for i in range(joints.shape[1]):
                if np.count_nonzero(joints[dat, i, :]) == 0:
                    continue
                ev[jt, 0] = joints[dat, i, 0]
                ev[jt, 1] = joints[dat, i, 1]
                jt += 1

            data.append(ev)

        return data

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in xrange(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        ret[0] = (sample[0] - self.ux) * sample[2] / self.fx
        ret[1] = (self.uy - sample[1]) * sample[2] / self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in xrange(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3, ), np.float32)
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = self.uy-sample[1]/sample[2]*self.fy
        ret[2] = sample[2]
        return ret

    def getCameraIntrinsics(self):
        """
        Get intrinsic camera matrix
        :return: 3x3 intrinsic camera matrix
        """
        ret = np.zeros((3, 3), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = -self.fy
        ret[2, 2] = 1
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        return ret

    def getCameraProjection(self):
        """
        Get homogenous camera projection matrix
        :return: 4x4 camera projection matrix
        """
        ret = np.zeros((4, 4), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = -self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        ret[3, 2] = 1.
        return ret

    def showAnnotatedDepth(self, frame):
        """
        Show the depth image
        :param frame: image to show
        :return:
        """
        import matplotlib
        import matplotlib.pyplot as plt

        print("img min {}, max {}".format(frame.dpt.min(), frame.dpt.max()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(frame.dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
        ax.scatter(frame.gtcrop[:, 0], frame.gtcrop[:, 1])

        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[1::-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[1::-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[3:1:-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[3:1:-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[5:3:-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[5:3:-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[7:5:-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[7:5:-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[10:7:-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[10:7:-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[11, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[11, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[12, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[12, 1])), c='r')

        def format_coord(x, y):
            numrows, numcols = frame.dpt.shape
            col = int(x+0.5)
            row = int(y+0.5)
            if col>=0 and col<numcols and row>=0 and row<numrows:
                z = frame.dpt[row,col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)
        ax.format_coord = format_coord

        for i in range(frame.gtcrop.shape[0]):
            ax.annotate(str(i), (int(frame.gtcrop[i, 0]), int(frame.gtcrop[i, 1])))

        plt.show()

    @staticmethod
    def depthToPCL(dpt, T, background_val=0.):

        # get valid points and transform
        pts = np.asarray(np.where(~np.isclose(dpt, background_val))).transpose()
        pts = np.concatenate([pts[:, [1, 0]] + 0.5, np.ones((pts.shape[0], 1), dtype='float32')], axis=1)
        pts = np.dot(np.linalg.inv(np.asarray(T)), pts.T).T
        pts = (pts[:, 0:2] / pts[:, 2][:, None]).reshape((pts.shape[0], 2))

        # replace the invalid data
        depth = dpt[(~np.isclose(dpt, background_val))]

        # get x and y data in a vectorized way
        row = (pts[:, 0] - 320.) / 588.03 * depth
        col = (240. - pts[:, 1]) / 587.07 * depth

        # combine x,y,depth
        return np.column_stack((row, col, depth))

