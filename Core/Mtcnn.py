from Core.Pnet import Pnet
from Core.Rnet import Rnet
from Core.Onet import Onet
import tensorflow as tf
import numpy as np
import cv2
import time


class MtcnnDetector(object):
    def __init__(self, pnet, rnet, onet):
        self.pnet_weight = pnet
        self.rnet_weight = rnet
        self.onet_weight = onet
        self._session = tf.Session()

        pweight = np.load(self.pnet_weight).item()
        rweight = np.load(self.rnet_weight).item()
        oweight = np.load(self.onet_weight).item()

        self.__pnet = Pnet(session=self._session)
        self.__pnet.set_weights(pweight)

        self.__rnet = Rnet(session=self._session)
        self.__rnet.set_weights(rweight)

        self.__onet = Onet(session=self._session)
        self.__onet.set_weights(oweight)

    def detectImage(self, imagepath=None, vis=True):
        img = cv2.imread(imagepath)
        t = time.time()
        box, boxalign = self.__pnet.detectbox(img, 0.75, vis=False)
        t1 = time.time() - t
        t = time.time()
        box2, boxalign2 = self.__rnet.detectbox(img, boxalign, vis=False)
        t2 = time.time() - t
        t = time.time()
        boxalign3 = self.__onet.detectbox(img, boxalign2, vis=False)
        t3 = time.time() - t
        print(boxalign3.shape)
        #
        if vis:
            for i in range(boxalign3.shape[0]):
                cv2.rectangle(img, (int(boxalign3[i][0]), int(boxalign3[i][1])),
                              (int(boxalign3[i][2]), int(boxalign3[i][3])), (200, 148, 0, 0.1), 2)
            cv2.imshow("0", img)
            cv2.waitKey(0)
            cv2.imwrite('./Result/test.jpg', img)


        print("time cost " + '{:.3f}'.format(t1+t2+t3) + '\t pnet {:.3f} rnet {:.3f} onet {:.3f}'.format(t1, t2, t3))

        if boxalign3 is None:
            return np.array([])
        return boxalign3