from Core.Network import Network
from Core.LayerUtils import LayerFactory
from Core.CommonUtils import nms, convert_to_square, pad
import numpy as np
import cv2


class Rnet(Network):
    def _config(self):
        layer_factory = LayerFactory(self)

        layer_factory.new_feed(name='input', layer_shape=(None, 24, 24, 3))
        layer_factory.new_conv(name='conv1', num_input_channels=3, filter_size=3,
                               num_output_channels=28, padding='VALID')
        layer_factory.new_max_pooling('pool1', kernel_size=3, strides=2, padding='SAME')
        layer_factory.new_conv(name='conv2', num_input_channels=28, filter_size=3,
                               num_output_channels=48, padding='VALID')
        layer_factory.new_max_pooling('pool2', kernel_size=3, strides=2, padding='VALID')
        layer_factory.new_conv(name='conv3', num_input_channels=48, filter_size=2,
                               num_output_channels=64, padding='VALID')
        layer_factory.new_full_connected(name='fc1', output_dim=128,relu=True)
        layer_factory.new_full_connected(name='fc2-1', output_dim=2)
        layer_factory.new_softmax(name='softmax', axis=1)
        layer_factory.new_full_connected(name='fc2-2', output_dim=4, input_layername='fc1')

    def _feed(self, image):
        return self._session.run(['rnet/fc2-2/fc2-2:0', 'rnet/softmax:0'], feed_dict={'rnet/input:0': image})

    def detectbox(self, image, dets, vis=False):

        h, w, c = image.shape

        if dets is None:
            return None, None

        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]

        feed_imgs = []
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = image[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            img = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
            feed_imgs.append(img)

        feed_imgs = np.array(feed_imgs)

        reg, cls_map = self._feed(feed_imgs)

        keep_inds = np.where(cls_map[:, 1] > 0.7)[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None
        keep = nms(boxes.copy(), 0.7, mode="Union")
        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        boxes = np.vstack([keep_boxes[:, 0],
                           keep_boxes[:, 1],
                           keep_boxes[:, 2],
                           keep_boxes[:, 3],
                           keep_cls[:, 0]
                          ])

        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        if vis:
            for i in range(align_topx.shape[0]):
                font = cv2.FONT_HERSHEY_TRIPLEX
                cv2.putText(image, 'face', (int(align_topx[i]), int(align_topy[i]-10)), font, 0.8, (255, 255, 0), 1, False)
                cv2.rectangle(image, (int(align_topx[i]), int(align_topy[i])),
                              (int(align_bottomx[i]), int(align_bottomy[i])), (200, 148, 0, 0.1), 2)
            cv2.imshow("1", image)
            cv2.waitKey(0)

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0]
                                 ])
        boxes = boxes.T
        boxes_align = boxes_align.T

        return boxes, boxes_align
