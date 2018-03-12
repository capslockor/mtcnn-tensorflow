from Core.Network import Network
from Core.LayerUtils import LayerFactory
from Core.CommonUtils import nms, resize_image, generate_bounding_box
import numpy as np
import cv2


class Pnet(Network):
    def _config(self):
        layer_factory = LayerFactory(self)

        layer_factory.new_feed(name='input', layer_shape=(None, None, None, 3))
        layer_factory.new_conv(name='conv1', num_input_channels=3, filter_size=3,
                               num_output_channels=10,
                               padding='VALID')
        layer_factory.new_max_pooling('pool1', kernel_size=2, strides=2, padding="SAME")

        layer_factory.new_conv(name='conv2', num_input_channels=10, filter_size=3,
                               num_output_channels=16,
                               padding='VALID')
        layer_factory.new_conv(name='conv3', num_input_channels=16, filter_size=3,
                               num_output_channels=32,
                               padding='VALID')
        layer_factory.new_conv(name='conv4-1', num_input_channels=32, filter_size=1,
                               num_output_channels=2,
                               padding='SAME',
                               use_relu=False)
        layer_factory.new_softmax(name='softmax', axis=3)
        layer_factory.new_conv(name='conv4-2', num_input_channels=32, filter_size=1,
                               num_output_channels=4,
                               padding='SAME',
                               use_relu=False,
                               input_layername='conv3')

    def _feed(self, image):
        feedimage = np.expand_dims(image, 0)
        return self._session.run(['pnet/conv4-2/BiasAdd:0', 'pnet/softmax:0'], feed_dict={'pnet/input:0': feedimage})

    def detectbox(self, image, threshold, vis=False):
        net_size = 12
        min_face_size = 24
        scale_factor = 0.709

        current_scale = float(net_size) / min_face_size
        im_resized = resize_image(image, current_scale)

        current_height, current_width, _ = image.shape

        all_boxes = list()

        while min(current_height, current_width) > net_size:
            reg, cls_map = self._feed(im_resized)
            boxes = generate_bounding_box(cls_map[0, :, :], reg, current_scale, threshold=threshold)

            current_scale *= scale_factor
            im_resized = resize_image(image, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue
            keep = nms(boxes[:, :5].copy(), 0.5, mode='Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None
        all_boxes = np.vstack(all_boxes)

        keep = nms(all_boxes[:, 0:5].copy(), 0.7, "Union")
        all_boxes = all_boxes[keep]

        bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        boxes = np.vstack([all_boxes[:, 0],
                           all_boxes[:, 1],
                           all_boxes[:, 2],
                           all_boxes[:, 3],
                           all_boxes[:, 4]
                           ])

        boxes = boxes.T

        aligin_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
        aligin_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
        aligin_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
        aligin_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh

        if vis:
            for i in range(aligin_topx.shape[0]):
                cv2.rectangle(image, (int(aligin_topx[i]), int(aligin_topy[i])),
                              (int(aligin_bottomx[i]), int(aligin_bottomy[i])), (200, 148, 0, 0.1), 2)
            cv2.imshow("0", image)
            cv2.waitKey(0)

        boxes_align = np.vstack([
            aligin_topx,
            aligin_topy,
            aligin_bottomx,
            aligin_bottomy,
            all_boxes[:, 4],
        ])
        boxes_align = boxes_align.T

        return boxes, boxes_align
