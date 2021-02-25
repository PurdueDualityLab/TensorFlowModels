import tensorflow as tf
from official.vision.beta.dataloaders import parser


class CenterNetParser(parser.Parser):
    def __init__(
        self,
        num_classes: int,
        max_num_instances: int,
        gaussian_iou: float
    ):
        self._num_classes = num_classes
        self._max_num_instances = max_num_instances
        self._gaussian_iou = gaussian_iou

    def _parse_train_data(self, decoded_tensors):
        """Generates images and labels that are usable for model training.

        Args:
            decoded_tensors: a dict of Tensors produced by the decoder.

        Returns:
            images: the image tensor.
            labels: a dict of Tensors that contains labels.
        """
        tl_heatmaps = tf.zeros((self._num_classes, output_size[0], output_size[1]), dtype=tf.float32)
        br_heatmaps = tf.zeros((self._num_classes, output_size[0], output_size[1]), dtype=tf.float32)
        ct_heatmaps = tf.zeros((self._num_classes, output_size[0], output_size[1]), dtype=tf.float32)
        tl_regrs = tf.zeros((self._max_num_instances, 2), dtype=tf.float32)
        br_regrs = tf.zeros((self._max_num_instances, 2), dtype=tf.float32)
        ct_regrs = tf.zeros((self._max_num_instances, 2), dtype=tf.float32)
        tl_tags = tf.zeros((self._max_num_instances), dtype=tf.int64)
        br_tags = tf.zeros((self._max_num_instances), dtype=tf.int64)
        ct_tags = tf.zeros((self._max_num_instances), dtype=tf.int64)
        tag_masks = tf.zeros((self._max_num_instances), dtype=tf.uint8)

        # TODO: input size, output size
        image = decoded_tensors["image"]

        width_ratio = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        for ind, detection in enumerate(decoded_tensors["groundtruth_boxes"]):
            category = int(detection[-1]) - 1
            # category = 0

            xtl, ytl = detection[0], detection[1]
            xbr, ybr = detection[2], detection[3]

            xct, yct = (
                (detection[2] + detection[0]) / 2,
                (detection[3] + detection[1]) / 2
            )

            fxtl = (xtl * width_ratio)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)
            fxct = (xct * width_ratio)
            fyct = (yct * height_ratio)

            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)
            xct = int(fxct)
            yct = int(fyct)

            if gaussian_bump:
                width = detection[2] - detection[0]
                height = detection[3] - detection[1]

                width = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)

                if gaussian_rad == -1:
                    radius = gaussian_radius((height, width), self._gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad
                tl_heatmaps = draw_gaussian(tl_heatmaps, category, [xtl, ytl], radius)
                br_heatmaps = draw_gaussian(br_heatmaps, category, [xbr, ybr], radius)
                ct_heatmaps = draw_gaussian(ct_heatmaps, category, [xct, yct], radius, scaling_factor=5)

            else:
                # tl_heatmaps[category, ytl, xtl] = 1
                # br_heatmaps[category, ybr, xbr] = 1
                # ct_heatmaps[category, yct, xct] = 1
                tl_heatmaps = tf.tensor_scatter_nd_update(tl_heatmaps, [[category, ytl, xtl]], [1])
                br_heatmaps = tf.tensor_scatter_nd_update(br_heatmaps, [[category, ytl, xtl]], [1])
                ct_heatmaps = tf.tensor_scatter_nd_update(ct_heatmaps, [[category, ytl, xtl]], [1])

            # tl_regrs[tag_ind, :] = [fxtl - xtl, fytl - ytl]
            # br_regrs[tag_ind, :] = [fxbr - xbr, fybr - ybr]
            # ct_regrs[tag_ind, :] = [fxct - xct, fyct - yct]
            # tl_tags[tag_ind] = ytl * output_size[1] + xtl
            # br_tags[tag_ind] = ybr * output_size[1] + xbr
            # ct_tags[tag_ind] = yct * output_size[1] + xct
            tl_regrs = tf.tensor_scatter_nd_update(tl_regrs, [[tag_ind, 0], [tag_ind, 1]], [fxtl - xtl, fytl - ytl])
            br_regrs = tf.tensor_scatter_nd_update(br_regrs, [[tag_ind, 0], [tag_ind, 1]], [fxbr - xbr, fybr - ybr])
            ct_regrs = tf.tensor_scatter_nd_update(ct_regrs, [[tag_ind, 0], [tag_ind, 1]], [fxct - xct, fyct - yct])
            tl_tags = tf.tensor_scatter_nd_update(tl_tags, [[tag_ind]], [ytl * output_size[1] + xtl])
            br_tags = tf.tensor_scatter_nd_update(br_tags, [[tag_ind]], [ybr * output_size[1] + xbr])
            ct_tags = tf.tensor_scatter_nd_update(ct_tags, [[tag_ind]], [yct * output_size[1] + xct])

        labels = {
            'tl_tags': tl_tags,
            'br_tags': br_tags,
            'ct_tags': ct_tags,
            'tl_heatmaps': tl_heatmaps,
            'br_heatmaps': br_heatmaps,
            'ct_heatmaps': ct_heatmaps,
            'tag_masks': tag_masks,
            'tl_regrs': tl_regrs,
            'br_regrs', br_regrs,
            'ct_regrs': ct_regrs,
        }
        return image, labels

    def _parse_eval_data(self, data):
        pass
