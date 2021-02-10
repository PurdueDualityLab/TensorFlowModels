import tensorflow as tf
from official.vision.beta.dataloaders import parser


class CenterNetParser(parser.Parser):
    def __init__(
        self,
        num_classes: int,
        gaussian_iou: float
    ):
        self.num_classes = num_classes
        self.gaussian_io = gaussian_iou

    def _parse_train_data(self, decoded_tensors):
        """Generates images and labels that are usable for model training.

        Args:
            decoded_tensors: a dict of Tensors produced by the decoder.

        Returns:
            images: the image tensor.
            labels: a dict of Tensors that contains labels.
        """
        # TODO: input size, output size
        image = decoded_tensors["image"]

        width_ratio = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        for ind, detection in enumerate(decoded_tensors["groundtruth_boxes"]):
            category = int(detection[-1]) - 1
            #category = 0

            xtl, ytl = detection[0], detection[1]
            xbr, ybr = detection[2], detection[3]
            xct, yct = (detection[2] + detection[0])/2., (detection[3]+detection[1])/2.

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
                    radius = gaussian_radius((height, width), self.gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad
                # TODO: implement gaussian
                # draw_gaussian(tl_heatmaps[b_ind, category], [xtl, ytl], radius)
                # draw_gaussian(br_heatmaps[b_ind, category], [xbr, ybr], radius)
                # draw_gaussian(ct_heatmaps[b_ind, category], [xct, yct], radius, delte=5)

            else:
                tl_heatmaps[category, ytl, xtl] = 1
                br_heatmaps[category, ybr, xbr] = 1
                ct_heatmaps[category, yct, xct] = 1

            tag_ind = tag_lens
            tl_regrs[tag_ind, :] = [fxtl - xtl, fytl - ytl]
            br_regrs[tag_ind, :] = [fxbr - xbr, fybr - ybr]
            ct_regrs[tag_ind, :] = [fxct - xct, fyct - yct]
            tl_tags[tag_ind] = ytl * output_size[1] + xtl
            br_tags[tag_ind] = ybr * output_size[1] + xbr
            ct_tags[tag_ind] = yct * output_size[1] + xct
            tag_lens += 1

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
