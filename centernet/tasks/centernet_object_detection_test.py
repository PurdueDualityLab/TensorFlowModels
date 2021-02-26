import tensorflow as tf
import centernet_object_detection as centernet_object_detection
import centernet.tasks as tasks
import centernet.utils as utils

class ObjectDetectionTest(tf.test.TestCase):

    def generate_heatmaps(self, dectections):

        for ind, detection in enumerate(detections):
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
                width  = detection[2] - detection[0]
                height = detection[3] - detection[1]

                width  = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)

                if gaussian_rad == -1:
                    radius = utils.gaussian_radius((height, width), gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad

                utils.draw_gaussian(tl_heatmaps[category], [xtl, ytl], radius)
                utils.draw_gaussian(br_heatmaps[category], [xbr, ybr], radius)
                utils.draw_gaussian(ct_heatmaps[category], [xct, yct], radius, delte = 5)