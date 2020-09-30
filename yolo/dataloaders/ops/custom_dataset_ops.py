import tensorflow as tf


def _yolo_coco_id_parser(value):
    #im = value.split('/')[-1].split(".")[0].split("_")[-1]
    value = tf.strings.split(input=tf.strings.split(input=tf.strings.split(
        input=value, sep="/")[-1],
                                                    sep=".")[0],
                             sep="_")[-1]
    return tf.strings.join([value, ".*"])


def _yolo_coco_label_parser(id, paths):
    #im = value.split('/')[-1].split(".")[0].split("_")[-1]
    ret_dict = dict.fromkeys(["bbox", "labels"])
    for path in paths:
        path = tf.strings.join([path, "/", id])
        ret_dict["bbox"] = path
        ret_dict["labels"] = path
    return ret_dict


def _get_images(file_paths):
    paths = []
    for path_instance in file_paths:
        path_instance = path_instance.strip().split(",")
        for path in path_instance:
            if "*" in path or "?" in path:
                temp_paths = tf.io.gfile.glob(path)
                if temp_paths == None:
                    raise IOError("no matches found to dataset path")
                paths.extend(temp_paths)
            else:
                paths.append(path)
    return paths


def _str_to_list(path):
    if isinstance(path, Dict):
        return paths
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path
    return paths
