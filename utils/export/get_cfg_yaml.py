from yolo.configs import yolo as exp_cfg
import yaml

config = exp_cfg.yolo_v4_coco()

e = yaml.dump(config.as_dict(), default_flow_style=False)
with open("yolo/configs/experiments/yolo.yaml", "w") as f:
  f.write(e)

f = yaml.load(e)
config.override(f, is_strict=False)
