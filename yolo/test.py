from yolo.utils.run_utils import expand_gpu, prep_gpu

expand_gpu()
try:
  prep_gpu()
except BaseException:
  print("GPUs ready")
