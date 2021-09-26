import tensorflow as tf
import cv2
import time



class Statistics:

  def __init__(self, function, alpha = 0.9):
    self._function = function
    self._alpha = alpha

    self._start = None
    self._fps_counter = 0
    self._clock_tick = 0 
    self._fps = 0
    self._latency = 0

  def __call__(self, *args, **kwargs):
    if self._start is None:
      self._start = time.time()

    pre_call = time.time()
    output = self._function(*args, **kwargs)
    post_call = time.time()
    self._fps_counter += 1

    if time.time() - self._start - self._clock_tick >= 1: 
      self._clock_tick += 1
      self._fps = self._fps_counter
      self._fps_counter = 0
    
    self._latency = self._alpha * self._latency + (1 - self._alpha) * (post_call - pre_call)
    return output
  
  @property
  def fps(self):
    return self._fps

  @property
  def latency(self):
    return self._latency

class VideoFile:

  def __init__(self, 
               file_name = 0, 
               batch_size = 1, 
               buffer_size = 0, 
               output_resolution = None, 
               include_statistics = False):
    self._file = cv2.VideoCapture(file_name)

    self._incude_statistics = include_statistics
    self._batch_size = batch_size
    self._output_resolution = output_resolution
    self._buffer_size = buffer_size

    self._width = int(self._file.get(3))
    self._height = int(self._file.get(4))

    if output_resolution is not None:
      og_height = self._height
      self._height = output_resolution
      self._width = int(self._width * (output_resolution / og_height))

    if include_statistics:
      self._read = Statistics(self._file.read)
    else: 
      self._read = self._file.read
    return 
  
  def _read_generator(self):
    while (self._file.isOpened()):
      success, image = self._read()
      if success:
        if self._output_resolution is not None:
          if image.shape[0] != self._height or image.shape[1] != self._width: 
            image = cv2.resize(image, (self._width, self._height))
        yield tf.convert_to_tensor(image)

  def get_generator(self):
    video = tf.data.Dataset.from_generator(self._read_generator, 
              output_signature=(
                tf.TensorSpec(shape = [None, None, 3], dtype = tf.uint8)
              ))
    if self._buffer_size == 0: # realtime 
      video = video.batch(self._batch_size, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True)
    else:
      video = video.prefetch(self._buffer_size).batch(self._batch_size, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True)
    return video
  
  @property
  def fps(self):
    if self._incude_statistics:
      return self._read.fps
    return None 
  
  @property
  def latency(self):
    if self._incude_statistics:
      return self._read.latency
    return None 

  @property
  def width(self):
    return self._width

  @property
  def height(self):
    return self._height
  
  @property
  def file_fps(self):
    return self._file.get(cv2.CAP_PROP_FPS)

  @property
  def playing(self):
    return self._file.isOpen()