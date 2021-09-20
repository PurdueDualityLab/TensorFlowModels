import cv2
import threading as t
from queue import Queue
import traceback
import time

from yolo.demos.three_servers.frame_que import FrameQue
from yolo.utils.demos import utils


class VideoServer(object):

  def __init__(self,
               file=0,
               disp_h=720,
               wait_time=0.001,
               que=10,
               post_process=None):

    self._file = file
    self._cap = cv2.VideoCapture(file)
    if not self._cap.isOpened():
      raise IOError("video file was not found")

    print(f"frame dims: {self._cap.get(3)}, {self._cap.get(4)}")
    self._og_height = int(self._cap.get(4))
    self._height = int(self._cap.get(4)) if disp_h is None else disp_h
    self._width = int(self._cap.get(3) * (self._height / self._og_height))

    self._wait_time = utils.get_wait_time(wait_time, 1)

    if isinstance(que, int):
      self._ret_que = FrameQue(que)
    else:
      self._ret_que = que

    self._running = False
    self._read_fps = 0
    self._load_thread = None
    self._send_thread = None
    self._socket = None
    self._postprocess_fn = post_process
    return

  @property
  def address(self):
    return self._address

  @property
  def PORT(self):
    return self._PORT

  @property
  def running(self):
    return self._running

  @running.setter
  def running(self, value):
    self._running = value

  def load_frames(self):
    start = time.time()
    l = 0
    tick = 0
    timeout = 0
    try:
      self._running = True
      while (self._cap.isOpened() and self._running):
        # wait for que to get some space,
        if (self._ret_que.full()):
          time.sleep(self._wait_time)
          continue

        success, image = self._cap.read()
        if not success:
          break

        if type(self._file) == int:
          image = cv2.flip(image, 1)
        image = cv2.resize(
            image, (self._width, self._height), interpolation=cv2.INTER_AREA)
        image = image / 255

        if self._postprocess_fn is not None:
          image = self._postprocess_fn(image)
        # then dump the image on the que
        self._ret_que.put(image)

        # compute the reading FPS
        l += 1
        if time.time() - start - tick >= 1:
          tick += 1
          # store the reading FPS so it can be printed clearly
          self._read_fps = l
          l = 0
        # sleep for default 0.01 seconds, to allow other functions the time to catch up or keep pace
        time.sleep(self._wait_time)
        #print("                                 \rread fps: \033[1;34;40m%d\033[0m " % (self._read_fps), end="\n")
      self._cap.release()
      self._running = False
    except KeyboardInterrupt:
      self._cap.release()
      self._running = False
    return

  def start(self):
    self._load_thread = t.Thread(target=self.load_frames, args=())
    self._load_thread.start()
    return self._load_thread

  def close(self, blocking=True):
    if not blocking:
      self._running = False
      self._cap.release()
    if self._load_thread is not None:
      self._load_thread.join()
    return

  def get(self):
    return self._ret_que.get()

  def read(self):
    return self._running, self.get()

  @property
  def wait_time(self):
    return self._wait_time

  @wait_time.setter
  def wait_time(self, value):
    self._wait_time = value


class VideoPlayer(object):

  def __init__(self,
               file=0,
               disp_h=720,
               wait_time=0.001,
               que=10,
               post_process=None):

    self._file = file
    self._cap = cv2.VideoCapture(file)
    if not self._cap.isOpened():
      raise IOError("video file was not found")

    print(f"frame dims: {self._cap.get(3)}, {self._cap.get(4)}")
    self._og_height = int(self._cap.get(4))
    self._height = int(self._cap.get(4)) if disp_h is None else disp_h
    self._width = int(self._cap.get(3) * (self._height / self._og_height))
    self._postprocess_fn = post_process
    return

  def get(self):
    return self.read()[1]

  def read(self):
    suc, image = self._cap.read()
    if suc:
      image = cv2.resize(
          image, (self._width, self._height), interpolation=cv2.INTER_AREA)
      image = image / 255
      if self._postprocess_fn is not None:
        image = self._postprocess_fn(image)
    return suc, image

  @property
  def running(self):
    return self._cap.isOpened()

  def start(self):
    return


# class DisplayThread(object):

#   def __init__(self,
#                frame_buffer=None,
#                wait_time=0.0001,
#                alpha=0.1,
#                fix_wt=False,
#                use_colab=False):
#     if frame_buffer is None:
#       self._frame_buffer = FrameQue(1000)
#     else:
#       self._frame_buffer = frame_buffer
#     self._thread = None
#     self._running = False
#     self._wait_time = wait_time
#     self._fps = 0
#     self._prev_fps = 0
#     self._fix_wt = fix_wt
#     self._alpha = alpha
#     self._use_colab = use_colab
#     return

#   def start(self):
#     self._running = True
#     self._thread = t.Thread(target=self.display, args=())
#     self._thread.start()
#     return

#   def close(self):
#     print()
#     self._running = False
#     if self._thread is not None:
#       self._thread.join()
#     return

#   def display(self):
#     try:
#       start = time.time()
#       l = 0
#       tick = 0
#       display_fps = 0

#       while (self._running):
#         # success, frame = self._frame_buffer.read()
#         success, frame = self._frame_buffer.read()
#         if success and frame is not None:
#           cv2.imshow("frame", frame)

#           if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#           del frame
#           l += 1
#           if time.time() - start - tick >= 1:
#             tick += 1
#             # store the fps to diplayed to the user
#             self._prev_fps = self._fps * self._alpha + self._prev_fps * (
#                 1 - self._alpha)
#             self._fps = l
#             l = 0

#           if not self._fix_wt:
#             if (self._prev_fps <= self._fps):
#               self._wait_time = self._wait_time - 0.02 * self._wait_time
#             else:
#               self._wait_time = self._wait_time + 0.01 * self._wait_time

#           if hasattr(self._frame_buffer, "wait_time"):
#             self._frame_buffer.wait_time = self._wait_time

#         time.sleep(self._wait_time)
#       self._running = False
#       cv2.destroyAllWindows()
#     except Exception as e:
#       print(e)
#       self._running = False
#       cv2.destroyAllWindows()
#     return

#   @property
#   def running(self):
#     return self._running

#   def put(self, frame):
#     return self._frame_buffer.put(frame)

#   def put_all(self, frames):
#     return self._frame_buffer.put_all(frames)

#   @property
#   def fps(self):
#     return self._fps


class Thread(object):

  def __init__(self):
    __metaclass__ = abc.ABCMeta
    self._thread = None
    self._running = False
    self._wait_time = 0
    super().__init__()
    return

  def start(self):
    pass

  def close(self):
    pass

  def running(self):
    pass

  def put(self):
    pass

  def put_all(self):
    pass


class DisplayThread(Thread):

  def __init__(self,
               frame_buffer=None,
               wait_time=0.0001,
               alpha=0.1,
               fix_wt=False,
               res = None, 
               use_colab=False):
    if frame_buffer is None:
      self._frame_buffer = FrameQue(1000)
    else:
      self._frame_buffer = frame_buffer
    self._thread = None
    self._running = False
    self._wait_time = wait_time
    self._fps = 0
    self._prev_fps = 0
    self._fix_wt = fix_wt
    self._alpha = alpha
    self._use_colab = use_colab
    self._res = res
    return

  def start(self):
    self._running = True
    self._thread = t.Thread(target=self.display, args=())
    self._thread.start()
    return

  def close(self):
    print()
    self._running = False
    if self._thread is not None:
      self._thread.join()
    return

  def display(self):
    if self._res is not None:
      fourcc = cv2.VideoWriter_fourcc(*'MJPG')
      out = cv2.VideoWriter("../../Videos/basketball.avi", fourcc,
                          45, [int(k) for k in self._res])
    try:
      start = time.time()
      l = 0
      tick = 0
      display_fps = 0

      while (self._running):
        # success, frame = self._frame_buffer.read()
        success, frame = self._frame_buffer.read()
        if success and frame is not None:
          if self._res[-1] <= 1080:
            cv2.imshow("frame", frame)
          if self._res is not None:
            frame = cv2.resize(frame, self._res)
            frame = cv2.convertScaleAbs(frame, alpha=(255.0))
            out.write(frame)

          if cv2.waitKey(1) & 0xFF == ord("q"):
            break

          del frame
          l += 1
          if time.time() - start - tick >= 1:
            tick += 1
            # store the fps to diplayed to the user
            self._prev_fps = self._fps * self._alpha + self._prev_fps * (
                1 - self._alpha)
            self._fps = l
            l = 0

          if not self._fix_wt:
            if (self._prev_fps <= self._fps):
              self._wait_time = self._wait_time - 0.02 * self._wait_time
            else:
              self._wait_time = self._wait_time + 0.01 * self._wait_time

          if hasattr(self._frame_buffer, "wait_time"):
            self._frame_buffer.wait_time = self._wait_time

        time.sleep(self._wait_time)
      if self._res is not None:
        out.release()
      self._running = False
      cv2.destroyAllWindows()
    except Exception as e:
      print(e)
      if self._res is not None:
        out.release()
      self._running = False
      cv2.destroyAllWindows()
    return

  @property
  def running(self):
    return self._running

  def put(self, frame):
    return self._frame_buffer.put(frame)

  def put_all(self, frames):
    return self._frame_buffer.put_all(frames)

  @property
  def fps(self):
    return self._fps


class saveFrameThread(Thread):

  def __init__(
      self,
      folder_name="frames/",
      frame_buffer=None,
  ):
    if frame_buffer is None:
      self._frame_buffer = FrameQue(1000)
    else:
      self._frame_buffer = frame_buffer
    self._thread = None
    self._running = False
    self._fps = 0
    self._foldername = folder_name
    self._count = 0

  def start(self):
    self._running = True
    self._thread = t.Thread(target=self.save, args=())
    self._thread.start()

  def close(self):
    self._running = False
    if self._thread is not None:
      self._thread.join()

  def save(self):
    self._count = 0
    try:
      while self._running:
        success, frame = self._frame_buffer.read()
        if success and frame is not None:
          frame = cv2.convertScaleAbs(frame, alpha=(255.0))
          plt.imsave(f'{self._foldername}/{str(self._count)}.jpg', frame)
          self._count = self._count + 1
      self._running = False
    except Exception as e:
      print(e)
      self._running = False

  @property
  def running(self):
    return self._running

  def put(self, frame):
    return self._frame_buffer.put(frame)

  def put_all(self, frames):
    return self._frame_buffer.put_all(frames)

  @property
  def fps(self):
    return self._fps


class saveVideoThread(Thread):

  def __init__(
      self,
      frame_buffer=None,
      file_name="demo.avi",
      draw_fps=30,
      resolution=(739, 416),
  ):
    if frame_buffer is None:
      self._frame_buffer = FrameQue(1000)
    else:
      self._frame_buffer = frame_buffer
    self._thread = None
    self._running = False
    self._file_name = file_name
    self._draw_fps = draw_fps
    self._draw_write_res = resolution
    self._fps = 0
    self._count = 0

  def start(self):
    self._running = True
    self._thread = t.Thread(target=self.save, args=())
    self._thread.start()

  def close(self):
    self._running = False
    if self._thread is not None:
      self._thread.join()

  def save(self):
    self._count = 0
    out = cv2.VideoWriter(self._file_name, cv2.VideoWriter_fourcc(*'DIVX'),
                          self._draw_fps, self._draw_write_res)
    try:
      while self._running:
        success, frame = self._frame_buffer.read()
        print(success)
        if success and frame is not None:
          frame = cv2.convertScaleAbs(frame, alpha=(255.0))
          out.write(frame)
      self._running = False
    except Exception as e:
      print(e)
      self._running = False

  @property
  def running(self):
    return self._running

  def put(self, frame):
    return self._frame_buffer.put(frame)

  def put_all(self, frames):
    return self._frame_buffer.put_all(frames)

  @property
  def fps(self):
    return self._fps


if __name__ == "__main__":
  video = VideoServer("videos/nyc.mp4", wait_time=0.000001, que=10)
  display = DisplayThread(video, wait_time=0.0001)
  video.start()
  display.start()

  while (display.running):
    time.sleep(0.001)

  video.close(blocking=False)
  display.close()
