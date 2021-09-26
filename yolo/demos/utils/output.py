from queue import Queue
import threading as t
import abc
import cv2
import time


class FrameQue(object):

  def __init__(self, size=-1):
    self._que = Queue(size)
    return

  def put(self, frame):
    if self._que.full():
      return False
    self._que.put(frame)
    return True

  def put_all(self, frames):
    for frame in frames:
      if not self.put(frame):
        _ = self.get()
        self.put(frame)
    return True

  def get(self):
    if self._que.empty():
      return None
    return self._que.get()

  def read(self):
    if self._que.empty():
      return False, None
    return True, self._que.get()

  def full(self):
    return self._que.full()

  def empty(self):
    return self._que.empty()


class Thread(object):

  def __init__(self):
    __metaclass__ = abc.ABCMeta
    self._thread = None
    self._running = False
    self._wait_time = 0
    super().__init__()
    return

  def start(self):
    print("Starting Thread")
    self._running = True
    self._thread = t.Thread(target=self.display, args=())
    self._thread.start()
    return

  def close(self):
    print("Closing Thread")
    self._running = False
    if self._thread is not None:
      self._thread.join()
    return

  @property
  def running(self):
    return self._running

  def put(self, frame):
    return self._frame_buffer.put(frame)

  def put_all(self, frames):
    return self._frame_buffer.put_all(frames)

class DisplayThread(Thread):

  def __init__(self,
               frame_buffer_size=1,
               wait_time=0.0001):
    self._frame_buffer = FrameQue(frame_buffer_size)
    self._thread = None
    self._running = False
    self._wait_time = wait_time
    return

  def display(self):
    timeout, counter = 10/self._wait_time, 0
    try:
      while (self._running):
        success, frame = self._frame_buffer.read()

        while (not success):
          if counter > timeout:
            raise Exception("display has timed out.")
          time.sleep(self._wait_time)
          success, frame = self._frame_buffer.read()
          counter += 1
        counter = 0

        if success and frame is not None:
          cv2.imshow("frame", frame)
          if cv2.waitKey(1) & 0xFF == ord("q"):
            break
          del frame
      self._running = False
      cv2.destroyAllWindows()
    except Exception as e:
      print(e)
      self._running = False
      cv2.destroyAllWindows()
      raise Exception(e)
    return


class RouteThread(Thread):

  def __init__(self,
               width = 960, 
               height = 720,
               frame_buffer_size=1,
               wait_time=0.0001):
    import pyfakewebcam

    # ls /dev/video*
    # sudo modprobe -r v4l2loopback
    # sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="v4l2loopback" exclusive_caps=1

    cam = '/dev/video20'
    print(f"rounting to webcam {cam}") 
    self._fake_cam = pyfakewebcam.FakeWebcam(cam, width, height) 
    self._width = width 
    self._height = height

    self._frame_buffer = FrameQue(frame_buffer_size)
    self._thread = None
    self._running = False
    self._wait_time = wait_time
    return

  def display(self):
    timeout, counter = 10/self._wait_time, 0
    try:
      while (self._running):
        success, frame = self._frame_buffer.read()

        while (not success):
          if counter > timeout:
            raise Exception("display has timed out.")
          time.sleep(self._wait_time)
          success, frame = self._frame_buffer.read()
          counter += 1
        counter = 0

        if success and frame is not None:

          if frame.shape[1] != self._height:
            frame = cv2.resize(frame, (self._width, self._height))
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame = cv2.flip(frame, 1)
          self._fake_cam.schedule_frame(frame)
      self._running = False
      cv2.destroyAllWindows()
    except Exception as e:
      print(e)
      self._running = False
      cv2.destroyAllWindows()
      raise Exception(e)
    return


class SaveVideoThread(Thread):

  def __init__(self,
               save_file,
               fps, 
               width = 960, 
               height = 720,
               frame_buffer_size=1,
               wait_time=0.0001):

    self._save_file = save_file
    self._width = int(width)
    self._height = int(height)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    self._out = cv2.VideoWriter(
      save_file, fourcc, fps, [self._width, self._height])

    self._frame_buffer = FrameQue(frame_buffer_size)
    self._thread = None
    self._running = False
    self._wait_time = wait_time
    return

  def display(self):
    timeout, counter = 10/self._wait_time, 0
    try:
      while (self._running):
        success, frame = self._frame_buffer.read()

        while (not success):
          if counter > timeout:
            raise Exception("display has timed out.")
          time.sleep(self._wait_time)
          success, frame = self._frame_buffer.read()
          counter += 1
        counter = 0

        if success and frame is not None:
          if frame.shape[1] != self._height:
            frame = cv2.resize(frame, (self._width, self._height))
          self._out.write(frame)
      self._running = False
      cv2.destroyAllWindows()
    except Exception as e:
      print(e)
      self._running = False
      cv2.destroyAllWindows()
      raise Exception(e)
    return

  def start(self):
    super().start()
    return

  def close(self):
    super().close()
    self._out.release()
    return
