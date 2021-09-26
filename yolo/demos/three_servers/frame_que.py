from queue import Queue
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
