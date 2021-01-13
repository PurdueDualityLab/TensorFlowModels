from queue import Queue


class FrameQue(object):

  def __init__(self, size):
    self._que = Queue(size)
    return

  def put(self, frame):
    if self._que.full():
      return False
    self._que.put(frame)
    return True

  def get(self):
    if self._que.empty():
      return None
    return self._que.get()

  def full(self):
    return self._que.full()

  def empty(self):
    return self._que.empty()
