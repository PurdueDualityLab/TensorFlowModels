import socket
from yolo.utils.demos import utils
from queue import Queue
import threading as t
import numpy as np
import struct


class UDPServer(object):
  MAX_DGRAM_SIZE = 2**16 - 30
  IMAGE_DGRAM_SIZE = MAX_DGRAM_SIZE - 64

  def __init__(self, address="localhost", PORT=5005, que_size=10):

    self._address = address
    self._PORT = PORT
    self._serving_que = Queue(que_size)

    self._running = False
    self._thread = None
    self._socket = None
    return

  def serve(self, file):
    if (self._serving_que.full()):
      return False
    self._serving_que.put(file)
    return True

  def send_file(self, file, ip):
    size = len(file)
    num_segments = np.ceil(size / IMAGE_DGRAM_SIZE)
    start_pos = 0
    while (num_segments > 0):
      end_pos = min(start_pos + IMAGE_DGRAM_SIZE, size)
      self._socket.sendto(struct.pack("B", file[start_pos:end_pos]), ip)
      start_pos = end_pos
      num_segments -= 1
    return

  def start(self):
    self._socket = utils.udp_socket(self.address, self.PORT, server=True)
    self._thread = t.Thread(target=self.load_frames, args=())
    self._thread.start()
    return

  def get_file(self, sock):
    sock.sendto(b"GET", (self._address, self._PORT))
    data = b""
    while (True):
      seg, addr = s.recvfrom(MAX_DGRAM_SIZE)
      if struct.unpack("B", seg[0:1])[0] > 1:
        dat += seg[1:]
      else:
        dat += seg[1:]
        break
    return data

  def stop(self):
    self._running = False
    if self._thread is not None:
      self._thread.join()
    if self._socket is not None:
      self._socket.close()
    return


class UDPPacket(object):
  IMP_SIZE = 2**16 - 2**8

  def __init__(self,
               frame_id: int,
               sqn_number: int,
               num_packs: int,
               total_size: int,
               data: bytes = b""):
    self._frame_id = frame_id
    self._sqn_number = sqn_number
    self._num_packs = num_packs
    self._total_size = total_size
    self._data = data
    return

  @property
  def bytes(self):
    return self._frame_id.to_bytes(4, "little") + self._sqn_number.to_bytes(
        4, "little") + self._num_packs.to_bytes(
            4, "little") + self._total_size.to_bytes(4, "little") + self._data

  def __len__(self):
    return len(self.bytes)

  @staticmethod
  def split(id, byte_str):
    pack_list = []
    num_packets = int(np.ceil(len(byte_str) / UDPPacket.IMP_SIZE)) + 1
    i = 0
    start_pos = 0
    while (i < num_packets):
      end_pos = np.min([start_pos + UDPPacket.IMP_SIZE, len(byte_str)])
      pack_list.append(
          UDPPacket(id, i, num_packets, len(byte_str),
                    byte_str[start_pos:end_pos]))
      start_pos = end_pos
      i += 1
    return pack_list

  @staticmethod
  def decode_bytes(byte_str):
    id = int.from_bytes(byte_str[0:4], "little")
    sqn_number = int.from_bytes(byte_str[4:8], "little")
    num_packets = int.from_bytes(byte_str[8:12], "little")
    total_size = int.from_bytes(byte_str[12:16], "little")
    data = byte_str[16:]
    return UDPPacket(id, sqn_number, num_packets, data)

  @property
  def sequence_number(self):
    return self._sqn_number

  @property
  def image_ID(self):
    return self._frame_id

  @property
  def num_packets(self):
    return self._num_packs

  @property
  def total_size(self):
    return self._total_size

  @property
  def data(self):
    return self._data

  @staticmethod
  def reconstruct(packet_list):
    if len(packet_list) < 0 and packet_list is None:
      return None
    num_packets = packet_list[-1].num_packets
    if len(packet_list) < num_packets:
      return None

    total_size = packet_list[-1].total_size
    packets = [None] * num_packets
    for packet in packet_list:
      packets[packet.sequence_number] = packet

    bdata = b""
    for i, packet in enumerate(packets):
      # if packet == None:
      #     if i != num_packets - 1:
      #         bdata += int(0).to_bytes(UDPPacket.IMP_SIZE, "little")
      #     else:
      #         bdata += int(0).to_bytes(total_size - (num_packets - 1) * UDPPacket.IMP_SIZE, "little")
      # else:
      bdata += packet.data
    return bdata


if __name__ == "__main__":
  import tensorflow as tf
  import tensorflow_datasets as tfds
  import matplotlib.pyplot as plt
  ds = tfds.load("coco", split="train")
  sample = ds.take(1)

  image = None
  jpg = None
  for data in sample:
    image = data["image"]
    image = tf.image.encode_jpeg(image)
    jpg = image.numpy()

  pack = UDPPacket.split(0, jpg)
  print(pack[3].num_packets)
  #del pack[2]

  # print(pack)
  pack[0], pack[2] = pack[2], pack[0]
  pack[0], pack[3] = pack[3], pack[0]
  # print(UDPPacket.decode_bytes(pack[2].bytes).num_packets)

  image = UDPPacket.reconstruct(pack)
  image = tf.io.decode_jpeg(image)
  plt.imshow(image)
  plt.show()

  # 2 udp servers, one for reciveing frames, one for sending frames. to reduce latency
  #
  # server 1: client send an image by splitting into packets, then store the packets in a dictionary with the the frame_id as the key
  # add frames until the dict[packet.image_ID] has same # of packets as packet.num_packets. when it does. then remove the key and reconstruct the packet.
  # then add it to the model serving que to be processed
  # server 2: take a frame from the model serving que and split it into packets, and send it back to the client.
  #
  # client 1: send frames
  # client 2: recieve frames
