import socket 
from yolo.utils.demos import utils 
from queue import Queue
import threading as t
import numpy as np
import struct

class UDPPacket(object):
    def __init__(self, sqn_number, data):
        self._sqn_number = sqn_number
        self._data = data
        return 


class UDPServer(object):
    MAX_DGRAM_SIZE = 2**16 - 30
    IMAGE_DGRAM_SIZE = MAX_DGRAM_SIZE - 64
    def __init__(self,
                 address = "localhost", 
                 PORT = 5005, 
                 que_size = 10):

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
        num_segments = np.ceil(size/IMAGE_DGRAM_SIZE)
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
        data = b''
        while(True):
            seg, addr = s.recvfrom(MAX_DGRAM_SIZE)
            if struct.unpack("B", seg[0:1])[0]>1:
                dat += seg[1:]
            else:
                dat += seg[1:]
                break
        return data
    
    def stop(self):
        self._running = False
        if self._thread != None:
            self._thread.join()
        if self._socket != None:
            self._socket.close()
        return
        
        