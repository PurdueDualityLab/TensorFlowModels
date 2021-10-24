import tensorflow as tf

def window_partition(x, window_size):
  """
  Args:
    x: (B, H, W, C)
    window_size (int): window size

  Returns:
    windows: (num_windows*B, window_size, window_size, C)
  """
  _, H, W, C = x.shape
  if isinstance(window_size, int):
    window_size = (window_size, window_size)
  x = tf.reshape(x, [-1, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C])
  x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
  x = tf.reshape(x, [-1, window_size[0], window_size[1], C])
  return x, H // window_size[0], W // window_size[1]

def window_reverse(x, window_size, H, W):
  """
  Args:
    windows: (num_windows*B, window_size, window_size, C)
    window_size (int): Window size
    H (int): Height of image
    W (int): Width of image

  Returns:
    x: (B, H, W, C)
  """
  _, _, _, C = x.shape
  if isinstance(window_size, int):
    window_size = (window_size, window_size)
  x = tf.reshape(x, [-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C])
  x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
  x = tf.reshape(x, [-1, H, W, C])
  return x

def pad(x, window_size):
  _, H, W, C = x.shape
  pad_l = pad_t = 0 
  pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
  pad_b = (window_size[0] - H % window_size[0]) % window_size[0]  
  x = tf.pad(x, [[0,0], [pad_t, pad_b], [pad_l, pad_r], [0, 0]]) 
  _, Hp, Wp, _ = x.shape
  return x, H, W, C, Hp, Wp

def pad_and_shift_input(x, window_size, shift_size, shift):
  x, H, W, C, Hp, Wp = pad(x, window_size)  

  if shift  == True:
    shifts = [(0, 0), (shift_size[0], shift_size[1])] # 6 ms latency, 9 ms latency 
  elif shift is None:
    shifts = [(shift_size[0], shift_size[1])]
  else:
    shifts = [(0, 0)] # 6 ms latency

  windows = []
  bsize = []
  for shift in shifts:
    # cyclic shift 
    if shift[0] != 0 or shift[1] != 0:
      shifted_x = x[:, (shift[0]):(Hp - (window_size[0] - shift[0])), (shift[1]):(Wp - (window_size[1] - shift[1])), :]
      attn_mask = None
    else:
      shifted_x = x
      attn_mask = None 

    x_windows, Ph, Pw = window_partition(shifted_x, window_size) # nW*B, window_size, window_size, C
    windows.append(x_windows)

    nwin = tf.shape(x_windows)
    bsize.append(nwin[0])
  x_windows = tf.concat(windows, axis = 0)
  return x, x_windows, shifts, bsize, H, W, C, Hp, Wp

def upad_and_unshift(attn_windows, split_sizes, Hp, Wp, window_size, shifts, H, W):
  x_output = None
  windows = tf.split(attn_windows, split_sizes, axis = 0)
  for shift, attn_windows in zip(shifts, windows):
    if shift[0] != 0 or shift[1] != 0:
      shifted_x = window_reverse(attn_windows, window_size, (Hp - window_size[0]), (Wp - window_size[1])) # B H' W' C
      shifted_x = tf.pad(shifted_x, [[0,0], [shift[0], window_size[0] - shift[0]], [shift[1], window_size[1] - shift[1]], [0, 0]]) 
    else:
      shifted_x = window_reverse(attn_windows, window_size, Hp, Wp) # B H' W' C

    if x_output is None:
      x_output = shifted_x
    else:
      x_output = x_output + shifted_x

  if Hp != H or Wp != W:
    x_output = x_output[:, :H, :W, :]
  return x_output

def pad_one_window(x, window_size):
  pad_l = window_size[1]
  pad_t = window_size[0]
  pad_r = window_size[1]
  pad_b = window_size[0]
  x = tf.pad(x, [[0,0], [pad_t, pad_b], [pad_l, pad_r], [0, 0]]) 
  return x

def roll_image_up_tlbr(image, P):
  B, Wh, Ww, C = image.shape

  image = tf.reshape(image, [-1, P, Wh, Ww, C])
  image_rollp1 = image[:, 1:, ...] 
  image_rollp1 = tf.pad(image_rollp1, [[0,0],[0, 1],[0,0],[0,0],[0,0]])
  image_rollp1 = image_rollp1[:, :, :, :-Ww//2, ...]

  image_rolln1 = image[:, :-1, ...] 
  image_rolln1 = tf.pad(image_rolln1, [[0,0],[1, 0],[0,0],[0,0],[0,0]])
  image_rolln1 = image_rolln1[:, :, :, Ww//2:, ...]

  image_cat1 = tf.concat([image_rolln1, image, image_rollp1], axis = -2)
  image_rollp2 = image_cat1[:, int(math.sqrt(P)):, ...] 
  image_rollp2 = tf.pad(image_rollp2, [[0,0],[0, int(math.sqrt(P))],[0,0],[0,0],[0,0]])

  image_rolln2 = image_cat1[:, :-int(math.sqrt(P)), ...] 
  image_rolln2 = tf.pad(image_rolln2, [[0,0],[int(math.sqrt(P)), 0],[0,0],[0,0],[0,0]])
  image = tf.concat([image_rolln2, image_cat1, image_rollp2], axis = -3)
  
  image = tf.reshape(image, [-1, Ww * 2, Wh * 3, C])
  return image

def window_partition_overlaps_tlbr(image, window_size, crop = True):
  image = pad_one_window(image, window_size)
  image, _,_,_,_,_ = pad(image, window_size)
  x, Ph, Pw = window_partition(image, window_size)
  _,_,_,C = x.shape
  x = roll_image_up_tlbr(x, Ph * Pw)
  x = tf.reshape(x, [-1, Ph, Pw, window_size[0] * 3, 2 * window_size[1], C])
  x = x[:, 1:-1, 1:-1, ...]
  x = tf.reshape(x, [-1, window_size[0] * 3 , 2 * window_size[1], C])
  Ph, Pw = 2 * Ph, 3 *Pw

  # if crop:
  #   p1 = window_size[0] * 3
  #   p2 = window_size[1] * 3

  #   s1 = int(window_size[0] * 2)
  #   s2 = int(window_size[1] * 2)

  #   t1 = (p1 - s1)//2
  #   t2 = (p2 - s2)//2
  #   x = x[:, t1:-t1, t2:-t2, :]

  #   #x = tf.slice(x, [0,(p1 - s1)//2, (p2 - s2)//2,0], [-1, s1, s2, -1])

  #   Ph, Pw = 2 * Ph, 2 *Pw
  return x, Ph, Pw

def roll_image_up_br(image, P):
  B, Wh, Ww, C = image.shape

  image = tf.reshape(image, [-1, P, Ww, Wh, C])
  image_roll1 = image[:, 1:, ...] 
  image_roll1 = tf.pad(image_roll1, [[0,0],[0, 1],[0,0],[0,0],[0,0]])
  image_roll1.shape

  image_cat1 = tf.concat([image, image_roll1], axis = -2)

  image_roll2 = image_cat1[:, int(math.sqrt(P)):, ...] 
  image_roll2 = tf.pad(image_roll2, [[0,0],[0, int(math.sqrt(P))],[0,0],[0,0],[0,0]])
  image = tf.concat([image_cat1, image_roll2], axis = -3)
  
  image = tf.reshape(image, [-1, Ww * 2, Wh * 2, C])
  return image

def window_partition_overlaps_br(image, window_size):
  image = pad_one_window(image, window_size)
  image, _,_,_,_,_ = pad(image, window_size)
  x, Ph, Pw = window_partition(image, window_size)

  _,_,_,C = x.shape
  x = roll_image_up_br(x, Ph * Pw)
  x = tf.reshape(x, [-1, Ph, Pw, 2 * window_size[0], 2 * window_size[1], C])
  x = x[:, 1:-1, 1:-1, ...]
  x = tf.reshape(x, [-1, 2 * window_size[0], 2 * window_size[1], C])
  return x, Ph, Pw

### import testing_fns
import matplotlib.pyplot as plt
import math
from skimage import io
def url_to_image(url):
  image = io.imread(url)
  return image

def draw_patches(patches, batch_size, dim = 0):
  B, H, W, C = patches.shape

  im = B//batch_size
  bh = bw = int(math.sqrt(im))
  
  fig, axe = plt.subplots(bh, bw)

  k = im * dim
  for i in range(bh):
    for j in range(bw):
      axe[i, j].imshow(patches[k])
      k += 1 
  
  fig.set_size_inches(18.5, 6.5, forward=True)
  plt.show()
  return 

if __name__ == "__main__":

  window_size = (400//4, 400//4)
  image = url_to_image("https://interactive-examples.mdn.mozilla.net/media/cc0-images/grapefruit-slice-332-332.jpg")
  image = tf.image.resize(image, (400, 400), method = "nearest")
  batches = 16

  image = tf.expand_dims(image, axis = 0)
  image = tf.tile(image, [batches, 1, 1, 1])
  image, _,_,_,_,_ = pad(image, window_size)
  image, Ph, Pw = window_partition_overlaps_tlbr(image, window_size)
  draw_patches(image, batches, 0)

  # import time
  # a = time.time()
  # for i in range(100):
  #   image_, _,_,_,_,_ = pad(image, window_size)
  #   image_, Ph, Pw = window_partition_overlaps_br(image_, window_size)
  #   del image_
  # b = time.time()
  # print(b - a)

  # a = time.time()
  # for i in range(100):
  #   image_, _,_,_,_,_ = pad(image, window_size)
  #   image_, Ph, Pw = window_partition_overlaps_tlbr(image_, window_size)
  # b = time.time()
  # print(b - a)

  # a = time.time()
  # for i in range(100):
  #   image_, _,_,_,_,_ = pad(image, window_size)
  #   image_, Ph, Pw = window_partition(image_, window_size)
  # b = time.time()
  # print(b - a)