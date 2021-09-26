f1 = open("yolov4_raw_output.txt", "r").readlines()
f2 = open("darknet_compare.txt", "r").readlines()

diff = []
for i in range(len(f1)):
  val1 = float(f1[i].replace("\n", ""))
  val2 = float(f2[i].replace("\n", ""))
  diff.append((abs(100 * (val1 - val2) / val2), i, val1, val2))

diff.sort(key=lambda x: x[0], reverse=True)
print(diff[:20])
