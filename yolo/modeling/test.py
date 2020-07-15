import os,sys


def printer():
    print("hello world")

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
 
print(sys.path)
print(parentdir)

# from yolo import custom
