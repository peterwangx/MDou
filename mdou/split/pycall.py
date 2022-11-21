#coding=utf-8

import ctypes
from ctypes import *

isWindows = True
import platform
sysstr = platform.system()
if (sysstr == "Windows"):
    isWindows =  True
elif (sysstr == "Linux"):
    isWindows = False
else:
    print("Other System ")
    isWindows = False

if isWindows:
    lib = ctypes.WinDLL("./SplitDou.dll")
else:
    lib = ctypes.cdll.LoadLibrary("./libSplit.so")

##call
INPUT = c_int * 15

input = INPUT()

input[0] = 1
input[1] = 1
input[2] = 1
input[3] = 1
input[4] = 1
input[5] = 0
input[6] = 0
input[7] = 1
input[8] = 0
input[9] = 1
input[10] = 0
input[11] = 0
input[12] = 3
input[13] = 1
input[14] = 1
fun=lib.getMinHands
fun.restype = c_int
print (fun(12, input))


