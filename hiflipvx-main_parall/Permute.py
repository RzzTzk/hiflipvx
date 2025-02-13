

import torch
import torch.nn as nn
import numpy as np
import math


#------------------ Parameters ----------------------
channel, col, row = 4,4,4
elements = channel*col*row

#---------------- row_vec col_vec ch_vec ---------------------
# Only_ch_vec, Ch_col_vec, Ch_col_row = 2 , 2 ,2
ch_vec , col_vec, row_vec = 2, 2, 2 
# para_method = 2
# if (para_method == Ch_col_row):
#     ch_vec = int(channel/2)
#     col_vec = int(col/2)
#     row_vec = int(row/2)
# elif (para_method == Ch_col_vec):
#     ch_vec = int(channel/2)
#     col_vec = int(col/2)
#     row_vec = 1
# elif (para_method == Only_ch_vec):
#     ch_vec = int(channel/2)
#     col_vec = 1
#     row_vec = 1
# else:
#     print("========error selection============")

sw_3 = np.zeros(channel*col*row)
hw_3 = []
ch_times = int(channel/ch_vec)
col_times = int(col/col_vec)
row_times = int(row/row_vec)
times = int(elements/(ch_vec*col_vec*row_vec))
for i in range (times):
    for m in range (row_vec):
        for k in range (col_vec):
            for j in range (ch_vec):
                if(i == 0):
                    start_point = m*channel*col + k*channel
                else:
                    start_point = math.floor(i/(col_times*ch_times))*col*channel*row_vec + m*channel*col + ((math.floor(i/ch_times))%col_times)*channel*col_vec + k*channel +  (i%ch_times)*ch_vec
                sw_3[start_point+j] = start_point + j
                hw_3.append(start_point+j)
#     # print("row",math.floor(i/(col_times*ch_times)))
#     # print("col", ((math.floor(i/ch_times))%col_times))
#     # print("ch",(i%ch_times))
#     # print("startpoint", start_point)
# print(row_vec)
# print(col_vec)
# print(ch_vec)
print(sw_3)
print(hw_3)

# # 
# sw_3 = np.zeros(channel*col*row)
# hw_3 = []
# for i in range (times):
#     for m in range (ch_vec):
#         for k in range (row_vec):
#             for j in range (col_vec):
#                 if(i == 0):
#                     start_point = m*row*col + k*col
#                 else:
#                     start_point = math.floor(i/(col_times*row_times))*col*row*ch_vec + m*row*col + ((math.floor(i/row_times))%row_times)*col*row_vec + k*col +  (i%col_times)*col_vec
#                 sw_3[start_point+j] = start_point + j
#                 hw_3.append(start_point+j)

# print(sw_3)
# print(hw_3)

# sw_3 = np.zeros(channel*col*row)
# hw_3 = []
# for i in range (times):
#     for m in range (row_vec):
#         for k in range (col_vec):
#             for j in range (ch_vec):
#                 if(i == 0):
#                     start_point = m*channel*col + k*channel
#                 else:
#                     start_point = math.floor(i/(col_times*ch_times))*col*channel*row_vec + m*channel*col + ((math.floor(i/col_times))%col_times)*channel*col_vec + k*channel +  (i%ch_times)*ch_vec
#                 sw_3[start_point+j] = start_point + j
#                 hw_3.append(start_point+j)

# print(sw_3)
# print(hw_3)

# ii=0
# hw_4 = np.zeros(channel*col*row)
# for i in range (times):
#     for m in range (ch_vec):
#         for k in range (row_vec):
#             for j in range (col_vec):
#                 if(i == 0):
#                     start_point = m*row*col + k*col
#                 else:
#                     start_point = math.floor(i/(col_times*row_times))*col*row*ch_vec + m*row*col + ((math.floor(i/row_times))%row_times)*col*row_vec + k*col +  (i%col_times)*col_vec
#                 a = hw_3[ii]
#                 ii += 1
#                 hw_4[start_point+j] = a

# print(hw_4)

# #---------------- only ch_vec ---------------------
# ch_vec = int(channel/2)
# col_vec = col
# row_vec = row
# sw = []

# for i in range (int(elements/ch_vec)):
#     for j in range (ch_vec):
#         sw.insert(i*ch_vec+j,i*ch_vec+j)
#         # print(i*ch_vec+j)
# print(sw)

#---------------- col_vec ch_vec ---------------------
# ch_vec = int(channel/2)
# col_vec = int(int(col/2))
# row_vec = row
# sw_2 = []
# hw_2 = []
# ch_times = int(channel/ch_vec)
# times = int(elements/(ch_vec*col_vec))

# for i in range (int(elements/(ch_vec*col_vec))):
#     for k in range (col_vec):
#         for j in range (ch_vec):
#             if (i==0):
#                 start_point = k*channel
#             else:
#                 start_point = (i%ch_times)*ch_vec + math.floor(i/ch_times)*channel*col_vec + k*channel
#             sw_2.insert(start_point+j,start_point+j)
#             hw_2.append(start_point+j)
# print(sw_2)
# print(hw_2)

