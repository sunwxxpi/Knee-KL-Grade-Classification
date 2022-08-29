import glob
import pandas as pd

a = glob.glob('./KneeXray/HealthHub Dataset/train/0/'+'*.jpg') # glob.glob >> list 형식으로 반환
a_label = ['0' for x in range(0, len(a), 1)]
b = glob.glob('./KneeXray/HealthHub Dataset/train/1/'+'*.jpg')
b_label = ['1' for x in range(0, len(b), 1)]
c = glob.glob('./KneeXray/HealthHub Dataset/train/2/'+'*.jpg')
c_label = ['2' for x in range(0, len(c), 1)]
d = glob.glob('./KneeXray/HealthHub Dataset/train/3/'+'*.jpg')
d_label = ['3' for x in range(0, len(d), 1)]
e = glob.glob('./KneeXray/HealthHub Dataset/train/4/'+'*.jpg')
e_label = ['4' for x in range(0, len(e), 1)]

file = a + b + c + d + e
label = a_label + b_label + c_label + d_label + e_label

a = pd.DataFrame({'data':file, 'label':label})
a.set_index('data', inplace=True) # 'data' 열을 index로 설정 EXCEL A열로 붙임
a.to_csv('./KneeXray/HealthHub Dataset/Train_HealthHub.csv')

a = glob.glob('./KneeXray/HealthHub Dataset/test/0/'+'*.jpg')
a_label = ['0' for x in range(0, len(a), 1)]
b = glob.glob('./KneeXray/HealthHub Dataset/test/1/'+'*.jpg')
b_label = ['1' for x in range(0, len(b), 1)]
c = glob.glob('./KneeXray/HealthHub Dataset/test/2/'+'*.jpg')
c_label = ['2' for x in range(0, len(c), 1)]
d = glob.glob('./KneeXray/HealthHub Dataset/test/3/'+'*.jpg')
d_label = ['3' for x in range(0, len(d), 1)]
e = glob.glob('./KneeXray/HealthHub Dataset/test/4/'+'*.jpg')
e_label = ['4' for x in range(0, len(e), 1)]

file = a + b + c + d + e
label = a_label + b_label + c_label + d_label + e_label

a = pd.DataFrame({'data':file, 'label':label})
a. set_index('data', inplace=True)
a.to_csv('./KneeXray/HealthHub Dataset/Test_correct_HealthHub.csv')

""" 
63 : 42 / 21
198 : 132 / 66
138 : 92 / 46
83 : 55 / 28
42 : 28 / 14
"""