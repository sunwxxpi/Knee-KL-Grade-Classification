import glob
import pandas as pd

a = glob.glob('./KneeXray/HH_1/0/'+'*.jpg') # glob.glob >> List 형식으로 반환
a_label = ['0' for x in range(0, len(a), 1)]
b = glob.glob('./KneeXray/HH_1/1/'+'*.jpg')
b_label = ['1' for x in range(0, len(b), 1)]
c = glob.glob('./KneeXray/HH_1/2/'+'*.jpg')
c_label = ['2' for x in range(0, len(c), 1)]
d = glob.glob('./KneeXray/HH_1/3/'+'*.jpg')
d_label = ['3' for x in range(0, len(d), 1)]
e = glob.glob('./KneeXray/HH_1/4/'+'*.jpg')
e_label = ['4' for x in range(0, len(e), 1)]

file = a + b + c + d + e
label = a_label + b_label + c_label + d_label + e_label

a = pd.DataFrame({'data':file, 'label':label})
a.set_index('data', inplace=True) # 'data' 열을 Index로 설정 EXCEL A열로 붙임
a.to_csv('./KneeXray/HH_1.csv')

""" a = glob.glob('./KneeXray/test/0/'+'*.png')
a_label = ['0' for x in range(0, len(a), 1)]
b = glob.glob('./KneeXray/test/1/'+'*.png')
b_label = ['1' for x in range(0, len(b), 1)]
c = glob.glob('./KneeXray/test/2/'+'*.png')
c_label = ['2' for x in range(0, len(c), 1)]
d = glob.glob('./KneeXray/test/3/'+'*.png')
d_label = ['3' for x in range(0, len(d), 1)]
e = glob.glob('./KneeXray/test/4/'+'*.png')
e_label = ['4' for x in range(0, len(e), 1)]

file = a + b + c + d + e
label = a_label + b_label + c_label + d_label + e_label

a = pd.DataFrame({'data':file, 'label':label})
a. set_index('data', inplace=True)
a.to_csv('./KneeXray/Test_correct.csv') """