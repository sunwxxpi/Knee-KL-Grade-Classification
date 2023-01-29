import glob
import pandas as pd

a = glob.glob('./KneeXray/train_(456, 456)/0/'+'*.png') # glob.glob >> list 형식으로 반환
a_label = ['0' for x in range(0, len(a), 1)]
b = glob.glob('./KneeXray/train_(456, 456)/1/'+'*.png')
b_label = ['1' for x in range(0, len(b), 1)]
c = glob.glob('./KneeXray/train_(456, 456)/2/'+'*.png')
c_label = ['2' for x in range(0, len(c), 1)]
d = glob.glob('./KneeXray/train_(456, 456)/3/'+'*.png')
d_label = ['3' for x in range(0, len(d), 1)]
e = glob.glob('./KneeXray/train_(456, 456)/4/'+'*.png')
e_label = ['4' for x in range(0, len(e), 1)]

file = a + b + c + d + e
label = a_label + b_label + c_label + d_label + e_label

a = pd.DataFrame({'data':file, 'label':label})
a.set_index('data', inplace=True) # 'data' 열을 index로 설정 EXCEL A열로 붙임
a.to_csv('./KneeXray/Train_(456, 456).csv')

a = glob.glob('./KneeXray/test_(456, 456)/0/'+'*.png')
a_label = ['0' for x in range(0, len(a), 1)]
b = glob.glob('./KneeXray/test_(456, 456)/1/'+'*.png')
b_label = ['1' for x in range(0, len(b), 1)]
c = glob.glob('./KneeXray/test_(456, 456)/2/'+'*.png')
c_label = ['2' for x in range(0, len(c), 1)]
d = glob.glob('./KneeXray/test_(456, 456)/3/'+'*.png')
d_label = ['3' for x in range(0, len(d), 1)]
e = glob.glob('./KneeXray/test_(456, 456)/4/'+'*.png')
e_label = ['4' for x in range(0, len(e), 1)]

file = a + b + c + d + e
label = a_label + b_label + c_label + d_label + e_label

a = pd.DataFrame({'data':file, 'label':label})
a. set_index('data', inplace=True)
a.to_csv('./KneeXray/Test_correct_(456, 456).csv')