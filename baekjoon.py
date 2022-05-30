import sys

'''
a, b = [int(sys.stdin.readline()) for i in range(2)]
bOne = b%10
bTen = (b//10)%10
bHundred = b//100
sum = a*bHundred*100 + a*bTen*10 + a*bOne
print(a*bOne, a*bTen, a*bHundred, sum, sep='\n')
'''

a = int(input())
b = input()

for i in b[::-1]:
    print(a*int(i))

print(a*int(b))