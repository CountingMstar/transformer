
k_list = [1, 4, 7]

for k in k_list:
    # def good(k):
    #     print('yse')
    #     print(k)

    # if __name__ == '__main__':
    #     good(k)

    print('this %s-%s' %(k, k+1))
    x = str(k)
    f = open('result/sex'+ x +'.txt', 'w')
    f.write(str('bleus'))
    f.close()


