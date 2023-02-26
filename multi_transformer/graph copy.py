"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import matplotlib.pyplot as plt
import re


def read(name):
    f = open(name, 'r')
    file = f.read()
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]


def draw(mode):
    if mode == 'loss':
        original_train = read('result/original_train_loss.txt')
        original_test = read('result/original_test_loss.txt')

        cat_train = read('result/cat1_train_loss.txt')
        cat_test = read('result/cat1_test_loss.txt')

        plt.plot(original_train, 'r', label='original_train')
        plt.plot(original_test, 'b', label='original_validation')

        plt.plot(cat_train, 'g', label='cat_train')
        plt.plot(cat_test, 'orange', label='cat_validation')

        plt.legend(loc='upper right')


    elif mode == 'bleu':
        original_bleu = read('result/original_bleu.txt')
        plt.plot(original_bleu, 'b', label='original_bleu score')

        cat_bleu = read('result/cat1_bleu.txt')
        plt.plot(cat_bleu, 'r', label='cat_bleu score')

        plt.legend(loc='lower right')

    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.title('training result')
    plt.grid(True, which='both', axis='both')
    plt.savefig('saved/transformer-base/%s' %mode)
    plt.show()


if __name__ == '__main__':
    draw(mode='loss')
    draw(mode='bleu')
