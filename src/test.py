# -*- coding:utf-8 -*-
#
#        Author : TangHanYi
#        E-mail : thydeyx@163.com
#   Create Date : 2018-06-08 16时19分57秒
# Last modified : 2018-06-08 16时31分09秒
#     File Name : test.py
#          Desc :

import jieba

class Solution:
    def __init__(self):
        self.testText = "习近平主席今日发表演讲。"
        jieba.load_userdict("../data/word_dict.list")

    def run(self, text):
        self.testText = text
        seg = jieba.cut(self.testText)
        print('/ '.join(seg))

if __name__ == "__main__":
    s = Solution()
    while True:
        text = raw_input('sentence: ')
        s.run(text)
