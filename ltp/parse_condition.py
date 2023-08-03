# Tag  Description                                 Example
# a   adjective：形容词                           美丽  
# b   other noun-modifier：其他的修饰名词           大型, 西式  
# c   conjunction：连词                           和, 虽然   
# d   adverb：副词                                很   
# e   exclamation：感叹词                          哎   
# g   morpheme：词素 
# h   prefix：前缀                                阿, 伪    
# i   idiom：成语                                 百花齐放    
# j   abbreviation：缩写                          公检法 
# k   suffix：后缀                                界, 率    
# m   number：数字                                一, 第一   
# n   general noun：一般名词                       苹果  
# nd  direction noun：方向名词                     右侧      
# nh  person name：人名                           杜甫, 汤姆  
# ni  organization name：公司名                    保险公司，中国银行
# nl  location noun：地点名词                      城郊
# ns  geographical name：地理名词                  北京
# nt  temporal noun：时间名词                      近日, 明代
# nz  other proper noun：其他名词                  诺贝尔奖
# o   onomatopoeia：拟声词                         哗啦
# p   preposition：介词                           在, 把，与
# q   quantity：量词                              个
# r   pronoun：代词                               我们
# u   auxiliary：助词                             的, 地
# v   verb：动词                                  跑, 学习
# wp  punctuation：标点                           ，。！
# ws  foreign words：国外词                       CPU
# x   non-lexeme：不构成词                        萄, 翱
# z  descriptive words 描写，叙述的词             瑟瑟，匆匆



# # 分句
# from pyltp import SentenceSplitter
# sents = SentenceSplitter.split('元芳你怎么看？我就趴窗口上看呗！')  # 分句
# print('\n'.join(sents))

# # 分词
# import os
# from pyltp import Segmentor
# LTP_DATA_DIR='D:/ltp_data_v3.4.0/ltp_data_v3.4.0'
# cws_model_path=os.path.join(LTP_DATA_DIR,'cws.model')
# segmentor=Segmentor(cws_model_path)
# # words=segmentor.segment('熊高雄你吃饭了吗')
# words=segmentor.segment('壁厚在1到5毫米的成功案例') #壁厚小于5毫米的成功案例
# print('\t'.join(words))
# segmentor.release()


# # 词性标注
# from pyltp import Postagger
# pdir=os.path.join(LTP_DATA_DIR,'pos.model')
# pos = Postagger(pdir)   
# postags = pos.postag(words)
# print(u"词性:", postags)
# pos.release()                                           
# data = {"words": words, "tags": postags}
# print(data)


# #命名实体识别
# from pyltp import NamedEntityRecognizer
# nermodel=os.path.join(LTP_DATA_DIR, 'ner.model')
# reg = NamedEntityRecognizer(nermodel)
# netags = reg.recognize(words, postags)
# print(u"命名实体识别:", netags)
# data={"reg": netags,"words":words,"tags":postags}
# print(data)
# reg.release()



# #依存句法分析
# from pyltp import Parser
# parmodel = os.path.join(LTP_DATA_DIR, 'parser.model')
# parser = Parser(parmodel)                                          #初始化命名实体实例
# # parser.load(parmodel)                                  #加载模型
# arcs = parser.parse(words, postags)              #句法分析


# print("\t".join("%d:%s" % (arc[0], arc[1]) for arc in arcs))

# rely_id = [arc[0] for arc in arcs]              # 提取依存父节点id
# relation = [arc[1] for arc in arcs]         # 提取依存关系
# heads = ['Root' if id == 0 else words[id-1] for id in rely_id]  # 匹配依存父节点词语
# for i in range(len(words)):
#     print(relation[i] + '(' + words[i] + ', ' + heads[i] + ')')

# parser.release()

# encoding: utf-8
# _*_ coding:utf-8 _*_
import os
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser

def parse_sentence(sentence):
    # 分词

    LTP_DATA_DIR='D:/ltp_data_v3.4.0/ltp_data_v3.4.0'
    cws_model_path=os.path.join(LTP_DATA_DIR,'cws.model')
    segmentor=Segmentor(cws_model_path)
    words=segmentor.segment(sentence)
    # print('\t'.join(words))
    segmentor.release()


    # 词性标注

    pdir=os.path.join(LTP_DATA_DIR,'pos.model')
    pos = Postagger(pdir)   
    postags = pos.postag(words)
    # print(u"词性:", postags)
    pos.release()                                           
    data = {"words": words, "tags": postags}
    # print(data)
    

    #依存句法分析

    parmodel = os.path.join(LTP_DATA_DIR, 'parser.model')
    parser = Parser(parmodel)                         # 初始化命名实体实例并加载模型
    arcs = parser.parse(words, postags)               # 句法分析
    # print("\t".join("%d:%s" % (arc[0], arc[1]) for arc in arcs))
    # rely_id = [arc[0] for arc in arcs]              # 提取依存父节点id
    # relation = [arc[1] for arc in arcs]             # 提取依存关系
    # heads = ['Root' if id == 0 else words[id-1] for id in rely_id]  # 匹配依存父节点词语
    # for i in range(len(words)):
    #     print(relation[i] + '(' + words[i] + ', ' + heads[i] + ')')
    parser.release()
    data['arcs'] = arcs

    return data

key_words = {'<':['小于'],
             '=':['等于'],
             '>':['大于'],
             'range':['到']}

def parse_condition(sentence, data):
    option = []
    subject = []
    value = []
    quantifier = []

    flag = False
    for key in key_words:
        for word in key_words[key]:
            if word in sentence:
                option.append(key)
                flag = True
            break 
        if flag==True:
            break

    for i, tag in enumerate(data['tags']):
        if tag == 'm':
            value.append(data['words'][i])
        if tag == 'q':
            quantifier.append(data['words'][i])
        if data['arcs'][i][1] == 'SBV':
            subject.append(data['words'][i])
    condition = {}
    condition['subject'] = subject
    condition['option'] = option
    condition['value'] = value
    condition['quantifier'] = quantifier
    return condition


if __name__=='__main__':
    question = '厚度在0到5毫米的产品' # input('input an question:')
    data = parse_sentence(question)
    # print(data)
    condition = parse_condition(question, data)
    print(condition)