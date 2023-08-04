import torch
from ltp import LTP

ltp = LTP("LTP/small")  # 默认加载 Small 模型

# 将模型移动到 GPU 上
if torch.cuda.is_available():
    # ltp.cuda()
    ltp.to("cuda")

output = ltp.pipeline("厚度在0到5毫米的产品", tasks=["cws", "pos", "ner", "srl", "dep", "sdp"])
# 使用字典格式作为返回结果
print(output.cws) # print(output[0]) / print(output['cws']) # 也可以使用下标访问
print(output.pos)
dep = list(zip(output.dep['head'], output.dep['label']))
print(dep, '1')
data = {'words':output.cws, 'tags':output.pos, 'arcs':dep}
print(data)


# # 使用感知机算法实现的分词、词性和命名实体识别，速度比较快，但是精度略低
# ltp = LTP("LTP/legacy")
# # cws, pos, ner = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "ner"]).to_tuple() # error: NER 需要 词性标注任务的结果
# cws, pos, ner = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "pos", "ner"]).to_tuple() # to tuple 可以自动转换为元组格式
# # 使用元组格式作为返回结果
# print(cws, pos, ner)