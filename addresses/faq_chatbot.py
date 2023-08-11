import os
import warnings
from gensim.models import doc2vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
#형태소 분석
import jpype
from konlpy.tag import Kkma
import multiprocessing


faqs = pd.read_csv('ChatbotData.csv',encoding ='utf-8')
print(faqs['Q'])
kkma = Kkma()
filter_kkma = ['NNG',  #보통명사
             'NNP',  #고유명사
             'OL' ,  #외국어
            ]

def tokenize_kkma(doc):
    jpype.attachThreadToJVM()
    token_doc = ['/'.join(word) for word in kkma.pos(doc)]
    return token_doc

def tokenize_kkma_noun(doc):
    jpype.attachThreadToJVM()
    token_doc = ['/'.join(word) for word in kkma.pos(doc) if word[1] in filter_kkma]
    return token_doc


cores = multiprocessing.cpu_count()

# 리스트에서 각 문장부분 토큰화
token_faqs = []
for i in range(len(faqs)):
    token_faqs.append([tokenize_kkma_noun(faqs['Q'][i]), i ])

# Doc2Vec에서 사용하는 태그문서형으로 변경
tagged_faqs = [TaggedDocument(d, [int(c)]) for d, c in token_faqs]

# 파일로부터 모델을 읽는다. 없으면 생성한다.
d2v_faqs = doc2vec.Doc2Vec(
                                vector_size=100,
                                alpha=0.025,
                                min_alpha=0.025,
                                hs=1,
                                negative=0,
                                dm=0,
                                window=3,
                                dbow_words = 1,
                                min_count = 1,
                                workers = cores,
                                seed=0,
                                epochs=100
                                    )
d2v_faqs.build_vocab(tagged_faqs)

# train document vectors
for epoch in range(50):
    d2v_faqs.train(tagged_faqs,
                               total_examples = d2v_faqs.corpus_count,
                               epochs = d2v_faqs.epochs
                               )
    d2v_faqs.alpha -= 0.0025 # decrease the learning rate
    d2v_faqs.min_alpha = d2v_faqs.alpha # fix the learning rate, no decay



# FAQ 답변
def faq_answer(input):
    # 테스트하는 문장도 같은 전처리를 해준다.
    tokened_test_string = tokenize_kkma_noun(input)
    print('인풋!' + str(tokened_test_string))
    print('hi')
    topn = 10
    test_vector = d2v_faqs.infer_vector(tokened_test_string)
    result = d2v_faqs.docvecs.most_similar([test_vector], topn=topn)
    answer_list = []

    for i in range(topn):
         print("{}위. {}, {} {} {}".format(i + 1, result[i][1], result[i][0], faqs['Q'][result[i][0]], faqs['A'][result[i][0]]))
         answer_list.append(dict(acc=result[i][1], question=faqs['Q'][result[i][0]], answer=faqs['A'][result[i][0]]))

    # 성능 측정
    raten = 1
    found = 0
    for i in range(len(faqs)):
        tstr = faqs['A'][i]
        ttok = tokenize_kkma_noun(tstr)
        tvec = d2v_faqs.infer_vector(ttok)
        re = d2v_faqs.docvecs.most_similar([tvec], topn=raten)
        for j in range(raten):
            if i == re[j][0]: found = found + 1

    print("정확도 = {} % ({}/{} )  ".format(found / len(faqs), found, len(faqs)))


    print(dict(acc1=result[0][1], question1=faqs['Q'][result[0][0]], answer1=faqs['A'][result[0][0]],
         acc2=result[1][1], question2=faqs['Q'][result[1][0]], answer2=faqs['A'][result[1][0]],
         acc3=result[2][1], question3=faqs['Q'][result[2][0]], answer3=faqs['A'][result[2][0]],
         acc4=result[3][1], question4=faqs['Q'][result[3][0]], answer4=faqs['A'][result[3][0]],
         acc5=result[4][1], question5=faqs['Q'][result[4][0]], answer5=faqs['A'][result[4][0]],
         acc6=result[5][1], question6=faqs['Q'][result[5][0]], answer6=faqs['A'][result[5][0]],
         acc7=result[6][1], question7=faqs['Q'][result[6][0]], answer7=faqs['A'][result[6][0]],
         acc8=result[7][1], question8=faqs['Q'][result[7][0]], answer8=faqs['A'][result[7][0]],
         acc9=result[8][1], question9=faqs['Q'][result[8][0]], answer9=faqs['A'][result[8][0]],
         acc10=result[9][1], question10=faqs['Q'][result[9][0]], answer10=faqs['A'][result[9][0]]))

    return dict(acc1=result[0][1], question1=faqs['Q'][result[0][0]], answer1=faqs['A'][result[0][0]],
         acc2=result[1][1], question2=faqs['Q'][result[1][0]], answer2=faqs['A'][result[1][0]],
         acc3=result[2][1], question3=faqs['Q'][result[2][0]], answer3=faqs['A'][result[2][0]],
         acc4=result[3][1], question4=faqs['Q'][result[3][0]], answer4=faqs['A'][result[3][0]],
         acc5=result[4][1], question5=faqs['Q'][result[4][0]], answer5=faqs['A'][result[4][0]],
         acc6=result[5][1], question6=faqs['Q'][result[5][0]], answer6=faqs['A'][result[5][0]],
         acc7=result[6][1], question7=faqs['Q'][result[6][0]], answer7=faqs['A'][result[6][0]],
         acc8=result[7][1], question8=faqs['Q'][result[7][0]], answer8=faqs['A'][result[7][0]],
         acc9=result[8][1], question9=faqs['Q'][result[8][0]], answer9=faqs['A'][result[8][0]],
         acc10=result[9][1], question10=faqs['Q'][result[9][0]], answer10=faqs['A'][result[9][0]])


def faq_search(inputs):
    keywords = None
    for word in inputs:
        if keywords is None:
            keywords = word
        else:
            keywords = keywords + '|' + word
    faqs[faqs.str.contains(keywords)]
    print(faqs)
    return 0



d2v_faqs.save('./a.model')
#d2v_faqs_2 = doc2vec.Doc2Vec.load(os.path.join('data','/content/drive/My Drive/data/d2v_faqs_size50_min1_batch50_epoch50_nounonly_dm0.mode