from math import sqrt
from collections import Counter
import numpy as np
import pandas as pd
import re

from soynlp.hangle import jamo_levenshtein, compose, decompose
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class Recommend():
    def __init__(self):
        self.score_data = pd.read_excel('./data_in/영화평점데이터.xlsx')
        self.movie_titles = [title for title in self.score_data['movie id']]
        self.critics = {}
        self.tf_, self.vocabs = [], []

        self.get_critics()
        self.make_tf_matrix()

    #피어슨 유사도를 위함-전처리
    def get_critics(self):
        for index in range(len(self.score_data)):
            for i in range(1, 20):
                if self.score_data[i][index]>0:
                    key = self.score_data['movie id'][index]
                    value = self.score_data[i][index]
                        
                    if i not in self.critics:
                        self.critics[i] = {key:value}
                    else:
                        self.critics[i][key] = value

    #문서 유사도(TF-IDF)를 위함-전처리
    def get_movie_data(self):
        # Term Frequency - Inverse Document Frequency를 만들기 위한 작업
        # vocabs : 전체 문서에 등장한 단어들과 각 단어의 등장 횟수
        # tf_    : 각 문서에 등장한 단어와 각 단어의 문서 내의 등장 횟수
        f = open('./data_in/영화데이터_tokenized.txt', 'r', encoding='utf-8')
        movie_data = f.readlines()

        vocabs = []
        for line in movie_data:
            line = line.replace('\n', '')
            if line:
                tmp = []
                for token in line.split():
                    if token.split('/')[1] in ['NNG', 'NNP']:
                        tmp.append(token)
                        vocabs.append(token)
                self.tf_.append(tmp)

        vocab_lists = Counter(vocabs)
        for k, v in vocab_lists.items():
            if v<3:
                continue
            else:
                self.vocabs.append(k)

    #문서 유사도(TF-IDF)를 위함-전처리
    def make_tf_matrix(self):
        self.get_movie_data()
        self.tf_matrix = np.zeros((len(self.tf_), len(self.vocabs)))
        for i, line in enumerate(self.tf_):
            for t in line:
                try:
                    j = self.vocabs.index(t)
                    self.tf_matrix[i][j]+=1
                except:
                    continue

    #편집 거리 알고리즘을 적용하여, 입력된 영화 제목을 목록에서 찾는 함수
    def title_search(self, input_):
        hangle= re.compile('[^ㄱ-ㅣ가-힣]+')
        results = []
        hangle_input = hangle.sub('', input_)

        #초/중/종성으로 분리
        #soynlp.hangle.decompose() 사용
        decomposed = []
        for eomjeol in hangle_input:
            seperated = decompose(eomjeol)
            
            # 초/중성이 없는 음절을 삭제
            # ex) input_('시빌ㄹ워') -> [(ㅅ,ㅣ,), (ㅂ,ㅣ,ㄹ), (ㄹ, , ), (ㅇ,ㅝ, )]
            #     (ㄹ, , )은 중성이 없으므로 오타로 추정하여 삭제
            if seperated[1]==' ' or seperated[0]==' ':
                continue

            decomposed.append(seperated)

        #오타로 추정되는 음절을 제거한 후, 다시 합함
        #soynlp.hangle.compose() 사용
        composed = ''
        for cho, jung, jong in decomposed:
            composed+=compose(cho, jung, jong)

        #오타 교정 후, 목록에서 비슷한 제목을 검색
        #데이터셋의 제목과 입력의 편집 거리를 구한 후, 일정값보다 작은 경우 리스트에 추가
        #입력이 제목의 일부일 경우, 편집 거리와 관계 없이 결과에 추가
        #soynlp.hangle.jamo_levenshtein() 사용
        for key in self.movie_titles:
            title = hangle.sub('', key)
            if composed in title:
                results.append(key)
                continue

            #distance : 입력과 목록의 제목과의 편집 거리
            distance = jamo_levenshtein(composed, title)
            if distance<(len(composed)/2) and distance<2:
                results.append(key)
        
        return list(Counter(results).keys())

    # user1과 user2의 피어슨 상관계수 계산
    def pearson_sim(self, user1, user2):
        df = pd.DataFrame(self.critics)
        similarity = df[user1].corr(df[user2], method='pearson')
        return similarity

    # name  : 현재 사용자 이름/아이디
    # index : 유사도가 높은 사용자들 중, 몇 명의 데이터를 사용하여 추천을 할건가
    def top_match(self, name, index=2):
        matches = []
        for user in self.critics.keys():
            if name!=user:
                # 자신이 아닐 때, 다른 사용자와의 피어슨 상관계수 계산
                # (피어슨 상관계수, 다른 사용자 이름)
                matches.append((self.pearson_sim(name, user), user))

        matches.sort(reverse=True)
        return matches[:index]

    #피어슨 상관계수를 이용한 추천
    def pearson_recommendation(self, name):
        matches = self.top_match(name, len(self.critics)) #전체 사용자 다 활용
        score_dict = {}
        result = []

        for sim_score, user in matches:
            if sim_score<0.5:
                continue
            
            critic = self.critics[user]
            for title in critic:
                if title not in self.critics[name]: #현재 사용자가 보지 않은 영화
                    value = []
                    if title in score_dict:
                        value = score_dict[title]
                    value.append((sim_score, critic[title])) #상관계수에 따른 가중합을 구하기 위함
                    score_dict[title] = value

        for title, values in score_dict.items():
            total_score, total_sim = 0, 0
            for value in values:
                total_score += value[0]*value[1]    #유사도 * 점수의 합
                total_sim += value[0]               #유사도의 합
            total_score = total_score/total_sim
            if total_score>8.5:
                result.append((total_score, title))
        
        result.sort(reverse=True)
        return result

    #TF-IDF와 cosine 유사도를 이용한 영화 추천
    def cosine_recommendation(self, name):
        self.tfidf = TfidfVectorizer
        self.cosine_sim = linear_kernel(self.tf_matrix, self.tf_matrix)

        # 현재 사용자가 좋아하는 영화(8.5점 이상) 추출
        preference = []
        for title, score in self.critics[name].items():
            if score>8.5:
                preference.append(title)

        results = []
        for title in preference:
            # idx : 영화 제목에 대한 index
            idx = self.movie_titles.index(title)
            sim_scores = list(enumerate(self.cosine_sim[idx]))                  #현재 영화문서에 대한 cosine 유사도(영화 index, 유사도)
            sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)    #유사도 기준 내림차순
            sim_scores = sim_scores[:10]                                        #유사도가 높은 10개의 영화

            movie_idxs = [i[0] for i in sim_scores]
            titles = [self.movie_titles[idx] for idx in movie_idxs]

            for title, score in zip(titles, sim_scores):
                # 유사도를 계산했을 때, 0번은 무조건 현재 기준이 되는 영화
                # 현재 기준이 되는 영화를 기준으로 너무 유사도가 떨어지면 추천 중단
                if score[1]<sim_scores[0][1]/3:
                    break
                if title not in self.critics[name]:
                    results.append(title)

        return list(Counter(results).keys())