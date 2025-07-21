# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility library of instructions."""

import os
import re

import immutabledict
from packaging.version import parse as parse_version


RANK = os.environ.get("LOCAL_RANK", "0")

# ISO 639-1 codes to language names.
LANGUAGE_CODES = immutabledict.immutabledict(
    {
        "en": "English",
        "es": "Spanish",
        "pt": "Portuguese",
        "ar": "Arabic",
        "hi": "Hindi",
        "fr": "French",
        "ru": "Russian",
        "de": "German",
        "ja": "Japanese",
        "it": "Italian",
        "bn": "Bengali",
        "uk": "Ukrainian",
        "th": "Thai",
        "ur": "Urdu",
        "ta": "Tamil",
        "te": "Telugu",
        "bg": "Bulgarian",
        "ko": "Korean",
        "pl": "Polish",
        "he": "Hebrew",
        "fa": "Persian",
        "vi": "Vietnamese",
        "ne": "Nepali",
        "sw": "Swahili",
        "kn": "Kannada",
        "mr": "Marathi",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "ml": "Malayalam",
        "fi": "Finnish",
        "zh": "Chinese",
    }
)

_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
# _STARTERS = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
# _ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"

_MIXED_ALPHABETS = "([A-Za-z가-힣])" # 한글과 영어 모두 포함
_KOREAN_LIST = "([가나다라마바사])"  # 한글 리스트 마커

def split_into_sentences(text):
    """Split the text into sentences. (답변을 문장 단위로 분리합니다.)
    기존 함수를 이용합니다. 한국어 문장 생성에서도 중간에 약어 등은 영어로 표기될 수 있습니다.
    Args:
      text: A string that consists of more than or equal to one sentences.
    Returns:
      A list of strings where each string is a sentence.
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(
        _MULTIPLE_DOTS,
        lambda match: "<prd>" * len(match.group(0)) + "<stop>",
        text,
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")

    # text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(_MIXED_ALPHABETS + "[.]" + _MIXED_ALPHABETS + "[.]" + _MIXED_ALPHABETS + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text) # 영어/한국어 약어 처리
    text = re.sub(_MIXED_ALPHABETS + "[.]" + _MIXED_ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text) # 영어/한국어 약어 처리
    
    # 기존 영어 약어 처리
    # text = re.sub(
    #     _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
    #     "\\1<prd>\\2<prd>\\3<prd>",
    #     text,
    # )
    # text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text) 
    # text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text) # _STARTERS는 사용하지 않음

    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    text = re.sub(r"\s" + _ALPHABETS + "[.]\s+(?=[가-힣])", " \\1<prd> ", text) # 영어 약어 + 직후 한글이 적힐 시 온점 아님 처리
    text = re.sub(r"\s" + _KOREAN_LIST + "[.]\s+", " \\1<prd> ", text) # 한글로 된 리스트 마커 처리

    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_syllables(text):
    """Counts the number of syllables for Korean text.
    입력된 한국어 텍스트의 모든 음절 갯수를 카운트하고 리턴합니다."""

    text = text.strip()
    if not text:
        return 0 
    
    # 음절 분리 (+ 대신 없이 사용하여 개별 문자 매칭)
    syllables = re.findall(r'[가-힣]', text)
    return len(syllables)


def count_words(text):
    """Counts the number of words for Korean text.
    띄어쓰기를 기준으로 한국어 문장의 단어를 분리합니다."""
    # 기존 코드
    # tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+") 
    # tokens = tokenizer.tokenize(text)
    # num_words = len(tokens) 

    text = text.strip()
    text = ' '.join(text.split())
    if not text:
        return 0 
    
    return len(text.split())


def count_sentences(text):
    """Count the number of sentences."""
    # tokenizer = _get_sentence_tokenizer()
    # tokenized_sentences = tokenizer.tokenize(text)
    tokenized_sentences = split_into_sentences(text)
    return len(tokenized_sentences)


# 제거된 원본 IFEval 함수
# def generate_keywords(num_keywords):
#     """Randomly generates a few keywords."""
#     return random.sample(WORD_LIST, k=num_keywords)

# @functools.lru_cache(maxsize=None)
# def _get_sentence_tokenizer():
#     return nltk.data.load("nltk:tokenizers/punkt/english.pickle")


class KoreanFormalityPatterns:
    """한국어 경어체/반말체 패턴을 관리하는 클래스"""
    
    def __init__(self):
        # 경어체 어말어미 패턴들
        self.formal_endings = {
            # 평서문 경어체
            'declarative': [
                r'습니다$', r'ㅂ니다$', r'입니다$',
                r'합니다$', r'됩니다$', r'있습니다$', r'없습니다$',
                r'갑니다$', r'옵니다$', r'같습니다$', r'겠습니다$',
                r'드립니다$', r'받습니다$', r'하겠습니다$',
                r'이겠습니다$', r'아니습니다$'
            ],
            
            # 의문문 경어체
            'interrogative': [
                r'습니까\?$', r'ㅂ니까\?$', r'입니까\?$',
                r'합니까\?$', r'있습니까\?$', r'없습니까\?$',
                r'갑니까\?$', r'옵니까\?$', r'같습니까\?$',
                r'겠습니까\?$', r'드릴까요\?$', r'드립니까\?$',
                r'하시겠습니까\?$', r'이십니까\?$', r'계십니까\?$'
            ],
            
            # 명령문 경어체
            'imperative': [
                r'하세요$', r'하십시오$', r'해주세요$', r'해주십시오$',
                r'가세요$', r'가십시오$', r'오세요$', r'오십시오$',
                r'드세요$', r'드십시오$', r'받으세요$', r'받으십시오$',
                r'보세요$', r'보십시오$', r'말씀해주세요$', r'말씀하세요$',
                r'주무세요$', r'주무십시오$'
            ],
            
            # 청유문 경어체
            'propositive': [
                r'합시다$', r'갑시다$', r'가겠습니다$',
                r'하겠습니다$', r'드리겠습니다$', r'해보겠습니다$'
            ],
            
            # 감탄문 경어체
            'exclamatory': [
                r'군요!$', r'네요!$', r'습니다!$', r'ㅂ니다!$'
            ]
        }
        
        # 반말체 어말어미 패턴들
        self.informal_endings = {
            # 평서문 반말체
            'declarative': [
                r'한다$', r'한다.$', r'다$', r'다.$', r'야$', r'이야$',
                r'어$', r'아$', r'지$', r'네$', r'거야$', r'이거든$',
                r'이지$', r'거든$', r'거지$', r'잖아$', r'해$', r'가$',
                r'와$', r'와.$', r'간다$', r'온다$', r'있어$', r'없어$'
            ],
            
            # 의문문 반말체
            'interrogative': [
                r'[니냐]\?$', r'어\?$', r'아\?$', r'지\?$', r'야\?$',
                r'거야\?$', r'거니\?$', r'나\?$', r'까\?$', r'을까\?$',
                r'갈까\?$', r'할까\?$', r'뭐야\?$', r'뭐\?$'
            ],
            
            # 명령문 반말체
            'imperative': [
                r'해$', r'가$', r'와$', r'봐$', r'줘$', r'도$', r'말아$',
                r'하지마$', r'가지마$', r'오지마$', r'먹어$', r'자$'
            ],
            
            # 청유문 반말체
            'propositive': [
                r'하자$', r'가자$', r'자자$', r'해보자$'
            ]
        }
        
        # 높임 표현 (존댓말 어휘)
        self.honorific_words = [
            r'께서', r'이십니다', r'하십니다', r'하시는', r'하신', r'하실',
            r'계십니다', r'계시는', r'계신', r'계실', r'주십시오', r'주세요',
            r'드립니다', r'드리는', r'드린', r'드릴', r'말씀', r'여쭤',
            r'여쭙', r'모십니다', r'모시는', r'모신', r'모실', r'진지',
            r'댁', r'따님', r'아드님', r'영감', r'할머님', r'할아버님'
        ]
    
    def get_all_formal_patterns(self):
        """모든 경어체 패턴을 하나의 리스트로 반환"""
        patterns = []
        for category in self.formal_endings.values():
            patterns.extend(category)
        return patterns
    
    def get_all_informal_patterns(self):
        """모든 반말체 패턴을 하나의 리스트로 반환"""
        patterns = []
        for category in self.informal_endings.values():
            patterns.extend(category)
        return patterns
    
    def add_formal_pattern(self, category, pattern):
        """새로운 경어체 패턴 추가"""
        if category not in self.formal_endings:
            self.formal_endings[category] = []
        self.formal_endings[category].append(pattern)
    
    def add_informal_pattern(self, category, pattern):
        """새로운 반말체 패턴 추가"""
        if category not in self.informal_endings:
            self.informal_endings[category] = []
        self.informal_endings[category].append(pattern)

def preprocess_korean_text(text):
    """텍스트 전처리"""
    # 공백 제거
    text = text.strip()
    
    # 연속된 공백을 하나로 변경
    text = re.sub(r'\s+', ' ', text)
    
    # 특수문자가 많은 경우 제거 (이모티콘 등)
    # 하지만 문장부호는 유지
    text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.!?~]', '', text)
    
    return text

# def split_sentences(text):
#     """텍스트를 문장 단위로 분할"""
#     # 문장 끝 표시자로 분할
#     sentences = re.split(r'[.!?]+\s*', text)
    
#     # 빈 문장 제거
#     sentences = [s.strip() for s in sentences if s.strip()]
    
#     return sentences


class RephrasePatterns:
    """고객 요청 확인 메시지 패턴을 관리하는 클래스"""
    
    def __init__(self):
        # 고객 요청 확인을 나타내는 다양한 표현 패턴들
        self.confirmation_patterns = {
            # 문의/질문 확인 패턴
            'inquiry': [
                r'문의[를하]?\s*(?:주셨|하셨)',
                r'질문[을를]?\s*(?:주셨|하셨)',
                r'궁금[하해]\s*(?:하신|하시는)',
                r'알고\s*싶[어으]\s*(?:하신|하시는)',
                r'[을를가]\s*(?:문의|질문)[하해]\s*(?:주셨|셨)',
                r'(?:에\s*대해서?|관련해서?|대하여)\s*(?:문의|질문|궁금)',
                r'(?:문의|질문)\s*(?:사항|내용)',
                r'[이가]\s*궁금[하해]'
            ],
            
            # 요청/부탁 확인 패턴
            'request': [
                r'요청[을를]?\s*(?:주셨|하셨)',
                r'부탁[을를]?\s*(?:드렸|하셨)',
                r'[해하]달라고\s*(?:하셨|요청)',
                r'[을를가]\s*(?:요청|부탁)[하해]\s*(?:주셨|셨)',
                r'도움[을를]?\s*(?:요청|부탁)',
                r'[을를가]\s*원[하해]\s*(?:신다|십니다)',
                r'[을를가]\s*희망[하해]\s*(?:신다|십니다)'
            ],
            
            # 확인/검증 요청 패턴
            'verification': [
                r'확인[을를]?\s*(?:요청|부탁|원[하해])',
                r'[을를가]\s*확인[하해]\s*(?:주셨|달라)',
                r'점검[을를]?\s*(?:요청|부탁)',
                r'검토[를을]?\s*(?:요청|부탁)',
                r'체크[를을]?\s*(?:요청|부탁)'
            ],
            
            # 정보 제공 요청 패턴
            'information': [
                r'정보[를을]?\s*(?:요청|문의)',
                r'[을를가]\s*알려\s*(?:주셨|달라)',
                r'설명[을를]?\s*(?:요청|부탁)',
                r'안내[를을]?\s*(?:요청|부탁)',
                r'[에대에서]\s*대한?\s*(?:정보|내용|설명)',
                r'[은는이가]\s*어떤지\s*(?:문의|질문)'
            ],
            
            # 상태/현황 확인 패턴
            'status': [
                r'상태[를을]?\s*(?:확인|문의)',
                r'현황[을를]?\s*(?:확인|문의)',
                r'진행\s*(?:상황|상태)[을를]?\s*(?:확인|문의)',
                r'[이가]\s*어떻게\s*(?:되는지|진행)',
                r'언제\s*(?:가능|될지|되는지)\s*(?:문의|질문)'
            ]
        }
        
        # 확인 메시지의 마무리 표현들
        self.confirmation_endings = [
            r'(?:문의|질문)[을를]?\s*(?:주셨습니다|하셨네요|드리셨네요)',
            r'(?:요청|부탁)[을를]?\s*(?:주셨습니다|하셨네요|드리셨네요)',
            r'(?:에\s*대해|관련해서?)\s*(?:문의|질문|궁금해)',
            r'[을를가]\s*(?:말씀|언급)[하해]?\s*(?:주셨|셨)',
            r'(?:라고|하고)\s*(?:하셨|말씀하셨)',
            r'(?:인지|는지)\s*(?:문의|질문|궁금)',
            r'(?:습니다|네요|군요|요)\.?$'
        ]
        
        # 부정적 패턴 (확인이 아닌 경우)
        self.negative_patterns = [
            r'^(?:네|예|안녕|감사)',  # 인사말로 시작
            r'^(?:죄송|미안)',        # 사과로 시작
            r'(?:답변|대답)[을를]?\s*(?:드리|해)',  # 답변 제공
            r'(?:설명|안내)[을를]?\s*(?:드리|해)',  # 설명 제공
            r'^(?:그렇|맞|틀)'        # 직접적인 답변
        ]
    
    def add_pattern(self, category, pattern):
        """새로운 패턴 추가"""
        if category not in self.confirmation_patterns:
            self.confirmation_patterns[category] = []
        self.confirmation_patterns[category].append(pattern)
    
    def get_all_positive_patterns(self):
        """모든 확인 패턴 반환"""
        patterns = []
        for category_patterns in self.confirmation_patterns.values():
            patterns.extend(category_patterns)
        patterns.extend(self.confirmation_endings)
        return patterns


def normalize_text(text):
    """텍스트 정규화"""
    if not text:
        return ""
    
    # 소문자 변환
    text = text.lower()
    
    # 유니코드 정규화
    text = unicodedata.normalize('NFKC', text)
    
    # 연속 공백 정리
    text = re.sub(r'\s+', ' ', text)
    
    # 문장부호 앞뒤 공백 정리
    text = re.sub(r'\s*([.!?,:;])\s*', r'\1 ', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text


def extract_keywords(text):
    """텍스트에서 핵심 키워드 추출"""
    # 조사, 어미 등을 제거하고 핵심 단어만 추출
    # 간단한 형태소 분석 대신 패턴 기반으로 처리
    
    # 조사 제거 패턴
    particles = [r'[은는이가을를에서의와과도로부터까지만']
    
    words = text.split()
    keywords = []
    
    for word in words:
        # 조사 제거
        for particle_pattern in particles:
            word = re.sub(particle_pattern + r'$', '', word)
        
        # 길이가 2 이상이고 의미있는 단어만 추출
        if len(word) >= 2 and not re.match(r'^[0-9]+$', word):
            keywords.append(word)
    
    return keywords

def calculate_semantic_similarity(text1, text2, threshold=0.4):
    """두 텍스트 간의 의미적 유사도 계산"""
    
    # 1. 정확한 문자열 포함 관계 확인
    normalized_text1 = normalize_text(text1)
    normalized_text2 = normalize_text(text2)
    
    # 더 긴 텍스트에서 짧은 텍스트를 찾기
    longer_text = normalized_text1 if len(normalized_text1) > len(normalized_text2) else normalized_text2
    shorter_text = normalized_text2 if len(normalized_text1) > len(normalized_text2) else normalized_text1
    
    if shorter_text in longer_text:
        return 1.0
    
    # 2. 키워드 기반 유사도 계산
    keywords1 = set(extract_keywords(normalized_text1))
    keywords2 = set(extract_keywords(normalized_text2))
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # 공통 키워드 비율 계산
    common_keywords = keywords1.intersection(keywords2)
    keyword_similarity = len(common_keywords) / min(len(keywords1), len(keywords2))
    
    # 3. 문자열 유사도 계산 (SequenceMatcher 사용)
    sequence_similarity = SequenceMatcher(None, normalized_text1, normalized_text2).ratio()
    
    # 4. 가중 평균으로 최종 유사도 계산
    final_similarity = (keyword_similarity * 0.7) + (sequence_similarity * 0.3)
    
    return final_similarity

