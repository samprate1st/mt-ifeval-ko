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

"""Library of instructions."""

import collections
import json
import markdown
import logging
import random
import re
import string
import unicodedata # Used to compare Korean characters
from typing import Dict, Optional, Sequence, Union

import langdetect

from ifeval.instruction_utils import ko_instructions_util, en_instructions_util


logger = logging.getLogger(__name__)

_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]

_LANGUAGES = ko_instructions_util.LANGUAGE_CODES

# The relational operation for comparison.
_COMPARISON_RELATION = ("less than", "at least", "equal to")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The number of digit lists.
_NUM_DIGIT_LIST = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = (
    "제 답변은 예입니다.",
    "제 답변은 아니요입니다.",
    "제 답변은 아마도입니다.",
)

# The options of starter keywords. 원본은 # 뒤에
_STARTER_OPTIONS = (
    "저라면", # "I would say",
    "제 답변은", # "My answer is",
    "제가 믿기에는", #"I believe",
    "제 의견은", # "In my opinion",
    "제 생각은" , # "I think",
    "제 생각에 따르면", # "I reckon",
    "제가 느끼기에는", # "I feel",
    "제 관점으로는", # "From my perspective",
    "제가 보기에는", # "As I see it",
    # "According to me",
    "제가 우려하는 부분은" # "As far as I'm concerned",
    "제가 이해하기로는", #"To my understanding",
    "저의 관점으로는", # "In my view",
    "저로써는", # "My take on it is",
    "제가 받아들이기로는", #"As per my perception",
)

# The options of ending keywords.
# 원본 ("Any other questions?", "Is there anything else I can help with?")
_ENDING_OPTIONS = ("다른 질문이 있나요?", "제가 도와드릴 수 있는 다른 것이 있나요?") # 원본을 번역하여 넣었습니다

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section splitter.
_SECTION_SPLITER = ("Section", "SECTION, 섹션") # "섹션" 추가

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500

# The number of syllables in the response.
_NUM_SYLLABLES_LOWER_LIMIT = 100
_NUM_SYLLABLES_UPPER_LIMIT = 500


class Instruction:
    """An instruction template."""

    def __init__(self, instruction_id):
        self.id = instruction_id

    def build_description(self, **kwargs):
        raise NotImplementedError("`build_description` not implemented.")

    def get_instruction_args(self):
        raise NotImplementedError("`get_instruction_args` not implemented.")

    def get_instruction_args_keys(self):
        raise NotImplementedError("`get_instruction_args_keys` not implemented.")

    def check_following(self, value):
        raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(Instruction):
    """Check the language of the entire response."""

    def build_description(self, *, language=None):
        """Build the instruction description.
        Args:
          language: A string representing the expected language of the response. The
            language has to comply to the 97 types defined in
            `langid.py` (https://pypi.org/project/langid/1.1.5/), which follows
            ISO 639-1 codes (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes);
            for example, `en` for English, `zh` for Chinese, `fr` for French.
        Returns:
          A string representing the instruction description.
        """
        self._language = language
        if self._language is None:
            self._language = random.choice(list(_LANGUAGES.keys()))
        # TODO(tianjianlu): opens the description generation to more choices.
        self._description_pattern = (
            "Your ENTIRE response should be in {language} language, no other "
            + "language is allowed."
        )
        return self._description_pattern.format(language=_LANGUAGES[self._language])

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"language": self._language}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["language"]

    def check_following(self, value):
        """Check if the language of the entire response follows the instruction.
        Args:
          value: A string representing the response.
        Returns:
          True if the language of `value` follows instruction; otherwise False.
        """
        assert isinstance(value, str)

        try:
            lang = langdetect.detect(value)
            if lang.startswith("zh"):
                lang = "zh"
            return lang == self._language
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return True


class FormalTextCheckerKorean(Instruction):
    """Check the format text in Korean language"""

    def build_description(self):
        self._description_pattern = (
            "한국어에서 종결어미가 '-니다/ -습니다'로 통일된 문장인 경우를 판단하고 true/false를 반환"
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Check if the entire response in Korean language follows the instruction.
        한국어에서 종결어미가 '-니다/ -습니다'로 통일된 문장인 경우를 판단하고 true/false를 반환

        Args:
          value: A string representing the response.

        Returns:
          True if the entire response in Korean language follows instruction; otherwise False.
        """
        try:
            assert isinstance(value, str)
        except:
            print('Failed for assertion, got non str type input,', value)
            return False

        # 답변이 한국어가 아니라는 의미이므로 즉시 False 반환
        if not check_korean_language_safely(value):
            return False

        value = value.strip()
        print(f"[DEBUG] FormalTextCheckerKorean / value: {value}")
        return check_formal_text_korean(value)

def check_korean_language_safely(text):
    try:
        if not text or text.strip() == "":
            return False
        lang = langdetect.detect(text)
        return lang == "ko"
    except langdetect.lang_detect_exception.LangDetectException:
        # 언어를 감지할 수 없는 경우 기본값 반환
        return False

def check_formal_text_korean(value, strict_mode=True, min_sentence_length=2):
    """한국어 문장이 존대말/경어체로 작성되었는지 판단합니다.
    
    Args:
        value (str): 판단할 한국어 문장
        strict_mode (bool): 엄격 모드. True면 모든 문장이 경어체여야 함.
                           False면 과반수가 경어체면 True 반환
        min_sentence_length (int): 판단 대상 최소 문장 길이
        
    Returns:
        bool: 경어체이면 True, 반말체이면 False
    """
    if not value or not value.strip():
        return False
    
    # 텍스트 전처리
    processed_text = ko_instructions_util.preprocess_korean_text(value)
    
    # 패턴 객체 생성
    patterns = ko_instructions_util.KoreanFormalityPatterns()
    
    # 문장 단위로 분할
    sentences = ko_instructions_util.split_into_sentences(processed_text)
    
    if not sentences:
        return False
    
    formal_count = 0
    informal_count = 0
    
    for sentence in sentences:
        if len(sentence) < min_sentence_length:
            continue
            
        # 경어체 패턴 확인
        is_formal = False
        is_informal = False
        
        # 1. 어말어미 패턴 확인
        for pattern in patterns.get_all_formal_patterns():
            if re.search(pattern, sentence):
                is_formal = True
                break
        
        # 2. 높임 어휘 확인
        if not is_formal:
            for honorific in patterns.honorific_words:
                if re.search(honorific, sentence):
                    is_formal = True
                    break
        
        # 3. 반말체 패턴 확인 (경어체가 아닌 경우에만)
        if not is_formal:
            for pattern in patterns.get_all_informal_patterns():
                if re.search(pattern, sentence):
                    is_informal = True
                    break
        
        # 결과 집계
        if is_formal:
            formal_count += 1
        elif is_informal:
            informal_count += 1
        # 패턴에 매치되지 않는 경우는 중립으로 처리
    
    # 판단 로직
    total_sentences = len([s for s in sentences if len(s) >= min_sentence_length])
    
    if total_sentences == 0:
        return False
    
    if strict_mode:
        # 엄격 모드: 반말이 하나라도 있으면 False
        return informal_count == 0 and formal_count > 0
    else:
        # 관대 모드: 경어체가 과반수이면 True
        return formal_count > informal_count


class NumberOfSentences(Instruction):
    """Check the number of sentences."""

    def build_description(self, *, num_sentences=None, relation=None):
        """Build the instruction description.
        Args:
          num_sentences: An integer specifying the number of sentences as a
            threshold.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of sentences < the threshold;
            if 'at least', the actual number of sentences >= the threshold.
        Returns:
          A string representing the instruction description.
        """
        # The number of sentences as a threshold for comparison.
        self._num_sentences_threshold = num_sentences
        if self._num_sentences_threshold is None or self._num_sentences_threshold < 0:
            self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = (
            "Your response should contain {relation} {num_sentences} sentences."
        )
        return self._description_pattern.format(
            relation=self._comparison_relation,
            num_sentences=self._num_sentences_threshold,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "relation"]

    def check_following(self, value):
        """Check if the number of sentences follows the instruction.
        Args:
          value: A string representing the response.
        Returns:
          True if the response follows the instruction.
        Raise:
            ValueError if the string in `instruction_args` is not in
            [`less_than`, `at_least`].
        """
        num_sentences = ko_instructions_util.count_sentences(value)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_sentences < self._num_sentences_threshold
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_sentences >= self._num_sentences_threshold
        elif self._comparison_relation == _COMPARISON_RELATION[2]:
            return num_sentences == self._num_sentences_threshold


class PlaceholderChecker(Instruction):
    """Check the placeholders in template writing."""

    def build_description(self, *, num_placeholders=None):
        """Build the instruction description.
        Args:
          num_placeholders: An integer denoting the minimum number of
            placeholders required in the response.
        Returns:
          A string representing the instruction description.
        """
        self._num_placeholders = num_placeholders
        if self._num_placeholders is None or self._num_placeholders < 0:
            self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
        self._description_pattern = (
            "The response must contain at least {num_placeholders} placeholders "
            + "represented by square brackets, such as [address]."
        )
        return self._description_pattern.format(num_placeholders=self._num_placeholders)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_placeholders": self._num_placeholders}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_placeholders"]

    def check_following(self, value):
        """Check if the number of placeholders follows the instruction.
        Args:
          value: A string representing the response.
        Returns:
          True if the actual number of placeholders in the response is greater than
          or equal to `num_placeholders`; otherwise, False.
        """
        placeholders = re.findall(r"\[.*?\]", value)
        num_placeholders = len(placeholders)
        return num_placeholders >= self._num_placeholders


class BulletListChecker(Instruction):
    """Checks the bullet list in the prompt."""

    def build_description(self, *, num_bullets=None):
        """Build the instruction description.
        Args:
          num_bullets: An integer specifying the exact number of bullet lists
            that is required to appear in the response.
        Returns:
          A string representing the instruction description.
        """
        self._num_bullets = num_bullets
        if self._num_bullets is None or self._num_bullets < 0:
            self._num_bullets = random.randint(1, _NUM_BULLETS)
        self._description_pattern = (
            "Your answer must contain exactly {num_bullets} bullet points. "
            + "Use the markdown bullet points such as:\n"
            + "* This is point 1. \n"
            + "* This is point 2"
        )
        return self._description_pattern.format(num_bullets=self._num_bullets)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_bullets": self._num_bullets}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_bullets"]

    def check_following(self, value):
        r"""Check if the number of bullet lists meets the requirement.
        Args:
          value: A string representing the response. The response is expected to
            contain some bullet lists that start with `\*`.
        Returns:
          True if the actual number of bullet lists in the response meets the
          requirement.
        """
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
        return num_bullet_lists == self._num_bullets



class DigitListChecker(Instruction):
    """Checks the digit list in the prompt."""

    def build_description(self, *, num_digit_list=None):
        """Build the instruction description.

        Args:
          num_digit_list: An integer specifying the exact number of digit lists
            that is required to appear in the response.

        Returns:
          A string representing the instruction description.
        """
        self._num_digit_list = num_digit_list
        if self._num_digit_list is None or self._num_digit_list < 0:
            self._num_digit_list = random.randint(1, _NUM_DIGIT_LIST)
        self._description_pattern = (
            "Your answer must contain exactly {num_digit_list} digit lists. "
            + "Use the digit list such as:\n"
            + "1. This is 1st digit point\n"
            + "2. This is 2nd digit point\n\n"
            + "Or use the digit list such as:\n"
            + "1) This is 1st digit point\n"
            + "2) This is 2nd digit point"
        )
        return self._description_pattern.format(num_digit_list=self._num_digit_list)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_digit_list": self._num_digit_list}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_digit_list"]

    def check_following(self, value):
        r"""Check if the number of digit lists meets the requirement.

        Args:
          value: A string representing the response. The response is expected to
            contain some digit+")" or digit+"." .

        Returns:
          True if the actual number of digit lists in the response meets the
          requirement.
        """
        num_digit_list = digit_list_checker_rule(value)

        return num_digit_list == self._num_digit_list

def digit_list_checker_rule(value):
    # 텍스트 내용에서 여러 항목을 아래와 같이 숫자로 나열하는 경우를 검색해서 숫자 나열 항목의 갯수를 리턴하는 함수
    # 숫자 나열 항목 예)
    #  1) 항목1
    #  2) 항목2
    # 또는, 
    #  1. 항목1
    #  2. 항목2

    digit_lists = re.findall(r"^\s*\d+\).*$", value, flags=re.MULTILINE)
    digit_lists_2 = re.findall(r"^\s*\d+\..*$", value, flags=re.MULTILINE)
    num_digit_list = len(digit_lists) + len(digit_lists_2)
    
    return num_digit_list

class ConstrainedResponseChecker(Instruction):
    """Checks the constrained response."""

    def build_description(self):
        """Build the instruction description."""
        # A sequence of string(s) representing the options of the expected response.
        self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
        self._description_pattern = (
            "Answer with one of the following options: {response_options}"
        )
        return self._description_pattern.format(
            response_options=self._constrained_responses
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response matches the constrained options.
        Args:
          value: A string representing the response.
        Returns:
          True if the actual response contains one of the options in the constrained
          responses; otherwise False.
        """
        value = value.strip()
        for constrained_response in self._constrained_responses:
            if constrained_response in value:
                return True
        return False


class ConstrainedCustomResponseChecker(Instruction):
    """Checks the constrained custom response."""

    def build_description(self, choices=None):
        """Build the instruction description."""
        # A sequence of string(s) representing the options of the expected response.
        # 만일 choices가 None이거나 빈 리스트이거나 리스트가 아니면 기본 제약 조건 사용
        if choices is None or len(choices) == 0 or not isinstance(choices, list):
            self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
        else:
            self._constrained_responses = choices
        self._description_pattern = (
            "Answer with one of the following options: {response_options}"
        )
        return self._description_pattern.format(
            response_options=self._constrained_responses
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return [{"choices": list(self._constrained_responses)}]

    def check_following(self, value):
        """Checks if the response matches the constrained options.

        Args:
          value: A string representing the response.

        Returns:
          True if the actual response contains one of the options in the constrained
          responses; otherwise False.
        """

        value = value.strip()

        # self._constrained_responses 로 주어진 문장들 중 하나라도 존재하면, True 반환
        for constrained_response in self._constrained_responses:
            response_pattern = r"^\s*" + constrained_response + r".*$"
            response_with_constrained_start = re.search(
                response_pattern, value, flags=re.MULTILINE
            )
            if response_with_constrained_start:
                return True
        return False


# instruction_registry에서 이용 안함
class ConstrainedStartChecker(Instruction):
    """Checks the response start."""

    def build_description(self, *, starter=None):
        """Build the instruction description.
        Args:
          starter: A string representing the keyword that the response should start
            with.
        Returns:
          A string representing the instruction description.
        """
        self._starter = starter.strip() if isinstance(starter, str) else starter
        if self._starter is None:
            self._starter = random.choice(_STARTER_OPTIONS)
        self._description_pattern = (
            "During the conversation, when it is your turn, "
            + "please always start with {starter}"
        )
        return self._description_pattern.format(starter=self._starter)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"starter": self._starter}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["starter"]

    def check_following(self, value):
        """Checks if the response starts with the constrained keyword or phrase.
        Args:
          value: A string representing the response.
        Returns:
          True if the response starts with the given phrase or keyword that is
          contained in `instruction_args`; otherwise, False.
        """
        response_pattern = r"^\s*" + self._starter + r".*$"
        response_with_constrained_start = re.search(
            response_pattern, value, flags=re.MULTILINE
        )
        return True if response_with_constrained_start else False


class HighlightSectionChecker(Instruction):
    """Checks the highlighted section."""

    def build_description(self, *, num_highlights=None):
        """Build the instruction description.
        Args:
          num_highlights: An integer specifying the minimum number of highlighted
            sections.
        Returns:
          A string representing the instruction description.
        """
        self._num_highlights = num_highlights
        if self._num_highlights is None or self._num_highlights < 0:
            self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

        self._description_pattern = (
            "Highlight at least {num_highlights} sections in your answer with "
            + "markdown, i.e. *highlighted section*."
        )

        return self._description_pattern.format(num_highlights=self._num_highlights)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_highlights": self._num_highlights}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_highlights"]

    def check_following(self, value):
        """Checks if the number of highlighted sections meets the requirement.
        Args:
          value: a string representing the response. The response is expected to
            contain highlighted sections in the format of *highlighted*.
        Returns:
          True if the actual number of highlighted sections in the format of
          *highlighted sections* meets the minimum requirement; otherwise False.
        """
        num_highlights = 0
        highlights = re.findall(r"\*[^\n\*]*\*", value)
        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1
        for highlight in double_highlights:
            if highlight.removeprefix("**").removesuffix("**").strip():
                num_highlights += 1

        return num_highlights >= self._num_highlights


class SectionChecker(Instruction):
    """Checks the sections."""

    def build_description(self, *, section_spliter=None, num_sections=None):
        """Build the instruction description.
        Args:
          section_spliter: A string represents the section spliter keyword that
            marks a new section, i.e., `Section` or `SECTION`.
          num_sections: An integer specifying the number of sections.
        Returns:
          A string representing the instruction description.
        """
        self._section_spliter = (
            section_spliter.strip()
            if isinstance(section_spliter, str)
            else section_spliter
        )
        if self._section_spliter is None:
            self._section_spliter = random.choice(_SECTION_SPLITER)

        self._num_sections = num_sections
        if self._num_sections is None or self._num_sections < 0:
            self._num_sections = random.randint(1, _NUM_SECTIONS)

        self._description_pattern = (
            "Your response must have {num_sections} sections. Mark the beginning "
            + "of each section with {section_spliter} X, such as:\n"
            + "{section_spliter} 1\n"
            + "[content of section 1]\n"
            + "{section_spliter} 2\n"
            + "[content of section 2]"
        )

        return self._description_pattern.format(
            num_sections=self._num_sections, section_spliter=self._section_spliter
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "section_spliter": self._section_spliter,
            "num_sections": self._num_sections,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["section_spliter", "num_sections"]

    def check_following(self, value):
        """Checks the response contains multiple sections.
        Args:
          value: A string representing the response. The response is expected
            to contain multiple sections (number of sections is greater than 1).
            A new section starts with `Section 1`, where the number denotes the
            section index.
        Returns:
          True if the number of sections in the response is greater than or equal to
          the minimum number of sections; otherwise, False.
        """
        section_splitter_patten = r"\s?" + self._section_spliter + r"\s?\d+\s?"
        sections = re.split(section_splitter_patten, value)
        num_sections = len(sections) - 1
        return num_sections >= self._num_sections


class ParagraphChecker(Instruction):
    """Checks the paragraphs."""

    def build_description(self, *, num_paragraphs=None):
        """Build the instruction description.
        Args:
          num_paragraphs: An integer specifying the number of paragraphs.
        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._description_pattern = (
            "There should be {num_paragraphs} paragraphs. "
            + "Paragraphs are separated with the markdown divider: ***"
        )

        return self._description_pattern.format(num_paragraphs=self._num_paragraphs)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_paragraphs": self._num_paragraphs}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs"]

    def check_following(self, value):
        """Checks the response contains required number of paragraphs.
        Args:
          value: A string representing the response. The response may contain
            paragraphs that are separated by the markdown divider: `***`.
        Returns:
          True if the actual number of paragraphs is the same as required;
          otherwise, False.
        """
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False

        return num_paragraphs == self._num_paragraphs


class PostscriptChecker(Instruction):
    """Checks the postscript."""

    def build_description(self, *, postscript_marker=None):
        """Build the instruction description.
        Args:
          postscript_marker: A string containing the keyword that marks the start
            of the postscript section.
        Returns:
          A string representing the instruction description.
        """
        self._postscript_marker = (
            postscript_marker.strip()
            if isinstance(postscript_marker, str)
            else postscript_marker
        )
        if self._postscript_marker is None:
            self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)

        self._description_pattern = (
            "At the end of your response, please explicitly add a postscript "
            + "starting with {postscript}"
        )

        return self._description_pattern.format(postscript=self._postscript_marker)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"postscript_marker": self._postscript_marker}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["postscript_marker"]

    def check_following(self, value):
        """Checks if the response follows the postscript format.
        Args:
          value: a string representing the response. The response is expected to
            contain a postscript section.
        Returns:
          True if the response contains a postscript section starting with
          the keyword containing in the `instruction_args`; otherwise False.
        """
        value = value.lower()
        if self._postscript_marker == "P.P.S":
            postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif self._postscript_marker == "P.S.":
            postscript_pattern = r"\s*p\.\s?s\..*$"
        else:
            postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        return True if postscript else False


class RephraseChecker(Instruction):
    """Checks the rephrase."""

    def build_description(self, *, original_message):
        """Build the instruction description.
        Args:
          original_message: A string representing the original message. The
            rephrased response should only change its words/sentences in between
            its two asterisks, for example, *change me*. Both original and rephrased
            messages should contain the changes in the form of *change me*.
        Returns:
          A string representing the instruction description.
        """
        if not self.is_change(original_message):
            raise ValueError(
                f"Message {original_message} does not contain changes "
                "in the form of *change me*."
            )

        self._reference_without_change = original_message
        self._description = (
            "Rephrasing: Your rephrased response should only"
            + "change the words/sentences in between two asterisks"
            + "such as *change me*."
        )
        return self._description

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"original_message": self._reference_without_change}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["original_message"]

    def check_following(self, value):
        r"""Checks if the rephrasing follows the instruction.
        Args:
          value: A string representing the response, which is expected to rephras
            the string of `instruction_args`.
        Returns:
          True if `value` and `instruction_args` only differ by the words/sentences
          in between two asterisks such as *change me*; otherwise, False.
        """

        if not self.is_change(value):
            raise ValueError(
                f"value {value} does not contain changes in the form of *change me*."
            )

        response_without_changes = self.strip_changes(value)
        reference_without_changes = self.strip_changes(self._reference_without_change)

        return response_without_changes == reference_without_changes

    def is_change(self, response):
        """Check if there is change in the response in the form of *change me*."""
        return re.search(r"\*.*\*", response)

    def strip_changes(self, response):
        """Strips off the changes."""
        return re.sub(r"\*.*\*", "", response)


class KeywordChecker(Instruction):
    """Check the exisitence of certain keywords."""

    def build_description(self, *, keywords=None):
        """Build the instruction description.
        Args:
          keywords: A sequence of strings representing the keywords that are
            expected in the response.
        Returns:
          A string representing the instruction description.
        """

        if not keywords:
            raise ValueError("Class KeywordChecker requires keywords")
            # self._keywords = instructions_util.generate_keywords(
            #     num_keywords=_NUM_KEYWORDS
            # )
        else:
            self._keywords = [unicodedata.normalize('NFC', keyword) for keyword in keywords]
        self._keywords = sorted(self._keywords)

        self._description_pattern = "Include keywords {keywords} in the response."

        return self._description_pattern.format(keywords=self._keywords)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"keywords": self._keywords}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keywords"]

    def check_following(self, value):
        """Check if the response contain the expected keywords."""
        for keyword in self._keywords:
            if not re.search(keyword, value, flags=re.IGNORECASE):
                return False
        return True


class KeywordFrequencyChecker(Instruction):
    """Check the keyword frequency."""

    def build_description(self, *, keyword=None, frequency=None, relation=None):
        """Build the instruction description.
        Args:
          keyword: A string representing a keyword that is expected in the response.
          frequency: An integer specifying the number of times `keyword` is expected
            to appear in the response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of occurrences < frequency;
            if 'at least', the actual number of occurrences >= frequency.
        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            raise ValueError("Class KeywordFrequencyChecker requires keyword")
            # self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._keyword = unicodedata.normalize('NFC', keyword.strip())

        self._frequency = frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = (
            "In your response, the word {keyword} should appear {relation} "
            + "{frequency} times."
        )

        return self._description_pattern.format(
            keyword=self._keyword,
            relation=self._comparison_relation,
            frequency=self._frequency,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword", "frequency", "relation"]

    def check_following(self, value):
        """Checks if the response contain the keyword with required frequency."""
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return actual_occurrences < self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return actual_occurrences >= self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[2]:
            return actual_occurrences == self._frequency


class NumberOfWords(Instruction):
    """Checks the number of words."""

    def build_description(self, *, num_words=None, relation=None):
        """Build the instruction description.
        Args:
          num_words: An integer specifying the number of words contained in the
            response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of words < num_words;
            if 'at least', the actual number of words >= num_words.
        Returns:
          A string representing the instruction description.
        """

        self._num_words = num_words
        if self._num_words is None or self._num_words < 0:
            self._num_words = random.randint(
                _NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT
            )

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = "Answer with {relation} {num_words} words."

        return self._description_pattern.format(
            relation=self._comparison_relation, num_words=self._num_words
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_words": self._num_words, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_words", "relation"]

    def check_following(self, value):
        """Checks if the response contains the expected number of words."""
        num_words = ko_instructions_util.count_words(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_words < self._num_words
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_words >= self._num_words
        elif self._comparison_relation == _COMPARISON_RELATION[2]:
            return num_words == self._num_words

class NumberOfSyllables(Instruction):
    """Checks the number of syllables."""

    def build_description(self, *, num_syllables=None, relation=None):
        """Build the instruction description.
        Args:
          num_syllables: An integer specifying the number of syllables contained in the
            response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of syllables < num_syllables;
            if 'at least', the actual number of syllables >= num_syllables.
        Returns:
          A string representing the instruction description.
        """

        self._num_syllables = num_syllables
        if self._num_syllables is None or self._num_syllables < 0:
            self._num_syllables = random.randint(
                _NUM_SYLLABLES_LOWER_LIMIT, _NUM_SYLLABLES_UPPER_LIMIT
            )

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = "Answer with {relation} {num_syllables} syllables."

        return self._description_pattern.format(
            relation=self._comparison_relation, num_syllables=self._num_syllables
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_syllables": self._num_syllables, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_syllables", "relation"]

    def check_following(self, value):
        """Checks if the response contains the expected number of syllables."""
        num_syllables = ko_instructions_util.count_syllables(value)

        print(f"[DEBUG] num_syllables: {num_syllables}, self._num_syllables: {self._num_syllables}, self._comparison_relation: {self._comparison_relation}")

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_syllables < self._num_syllables
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_syllables >= self._num_syllables
        elif self._comparison_relation == _COMPARISON_RELATION[2]:
            return num_syllables == self._num_syllables

class JsonFormat(Instruction):
    """Check the Json format."""

    def build_description(self):
        self._description_pattern = (
            "Entire output should be wrapped in JSON format. You can use markdown"
            " ticks such as ```."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
        except ValueError:
            return False
        return True

class MarkdownFormat(Instruction):
    """Check the markdown format."""

    def build_description(self):
        self._description_pattern = (
            "Entire output should be wrapped in markdown format. You can use markdown"
            " ticks such as ```."
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        value = (
            value.strip()
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        if not value:
            return False
        try:
            markdown.markdown(value)
        except ValueError:
            return False
        return True


class ParagraphFirstWordCheck(Instruction):
    """Check the paragraph and the first word of the nth paragraph."""

    def build_description(
        self, num_paragraphs=None, nth_paragraph=None, first_word=None
    ):
        r"""Build the instruction description.
        Args:
          num_paragraphs: An integer indicating the number of paragraphs expected
            in the response. A paragraph is a subset of the string that is
            expected to be separated by '\n\n'.
          nth_paragraph: An integer indicating the paragraph number that we look at.
            Note that n starts from 1.
          first_word: A string that represent the first word of the bth paragraph.
        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._nth_paragraph = nth_paragraph
        if (
            self._nth_paragraph is None
            or self._nth_paragraph <= 0
            or self._nth_paragraph > self._num_paragraphs
        ):
            self._nth_paragraph = random.randint(1, self._num_paragraphs + 1)

        self._first_word = first_word
        if self._first_word is None:
            raise ValueError("Class ParagraphFirstWordCheck requires first_word")
            # self._first_word = instructions_util.generate_keywords(num_keywords=1)[0] 
        self._first_word = self._first_word.lower()

        self._description_pattern = (
            "There should be {num_paragraphs} paragraphs. "
            + "Paragraphs and only paragraphs are separated with each other by two "
            + "new lines as if it was '\\n\\n' in python. "
            + "Paragraph {nth_paragraph} must start with word {first_word}."
        )

        return self._description_pattern.format(
            num_paragraphs=self._num_paragraphs,
            nth_paragraph=self._nth_paragraph,
            first_word=self._first_word,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs", "nth_paragraph", "first_word"]

    def check_following(self, value):
        """Checks for required number of paragraphs and correct first word.
        Args:
          value: a string representing the response. The response may contain
            paragraphs that are separated by two new lines and the first word of
            the nth paragraph will have to match a specified word.
        Returns:
          True if the number of paragraphs is the same as required and the first
          word of the specified paragraph is the same as required. Otherwise, false.
        """

        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)

        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1

        # check that index doesn't go out of bounds
        if self._nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[self._nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False

        first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"'}

        # get first word and remove punctuation
        word = paragraph.split()[0].strip()
        # TODO(jeffrey): make more complex?
        word = word.lstrip("'")
        word = word.lstrip('"')

        for letter in word:
            if letter in punctuation:
                break
            first_word += letter.lower()

        return num_paragraphs == self._num_paragraphs and first_word == self._first_word


class ParagraphFirstSentenceCheck(Instruction):
    """Check the paragraph and the first sentence of the nth paragraph."""

    def build_description(
        self, num_paragraphs=None, nth_paragraph=None, first_sentence=None
    ):
        r"""Build the instruction description.

        Args:
          num_paragraphs: An integer indicating the number of paragraphs expected
            in the response. A paragraph is a subset of the string that is
            expected to be separated by '\n\n'.
          nth_paragraph: An integer indicating the paragraph number that we look at.
            Note that n starts from 1.
          first_sentence: A string that represent the first sentence of the bth paragraph.

        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._nth_paragraph = nth_paragraph
        if (
            self._nth_paragraph is None
            or self._nth_paragraph <= 0
            or self._nth_paragraph > self._num_paragraphs
        ):
            self._nth_paragraph = random.randint(1, self._num_paragraphs + 1)

        self._first_sentence = first_sentence
        if self._first_sentence is None:
            raise ValueError("Class ParagraphFirstSentenceCheck requires first_sentence")
            # self._first_sentence = ko_instructions_util.generate_keywords(num_keywords=1)[0]
        self._first_sentence = self._first_sentence.lower()


        self._description_pattern = (
            "There should be {num_paragraphs} paragraphs. "
            + "Paragraphs and only paragraphs are separated with each other by two "
            + "new lines as if it was '\\n\\n' in python. "
            + "Paragraph {nth_paragraph} must start with sentence {first_sentence}."
        )

        return self._description_pattern.format(
            num_paragraphs=self._num_paragraphs,
            nth_paragraph=self._nth_paragraph,
            first_sentence=self._first_sentence,
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {
            "num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_sentence": self._first_sentence,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs", "nth_paragraph", "first_sentence"]

    def check_following(self, value):
        """Checks for required number of paragraphs and correct first word.

        Args:
          value: a string representing the response. The response may contain
            paragraphs that are separated by two new lines and the first word of
            the nth paragraph will have to match a specified word.

        Returns:
          True if the number of paragraphs is the same as required and the first
          word of the specified paragraph is the same as required. Otherwise, false.
        """
        try:
            lang = langdetect.detect(value)
        except:
            print("Failed to detect language, got value:", value)
            lang = 'en'

        # split_paragraphs_llm() 함수를 사용하여 문단 분리
        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)

        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1

        # check that index doesn't go out of bounds
        if self._nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[self._nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False

        # get first sentence and remove punctuation
        if lang == 'ko':
            sentences = ko_instructions_util.split_into_sentences(paragraph)
        else:
            sentences = en_instructions_util.split_into_sentences(paragraph)

        sentence = sentences[0].strip()
        sentence = sentence.lstrip("'")
        sentence = sentence.lstrip('"')

        # 문장 맨 뒤의 문장부호는 제거
        first_sentence = sentence.lower() 
        first_sentence = first_sentence.rstrip(".").rstrip(",").rstrip("?").rstrip("!").rstrip("'").rstrip('"')
        self._first_sentence = self._first_sentence.lower().lstrip("'").lstrip('"')
        self._first_sentence = self._first_sentence.rstrip(".").rstrip(",").rstrip("?").rstrip("!").rstrip("'").rstrip('"')

        return num_paragraphs == self._num_paragraphs and first_sentence == self._first_sentence

# TODO(jeffrey) add relation - at least/at most?
class KeySentenceChecker(Instruction):
    """Check the existence of certain key sentences."""

    def build_description(self, key_sentences=None, num_sentences=None):
        """Build the instruction description.
        Args:
          key_sentences: A sequences of strings representing the key sentences that
            are expected in the response.
          num_sentences: The number of key sentences that are expected to be seen in
            the response.
        Returns:
          A string representing the instruction description.
        """

        if not key_sentences:
            # TODO(jeffrey) make a generate sentences function? wonderwords package
            self._key_sentences = set(["For now, this is fine."])
        else:
            self._key_sentences = key_sentences

        if not num_sentences:
            self._num_sentences = random.randint(1, len(self._key_sentences))
        else:
            self._num_sentences = num_sentences

        self._key_sentences = [unicodedata.normalize('NFC', key_sentence) for key_sentence in self._key_sentences]

        self._description_pattern = (
            "Include {num_sentences} of the following sentences {key_sentences}"
        )

        return self._description_pattern.format(
            num_sentences=self._num_sentences, key_sentences=self._key_sentences
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "num_sentences": self._num_sentences,
            "key_sentences": list(self._key_sentences),
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "key_sentences"]

    def check_following(self, value):
        """Checks if the response contains the expected key sentences."""
        count = 0
        all_sentences = ko_instructions_util.split_into_sentences(value)
        for key_sentence in self._key_sentences:
            if key_sentence in all_sentences:
                count += 1

        return count == self._num_sentences


class ForbiddenWords(Instruction):
    """Checks that specified words are not used in response."""

    def build_description(self, forbidden_words=None):
        """Build the instruction description.
        Args:
          forbidden_words: A sequences of strings representing words that are not
            allowed in the response.
        Returns:
          A string representing the instruction description.
        """

        if not forbidden_words:
            raise ValueError("Class ForbiddenWords requires forbidden_words")
            # self._forbidden_words = instructions_util.generate_keywords(
            #     num_keywords=_NUM_KEYWORDS
            # )
        else:
            self._forbidden_words = [unicodedata.normalize('NFC', forbidden_word) for forbidden_word in forbidden_words]
        self._forbidden_words = sorted(self._forbidden_words)
        self._description_pattern = (
            "Do not include keywords {forbidden_words} in the response."
        )

        return self._description_pattern.format(forbidden_words=self._forbidden_words)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"forbidden_words": self._forbidden_words}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["forbidden_words"]

    def check_following(self, value):
        """Check if the response does not contain the expected keywords."""
        for word in self._forbidden_words:
            if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
                return False
        return True


# 현재(25.04.17 lm-eval-harness ifeval 기준) 사용하지 않는 채점기준
class RephraseParagraph(Instruction):
    """Checks that the paragraph is rephrased."""

    def build_description(self, *, original_paragraph, low, high):
        """Builds the instruction description.
        Args:
          original_paragraph: A string presenting the original paragraph. The
            rephrases response should have betweeb low-high words in common.
          low: An integer presenting the lower bound of similar words.
          high: An integer representing the upper bound of similar words.
        Returns:
          A string representing the instruction description.
        """
        # TODO(jeffrey) make more encompassing
        self._original_paragraph = original_paragraph
        self._low = low
        self._high = high

        self._description = (
            "Rephrase the following paragraph: "
            + "{original_paragraph}\nYour response should have "
            + "between {low} and {high} of the same words. "
            + "Words are the same if and only if all of the "
            + "letters, ignoring cases, are the same. For "
            + "example, 'run' is the same as 'Run' but different "
            + "to 'ran'."
        )

        return self._description.format(
            original_paragraph=original_paragraph, low=self._low, high=self._high
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "original_paragraph": self._original_paragraph,
            "low": self._low,
            "high": self._high,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["original_paragraph", "low", "high"]

    def check_following(self, value):
        val_words = re.findall(r"\w+", value.lower())
        original_words = re.findall(r"\w+", self._original_paragraph.lower())
        similar_words = 0

        dict_val = collections.Counter(val_words)
        dict_original = collections.Counter(original_words)

        for word in dict_original:
            similar_words += min(dict_original[word], dict_val[word])

        return similar_words >= self._low and similar_words <= self._high


class TwoResponsesChecker(Instruction):
    """Check that two responses were given."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Give two different responses. Responses and only responses should"
            " be separated by 6 asterisk symbols: ******."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response has two different answers.
        Args:
          value: A string representing the response.
        Returns:
          True if two responses are detected and false otherwise.
        """
        valid_responses = list()
        responses = value.split("******")
        for index, response in enumerate(responses):
            if not response.strip():
                if index != 0 and index != len(responses) - 1:
                    return False
            else:
                valid_responses.append(response)
        return (
            len(valid_responses) == 2
            and valid_responses[0].strip() != valid_responses[1].strip()
        )


class RepeatPromptThenAnswer(Instruction):
    """Checks that Prompt is first repeated then answered."""

    def build_description(self, *, prompt_to_repeat=None):
        """Build the instruction description.
        Args:
          prompt_to_repeat: The prompt that is meant to be repeated.
        Returns:
          A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        self._prompt_to_repeat = unicodedata.normalize('NFC', self._prompt_to_repeat)
        self._description_pattern = (
            "First repeat the request word for word without change,"
            " then give your answer (1. do not say any words or characters"
            " before repeating the request; 2. the request you need to repeat"
            " does not include this sentence)"
        )
        return self._description_pattern

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat"]

    def check_following(self, value):
        if value.strip().lower().startswith(self._prompt_to_repeat.strip().lower()):
            return True
        return False


class RephrasingPromptThenAnswer(Instruction):
    """Checks that Prompt is first rephrased then answered."""

    def build_description(self, *, prompt_to_rephrase=None):
        """Build the instruction description.

        Args:
          prompt_to_rephrase: The prompt that is meant to be rephrased.

        Returns:
          A string representing the instruction description.
        """
        if not prompt_to_rephrase:
            raise ValueError("prompt_to_rephrase must be set.")
        else:
            self._prompt_to_rephrase = prompt_to_rephrase
        self._description_pattern = (
            "First rephrase the request word for word without change,"
            " then give your answer (1. do not say any words or characters"
            " before rephrasing the request; 2. the request you need to rephrase"
            " does not include this sentence)"
        )
        return self._description_pattern

    def get_instruction_args(self):
        return {"prompt_to_rephrase": self._prompt_to_rephrase}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_rephrase"]

    def check_following(self, value):

        if len(value.strip()) == 0:
            return False
        
        first_sentence = value.split("\n")[0].lower()

        answer = check_rephrase(first_sentence, self._prompt_to_rephrase)
        return answer


def check_rephrase(first_sentence, prompt_to_rephrase, similarity_threshold=0.4, debug=False):
    """
    첫 번째 문장이 고객 요청을 확인하는 메시지인지 판단
    
    Args:
        first_sentence (str): 분석할 첫 번째 문장 (LLM 응답의 첫 문장)
        prompt_to_rephrase (str): 원본 고객 요청
        similarity_threshold (float): 의미적 유사도 임계값
        debug (bool): 디버그 정보 출력 여부
    
    Returns:
        bool: 고객 요청을 확인하는 메시지이면 True, 아니면 False
    """
    
    if not first_sentence or not prompt_to_rephrase:
        return False
    
    # 텍스트 정규화
    normalized_sentence = ko_instructions_util.normalize_text(first_sentence)
    normalized_prompt = ko_instructions_util.normalize_text(prompt_to_rephrase)
    
    patterns = ko_instructions_util.RephrasePatterns()
    
    debug_info = {
        'original_sentence': first_sentence,
        'normalized_sentence': normalized_sentence,
        'original_prompt': prompt_to_rephrase,
        'normalized_prompt': normalized_prompt,
        'pattern_matches': [],
        'negative_matches': [],
        'semantic_similarity': 0.0,
        'final_decision': False
    }
    
    # 1. 부정적 패턴 확인 (확인이 아닌 경우)
    for pattern in patterns.negative_patterns:
        if re.search(pattern, normalized_sentence):
            debug_info['negative_matches'].append(pattern)
            if debug:
                print(f"[DEBUG] Negative pattern matched: {pattern}")
            debug_info['final_decision'] = False
            if debug:
                print(f"[DEBUG] Final result: {debug_info}")
            return False
    
    # 2. 확인 패턴 매칭 확인
    positive_pattern_found = False
    for pattern in patterns.get_all_positive_patterns():
        if re.search(pattern, normalized_sentence):
            debug_info['pattern_matches'].append(pattern)
            positive_pattern_found = True
            if debug:
                print(f"[DEBUG] Positive pattern matched: {pattern}")
    
    # 3. 의미적 유사도 계산
    similarity = ko_instructions_util.calculate_semantic_similarity(normalized_sentence, normalized_prompt)
    debug_info['semantic_similarity'] = similarity
    
    if debug:
        print(f"[DEBUG] Semantic similarity: {similarity}")
    
    # 4. 최종 판단
    # 확인 패턴이 있고, 의미적 유사도가 임계값 이상이면 True
    final_decision = positive_pattern_found and similarity >= similarity_threshold
    debug_info['final_decision'] = final_decision
    
    if debug:
        print(f"[DEBUG] Pattern found: {positive_pattern_found}, Similarity: {similarity} >= {similarity_threshold}")
        print(f"[DEBUG] Final result: {final_decision}")
        print(f"[DEBUG] Debug info: {debug_info}")
    
    return final_decision

class EndChecker(Instruction):
    """Checks that the prompt ends with a given phrase."""

    def build_description(self, *, end_phrase=None):
        """Build the instruction description.
        Args:
          end_phrase: A string representing the phrase the response should end with.
        Returns:
          A string representing the instruction description.
        """
        self._end_phrase = (
            end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        )
        if self._end_phrase is None:
            self._end_phrase = random.choice(_ENDING_OPTIONS) # _ENDING_OPTIONS 는 번역됨
        self._end_phrase = unicodedata.normalize('NFC', self._end_phrase)
        self._description_pattern = (
            "Finish your response with this exact phrase {ender}. "
            "No other words should follow this phrase."
        )
        return self._description_pattern.format(ender=self._end_phrase)

    def get_instruction_args(self):
        return {"end_phrase": self._end_phrase}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["end_phrase"]

    def check_following(self, value):
        """Checks if the response ends with the expected phrase."""
        value = value.strip().strip('"').lower()
        self._end_phrase = self._end_phrase.strip().lower()
        return value.endswith(self._end_phrase)


class TitleChecker(Instruction):
    """Checks the response for a title."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Your answer must contain a title, wrapped in double angular brackets,"
            " such as <<poem of joy>>."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response contains a title."""
        pattern = r"<<[^\n]+>>"
        re_pattern = re.compile(pattern)
        titles = re.findall(re_pattern, value)

        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False

# 한국어 특성과 무관
class LetterFrequencyChecker(Instruction):
    """Checks letter frequency."""

    def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
        """Build the instruction description.
        Args:
          letter: A string representing a letter that is expected in the response.
          let_frequency: An integer specifying the number of times `keyword` is
            expected to appear in the response.
          let_relation: A string in (`less than`, `at least`), defining the
            relational operator for comparison. Three relational comparisons are
            supported for now; if 'less than', the actual number of
            occurrences < frequency; if 'at least', the actual number of
            occurrences >= frequency; if 'equal to', the actual number of
            occurrences == frequency.
        Returns:
          A string representing the instruction description.
        """
        if (
            not letter
            or len(letter) > 1
            or ord(letter.lower()) < 97
            or ord(letter.lower()) > 122
        ):
            self._letter = random.choice(list(string.ascii_letters))
        else:
            self._letter = letter.strip()
        self._letter = self._letter.lower()

        self._frequency = let_frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _LETTER_FREQUENCY)

        if let_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif let_relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {let_relation} is given."
            )
        else:
            self._comparison_relation = let_relation

        self._description_pattern = (
            "In your response, the letter {letter} should appear {let_relation}"
            " {let_frequency} times."
        )

        return self._description_pattern.format(
            letter=self._letter,
            let_frequency=self._frequency,
            let_relation=self._comparison_relation,
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value):
        """Checks that the response contains the letter at the right frequency."""
        value = value.lower()
        letters = collections.Counter(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return letters[self._letter] < self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return letters[self._letter] >= self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[2]:
            return letters[self._letter] == self._frequency

# 한국어 특성과 무관
class CapitalLettersEnglishChecker(Instruction):
    """Checks that the response is in english and is in all capital letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Your entire response should be in English, and in all capital letters."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response is in English and in all capital letters."""
        assert isinstance(value, str)

        try:
            return value.isupper() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return True

# 한국어 특성과 무관
class LowercaseLettersEnglishChecker(Instruction):
    """Checks that the response is in english and is in all lowercase letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Your entire response should be in English, and in all lowercase"
            " letters. No capital letters are allowed."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response is in English and in all lowercase letters."""
        assert isinstance(value, str)

        try:
            return value.islower() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return True


class CommaChecker(Instruction):
    """Checks the response for no commas."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "In your entire response, refrain from the use of any commas."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response does not contain commas."""
        return not re.search(r"\,", value)


class PeriodChecker(Instruction):
    """Checks the response for no periods."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "In your entire response, refrain from the use of any periods."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response does not contain periods."""
        return not re.search(r"\.", value)


# 한국어 특성과 무관
class CapitalWordFrequencyChecker(Instruction):
    """Checks frequency of words with all capital letters."""

    def build_description(
        self,
        capital_frequency=None,
        capital_relation=None,
    ):
        """Build the instruction description.
        Args:
          capital_frequency: An integer that represents the number of words that
            should be in all capital letters.
          capital_relation: A string that is 'at least', 'equal to', or 'at most' that refers to
            the frequency.
        Returns:
          A string representing the instruction description.
        """
        self._frequency = capital_frequency
        if self._frequency is None:
            self._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)

        self._comparison_relation = capital_relation
        if capital_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif capital_relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {capital_relation} is given."
            )

        self._description_pattern = (
            "In your response, words with all capital letters should appear"
            " {relation} {frequency} times."
        )

        return self._description_pattern.format(
            frequency=self._frequency, relation=self._comparison_relation
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "capital_frequency": self._frequency,
            "capital_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["capital_frequency", "capital_relation"]

    def check_following(self, value):
        """Checks the frequency of words with all capital letters."""
        # Hyphenated words will count as one word
        words = ko_instructions_util.nltk.word_tokenize(value)
        capital_words = [word for word in words if word.isupper()]

        capital_words = len(capital_words)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return capital_words < self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return capital_words >= self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[2]:
            return capital_words == self._frequency


class QuotationChecker(Instruction):
    """Checks response is wrapped with double quotation marks."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Wrap your entire response with double quotation marks."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response is wrapped with double quotation marks."""
        value = value.strip()
        return len(value) > 1 and value[0] == '"' and value[-1] == '"'

class SingleQuoteChecker(Instruction):
    """Checks response is wrapped with single quotation marks."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Wrap your entire response with single quotation marks."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response is wrapped with single quotation marks."""
        value = value.strip()
        return len(value) > 1 and value[0] == "'" and value[-1] == "'"
