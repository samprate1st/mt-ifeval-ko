# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Registry of all instructions."""
from ifeval.instructions import en_instructions
from ifeval.instructions import ko_instructions

_KEYWORD = "keywords:"

_LANGUAGE = "language:"

_LENGTH = "length_constraints:"

_CONTENT = "detectable_content:"

_FORMAT = "detectable_format:"

_MULTITURN = "multi-turn:"

_COMBINATION = "combination:"

_STARTEND = "startend:"

_CHANGE_CASES = "change_case:"

_PUNCTUATION = "punctuation:"

_LETTERS = "letters:"

_SPECIAL_CHARACTER = "special_character:"

EN_INSTRUCTION_DICT = {
    _KEYWORD + "existence": en_instructions.KeywordChecker,
    _KEYWORD + "frequency": en_instructions.KeywordFrequencyChecker,
    # TODO(jeffreyzhou): make a proper set of sentences to choose from
    # _KEYWORD + "key_sentences": instructions.KeySentenceChecker,
    _KEYWORD + "forbidden_words": en_instructions.ForbiddenWords,
    _KEYWORD + "letter_frequency": en_instructions.LetterFrequencyChecker,
    _LANGUAGE + "response_language": en_instructions.ResponseLanguageChecker,
    _LENGTH + "number_sentences": en_instructions.NumberOfSentences,
    _LENGTH + "number_paragraphs": en_instructions.ParagraphChecker,
    _LENGTH + "number_words": en_instructions.NumberOfWords,
    _LENGTH + "nth_paragraph_first_word": en_instructions.ParagraphFirstWordCheck,
    _CONTENT + "number_placeholders": en_instructions.PlaceholderChecker,
    _CONTENT + "postscript": en_instructions.PostscriptChecker,
    _FORMAT + "number_bullet_lists": en_instructions.BulletListChecker,
    # TODO(jeffreyzhou): Pre-create paragraph or use prompt to replace
    # _CONTENT + "rephrase_paragraph": instructions.RephraseParagraph,
    _FORMAT + "constrained_response": en_instructions.ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": (
        en_instructions.HighlightSectionChecker),
    _FORMAT + "multiple_sections": en_instructions.SectionChecker,
    # TODO(tianjianlu): Re-enable rephrasing with preprocessing the message.
    # _FORMAT + "rephrase": instructions.RephraseChecker,
    _FORMAT + "json_format": en_instructions.JsonFormat,
    _FORMAT + "title": en_instructions.TitleChecker,
    # TODO(tianjianlu): Re-enable with specific prompts.
    # _MULTITURN + "constrained_start": instructions.ConstrainedStartChecker,
    _COMBINATION + "two_responses": en_instructions.TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": en_instructions.RepeatPromptThenAnswer,
    _STARTEND + "end_checker": en_instructions.EndChecker,
    _CHANGE_CASES
    + "capital_word_frequency": en_instructions.CapitalWordFrequencyChecker,
    _CHANGE_CASES
    + "english_capital": en_instructions.CapitalLettersEnglishChecker,
    _CHANGE_CASES
    + "english_lowercase": en_instructions.LowercaseLettersEnglishChecker,
    _PUNCTUATION + "no_comma": en_instructions.CommaChecker,
    _STARTEND + "quotation": en_instructions.QuotationChecker,
}


KO_INSTRUCTION_DICT = {
    _KEYWORD + "existence": ko_instructions.KeywordChecker,
    _KEYWORD + "frequency": ko_instructions.KeywordFrequencyChecker,
    # TODO(jeffreyzhou): make a proper set of sentences to choose from
    # _KEYWORD + "key_sentences": instructions.KeySentenceChecker,
    _KEYWORD + "forbidden_words": ko_instructions.ForbiddenWords,
    _KEYWORD + "letter_frequency": ko_instructions.LetterFrequencyChecker,
    _LANGUAGE + "response_language": ko_instructions.ResponseLanguageChecker,
    _LENGTH + "number_paragraphs": ko_instructions.ParagraphChecker,
    _LENGTH + "number_sentences": ko_instructions.NumberOfSentences,
    _LENGTH + "number_words": ko_instructions.NumberOfWords,
    _LENGTH + "number_syllables": ko_instructions.NumberOfSyllables,
    _LENGTH + "nth_paragraph_first_word": ko_instructions.ParagraphFirstWordCheck,
    _CONTENT + "number_placeholders": ko_instructions.PlaceholderChecker,
    _CONTENT + "postscript": ko_instructions.PostscriptChecker,
    _FORMAT + "number_bullet_lists": ko_instructions.BulletListChecker,
    # TODO(jeffreyzhou): Pre-create paragraph or use prompt to replace
    # _CONTENT + "rephrase_paragraph": instructions.RephraseParagraph,
    _FORMAT + "constrained_response": ko_instructions.ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": (
        ko_instructions.HighlightSectionChecker),
    _FORMAT + "multiple_sections": ko_instructions.SectionChecker,
    # TODO(tianjianlu): Re-enable rephrasing with preprocessing the message.
    # _FORMAT + "rephrase": instructions.RephraseChecker,
    _FORMAT + "json_format": ko_instructions.JsonFormat,
    _FORMAT + "markdown_format": ko_instructions.MarkdownFormat,
    _FORMAT + "title": ko_instructions.TitleChecker,
    # TODO(tianjianlu): Re-enable with specific prompts.
    # _MULTITURN + "constrained_start": instructions.ConstrainedStartChecker,
    _COMBINATION + "two_responses": ko_instructions.TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": ko_instructions.RepeatPromptThenAnswer,
    _STARTEND + "end_checker": ko_instructions.EndChecker,
    _CHANGE_CASES
    + "capital_word_frequency": ko_instructions.CapitalWordFrequencyChecker,
    _CHANGE_CASES
    + "english_capital": ko_instructions.CapitalLettersEnglishChecker,
    _CHANGE_CASES
    + "english_lowercase": ko_instructions.LowercaseLettersEnglishChecker,
    _PUNCTUATION + "no_comma": ko_instructions.CommaChecker,
    _STARTEND + "quotation": ko_instructions.QuotationChecker,
    _STARTEND + "single_quote": ko_instructions.SingleQuoteChecker,
}


INSTRUCTION_DICT = {}
INSTRUCTION_DICT.update({"en:" + k: v for k, v in EN_INSTRUCTION_DICT.items()})
INSTRUCTION_DICT.update({"ko:" + k: v for k, v in KO_INSTRUCTION_DICT.items()})
