import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Iterable
from num2words import num2words
import logging

LOG = logging.getLogger()


class SentenceProcessor:
    def __init__(self,
                 letters: List[str],
                 replace_numbers_lang: Optional[str],
                 replacement_rules: Dict[str, str]):
        self.letters_regex = re.compile('[' + ''.join(letters) + ']')
        self.non_letters_regex = re.compile('[^' + ''.join(letters) + ']')
        self.replace_numbers_lang = replace_numbers_lang
        self.replacement_rules = replacement_rules

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = ' '.join(filter(len, text.lower().split()))
        return text

    def _num_to_word(self, num: str):
        word = num
        if num.isdigit():
            try:
                word = num2words(num, lang=self.replace_numbers_lang)
            except:
                pass
        return word

    # noinspection PyMethodMayBeStatic
    def _apply_custom_replacement(self, sentence: str) -> str:
        return sentence

    def process_sentence(self, sentence: str) -> str:
        sentence = self._normalize_text(sentence)
        words = []
        for word in re.findall(r'([a-z-]+|[а-яё-]+|[0-9-]+|.?)', sentence):
            words.append(word)
        sentence = self._normalize_text(' '.join(words))
        sentence = self._apply_custom_replacement(sentence)
        sentence = f' {sentence} '
        for rule in self.replacement_rules.items():
            key, value = rule
            key = self._normalize_text(key)
            value = self._normalize_text(value)
            sentence = sentence.replace(key, value)
        if self.replace_numbers_lang:
            sentence = ' '.join(map(self._num_to_word, sentence.split()))
        sentence = self.non_letters_regex.sub(' ', sentence)
        sentence = self._normalize_text(sentence)
        return sentence


class TextProcessor:
    def __init__(self,
                 sentence_processor: SentenceProcessor):
        self.eos_regex = re.compile('[.?!;\n]')
        self.sentence_processor = sentence_processor

    @staticmethod
    def _decode_blob(blob: bytes) -> Optional[str]:
        for encoding in ['utf-8', 'cp1251']:
            try:
                text = blob.decode(encoding)
                return text
            except:
                pass
        return None

    def process_file(self, path: Path) -> Iterable[str]:
        if not path.is_file():
            return
        print(f'Processing file {path}', file=sys.stderr)
        blob = path.read_bytes()
        text = self._decode_blob(blob)
        if text is None:
            print(f'Unable to decode {path}', file=sys.stderr)
            return
        for sentence in self.eos_regex.split(text):
            new_sentence = self.sentence_processor.process_sentence(sentence)
            if len(new_sentence) > 0:
                yield new_sentence



class FilterBadTokensTextProcessor:
    def __init__(self):
        pass

    def process_tokens(self, c):
        tokens = {"Ё": "Е",
                  "ё": "е",
                  "<" : "",
                  ">" : "",
                  "?" : "",
                  "!" : "",
                  "," : "",
                  "." : "",
                  "?" : "",
                  ":" : "",
                  "`" : "",
                  "%" : "",
                  "-" : " ",
                  "–" : "",
                  "—" : ""}
        return c if c not in tokens else tokens[c]

    def process_sentence(self, text):
        processed_text = []
        for word in text.split(' '):
            processed_word = ''.join([self.process_tokens(c) for c in word.upper()])
            if len(processed_word) == 0:
                continue
            if not all(c.isalpha() for c in processed_word):
                LOG.debug('Skipping word {}'.format(processed_word))
                continue
            processed_text.append(processed_word)
        return ' '.join(processed_text)
