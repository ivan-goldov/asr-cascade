from typing import List, Tuple
from sacrebleu import corpus_bleu
from stt_metrics.wer import WER
from itertools import permutations
from tqdm import tqdm


def __levenshtein(a: List, b: List) -> int:
    """Calculates the Levenshtein distance between a and b.
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


refs = []
#     ('АЛЛО', ['АЛО', 'АЛЕ', 'АЛЛЕ']),
#     ('ЦБЭСС', ['ЦЭБЭЭС', 'CBS', 'ЦБЭС']),
#     ('КЛОУЗД', ['CLOSED']),
#     ('ПРЕПРОЦЕССИНГА', ['ПРИПРОЦЕССИНГА']),
#     ('ДА', ['ДАМ']),
#     ('КТО ТО', ['КТОТО']),
#     ('ЧТО ТО', ['ЧТОТО']),
#     ('КАК ТО', ['КАКТО']),
#     ('ГДЕ ТО', ['ГДЕТО']),
#     ('ПОЧЕМУ ТО', ['ПОЧЕМУТО']),
#     ('КАКОЕ ТО', ['КАКОЕТО']),
#     ('КОГО ТО', ['КОГОТО']),
#     ('КАКИЕ ТО', ['КАКИЕТО']),
#     ('КАКОЙ ТО', ['КАКОЙТО']),
#     ('КАКАЯ ТО', ['КАКАЯТО']),
#     ('ПСЕВДОПЬЕСЫ', ['ПСЕВДО ПЬЕСЫ']),
#     ('СЕЙЧАС', ['ЩАС']),
#     ('НЕВАЖНО', ['НЕ ВАЖНО']),
#     ('НАВЕРНОЕ', ['НАВЕРНО']),
#     ('Ф', ['ЭФ', 'F']),
#     ('НЕ КАРДИНАЛЬНЫЙ', ['НЕКАРДИНАЛЬНЫЙ']),
#     ('НЕТ У', ['НЕТУ']),
#     ('НЕТ', ['НЕ']),
#     ('СВЯЗАНО', ['СВЯЗАНЫ']),
#     ('В ВИДУ', ['ВВИДУ']),
#     ('ВСЕ ТАКИ', ['ВСЕТАКИ']),
# ]


def apply_aliases(h):
    h = h.upper()
    h = ' ' + h + ' '
    for center, aliases in refs:
        for a in aliases:
            h = h.replace(' ' + a + ' ', ' ' + center + ' ')
    return h[1:-1]


def word_error_rate(hypotheses: List[str], references: List[str], return_data=False) -> Tuple[float, int, int]:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same length.

    Args:
        hypotheses: list of hypotheses
        references: list of references

    Returns:
        (float) average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError("In word error rate calculation, hypotheses and reference"
                         " lists must have the same number of elements. But I got:"
                         "{0} and {1} correspondingly".format(len(hypotheses), len(references)))
    wer = WER()
    wers = []
    for h, r in zip(hypotheses, references):
        h_list = apply_aliases(h).split()
        r_list = apply_aliases(r).split()
        wer_data = wer.get_metric_data(hyp=' '.join(h_list), ref=' '.join(r_list))
        wer_value = wer.calculate_metric(wer_data)
        wr = max(len(r_list), len(h_list), 1)
        sc = round(wer_value * wr)
        # sc = __levenshtein(h_list, r_list)
        words += wr
        scores += sc
        wers.append((wer_data.diff_hyp, wer_data.diff_ref, sc / wr))
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = None
    if return_data:
        return wer, scores, words, wers
    else:
        return wer, scores, words


def speaker_independent_word_error_rate(hypotheses: List[List[str]], references: List[List[str]], return_data=False) \
        -> Tuple[float, int, int]:
    scores, words = 0, 0
    all_data = []
    wers = []
    for h, r in tqdm(zip(hypotheses, references), total=len(hypotheses)):
        best_wer, best_sc, best_wr = 10, None, None
        best_data = []
        for perm_h in permutations(h):
            wer, sc, wr, data = word_error_rate(list(perm_h), r, return_data=True)
            if wer <= best_wer:
                best_wer, best_sc, best_wr = wer, sc, wr
                best_data = data
        wers.append(best_wer)
        scores += best_sc
        words += best_wr
        all_data += best_data
    if len(wers) != 0:
        wer = sum(wers) / len(wers)
    else:
        wer = 0
    if return_data:
        return wer, scores, words, all_data
    else:
        return wer, scores, words


def calculate_bleu(hypotheses: List[str], references: List[str]) -> float:
    """
    Computes sacrebleu between two texts represented as
    corresponding lists of string. Hypotheses and references must have same length.

    Args:
        hypotheses: list of hypotheses
        references: list of references

    Returns:
        (float) bleu score
    """
    bleu = corpus_bleu(hypotheses, [references])
    return bleu.score
