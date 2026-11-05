"""Heuristic post-generation lyric comparison, NOT a guarantee of sung words.

Whisper frequently misrecognizes singing. Keep this an optional measurement,
never a gate that silently discards a musically correct output.
"""
from __future__ import annotations
import re
import unicodedata

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def word_tokens(text):
    normalized = unicodedata.normalize('NFKC', str(text)).casefold()
    return _WORD.findall(normalized)


def _lcs_count(reference, observed):
    """Count only words appearing in the same order, unlike a token bag.

    The old `expected_word_coverage_proxy` was 100% for an audio with all
    requested words in random order plus 90 unwanted lyrics. This statistic
    is strictly diagnostic and remains subject to singing-ASR errors.
    """
    if not reference or not observed:
        return 0
    previous = [0] * (len(observed) + 1)
    for item in reference:
        current = [0]
        for index, other in enumerate(observed, 1):
            current.append(previous[index - 1] + 1 if item == other else
                           max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _edit_distance(reference, observed):
    previous = list(range(len(observed) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(observed, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1]


def compare_lyrics(expected_lyrics, recognized_text, reference_text="", source_text=""):
    from .alignment import lyric_parts
    _, parts = lyric_parts(expected_lyrics)
    expected = word_tokens(' '.join(line for part in parts for line in part.sung_lines))
    observed = word_tokens(recognized_text)
    if not expected:
        return {'status': 'not_applicable', 'expected_words': 0}
    # Avoid excessively large quadratic DP for thousands of words.
    if len(expected) * len(observed) > 250000:
        return {'status': 'too_many_words_for_heuristic', 'expected_words': len(expected),
                'recognized_words': len(observed)}
    edits = _edit_distance(expected, observed)
    in_order_hits = _lcs_count(expected, observed)
    from collections import Counter
    remaining = Counter(expected)
    matched = 0
    for word in observed:
        if remaining[word]:
            remaining[word] -= 1
            matched += 1
    novel = len(observed) - matched
    expected_vocabulary = set(expected)
    extra = [word for word in observed if word not in expected_vocabulary]
    extra_sample = extra[:16]
    # Only words absent from the TARGET vocabulary can be attributed as
    # suspicious. Overlapping words cannot prove contamination.
    reference_vocabulary = set(word_tokens(reference_text)) - expected_vocabulary
    source_vocabulary = set(word_tokens(source_text)) - expected_vocabulary
    reference_hits = [word for word in extra if word in reference_vocabulary]
    source_hits = [word for word in extra if word in source_vocabulary]
    decision = ('review_required' if edits/len(expected) >= .30
                or novel > max(3,int(len(expected)*.15))
                or in_order_hits / len(expected) < .72 else 'provisional_pass')
    return {'quality_decision':decision,
            'possible_reference_contamination_words':len(reference_hits),
            'possible_source_contamination_words':len(source_hits),
            'possible_reference_contamination_sample':reference_hits[:12],
            'possible_source_contamination_sample':source_hits[:12],
            'reference_comparison_available':bool(reference_text),
            'source_comparison_available':bool(source_text),
            'status': 'measured_heuristic_only', 'expected_words': len(expected),
            'recognized_words': len(observed),
            'word_error_rate': round(edits / len(expected), 4),
            'expected_word_coverage_proxy': round(in_order_hits / len(expected), 4),
            'recognized_word_precision_proxy': round(in_order_hits / max(1,len(observed)), 4),
            'unordered_vocabulary_overlap_proxy': round(matched / len(expected), 4),
            'word_order_matched_count': in_order_hits,
            'unmatched_recognized_words': novel,
            'potential_unrequested_words_sample': extra_sample,
            'warning': 'Whisper on singing can hallucinate or omit words; listen to the audio to validate.'}
