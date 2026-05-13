import unicodedata
import re

def detokenize_punctuation(s: str) -> str:
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\[\s+", "[", s)
    s = re.sub(r"\s+\]", "]", s)
    s = re.sub(r'"\s+', '"', s)
    s = re.sub(r'\s+"', '"', s)
    s = re.sub(r"'\s+", "'", s)
    s = re.sub(r"\s+'", "'", s)
    return s

def sanitize(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = detokenize_punctuation(s)
    return s

def test_sanitize_detokenizes_punctuation():
    source = "The side effect of depending only on cars as the main transportation are remarkable ."

    assert sanitize(source) == "The side effect of depending only on cars as the main transportation are remarkable."


def test_sanitize_removes_control_characters_and_newlines():
    assert sanitize("Hello\x00\n world !") == "Hello world!"
