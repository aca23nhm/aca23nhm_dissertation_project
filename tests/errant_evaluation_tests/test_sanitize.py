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

# Test with the first example
source = "The side effect of depending only on cars as the main transportation are remarkable ."
reference = "The side effect of depending only on cars as our main means of transportation are remarkable ."

print("Original source:", repr(source))
print("Sanitized source:", repr(sanitize(source)))
print("Original reference:", repr(reference))
print("Sanitized reference:", repr(sanitize(reference)))