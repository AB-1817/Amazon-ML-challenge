import re
import unicodedata
from typing import Set, Tuple, List

# Compiled regex patterns for speed
RE_PROTO_WWW = re.compile(r'https?://(?:www\.)?|www\.', re.IGNORECASE)
RE_DOMAIN_SUFFIX = re.compile(r'\.(?:com|org|net|in|co|gov|edu|fr|io)\b', re.IGNORECASE)
RE_PUNCT = re.compile(r'[^\w\s]')
RE_SPACES = re.compile(r'\s+')
RE_NUMBERS = re.compile(r'\b\d+\b')

# Legal suffixes to strip from business names for normalized matching
# Ordered with longer multi-word phrases first
SUFFIXES = [
    r'\bprivate\s+limited\b', r'\bpvt\s+ltd\b', r'\bpvt\s+limited\b',
    r'\bincorporated\b', r'\bcorporation\b', r'\benterprises\b', r'\benterprise\b',
    r'\bassociates\b', r'\bindustries\b', r'\bindustry\b',
    r'\bholding\b', r'\bholdings\b', r'\bservices\b', r'\bservice\b',
    r'\bcompany\b', r'\blimited\b',
    r'\bllc\b', r'\binc\b', r'\bcorp\b', r'\bltd\b', r'\bllp\b', r'\bco\b',
    r'\bsarl\b', r'\bsas\b', r'\bsa\b', r'\bsci\b', r'\beurl\b', r'\bsasu\b'
]
RE_LEGAL_SUFFIXES = re.compile(r'|'.join(SUFFIXES), re.IGNORECASE)

# Address canonical substitutions
ADDR_SUBSTITUTIONS = {
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bdr\b': 'drive',
    r'\bave\b': 'avenue',
    r'\bav\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bbd\b': 'boulevard',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bpkwy\b': 'parkway',
    r'\bhwy\b': 'highway',
    r'\bapt\b': 'unit',
    r'\bste\b': 'unit',
    r'\bsuite\b': 'unit',
    r'\bopp\b': 'near',
    r'\bopposite\b': 'near',
}
RE_ADDR_SUBS = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in ADDR_SUBSTITUTIONS.items()]


def clean_unicode(text: str) -> str:
    """Normalize unicode characters (e.g. French accents) and replace replacement chars."""
    if not text:
        return ""
    # NFKD normalisation decomposes accents (e.g., 'e' with acute -> 'e' + combining mark)
    text = unicodedata.normalize('NFKD', str(text))
    # Encode to ASCII bytes, ignoring non-ASCII combining marks, then decode back
    text = text.encode('ASCII', 'ignore').decode('ASCII')
    return text


def clean_business_name(raw_name: str) -> Tuple[str, str]:
    """Clean business name and return (normalized_full_name, suffix_stripped_name)."""
    if not raw_name:
        return "", ""
    
    # 1. Unicode cleaning
    text = clean_unicode(raw_name).lower()
    
    # 2. Remove URLs/domain references while preserving stem
    text = RE_PROTO_WWW.sub('', text)
    text = RE_DOMAIN_SUFFIX.sub(' ', text)
    
    # 3. Replace punctuation with space
    text = RE_PUNCT.sub(' ', text)
    
    # 4. Clean up spaces
    normalized = RE_SPACES.sub(' ', text).strip()
    
    # 5. Create suffix-stripped core name
    core = RE_LEGAL_SUFFIXES.sub(' ', normalized)
    core = RE_SPACES.sub(' ', core).strip()
    
    # Fallback to normalized if stripping legal suffixes left nothing
    if not core:
        core = normalized
        
    return normalized, core


def clean_address(raw_address: str) -> Tuple[str, List[str]]:
    """Clean address string and extract list of standalone numerical tokens."""
    if not raw_address:
        return "", []
    
    # 1. Unicode cleaning & lowercasing
    text = clean_unicode(raw_address).lower()
    
    # 2. Extract discrete numbers before punctuation stripping
    # (e.g. house numbers, postal codes, zip codes)
    num_tokens = sorted(list(set(RE_NUMBERS.findall(text))))
    
    # 3. Standardize common street/unit abbreviations
    for pattern, repl in RE_ADDR_SUBS:
        text = pattern.sub(repl, text)
        
    # 4. Replace punctuation with space
    text = RE_PUNCT.sub(' ', text)
    cleaned_address = RE_SPACES.sub(' ', text).strip()
    
    return cleaned_address, num_tokens
