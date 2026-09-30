"""DraCor corpus fetching, parsing, and pattern-matching for dialogue tasks."""
import gzip
import json
import pickle
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree as ET

API = "https://dracor.org/api/v1"
CACHE = Path(__file__).parent / "_dracor_cache"
TEI = "{http://www.tei-c.org/ns/1.0}"
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _http_get(url, accept):
    req = urllib.request.Request(url, headers={"Accept": accept, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    if data[:2] == b"\x1f\x8b":  # gzip magic — server compressed regardless
        data = gzip.decompress(data)
    return data.decode("utf-8")


def list_plays(corpus):
    cache = CACHE / f"{corpus}_plays.json"
    if cache.exists():
        return json.loads(cache.read_text())
    data = json.loads(_http_get(f"{API}/corpora/{corpus}", "application/json"))
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def fetch_tei(corpus, play_name):
    cache = CACHE / corpus / f"{play_name}.xml"
    if cache.exists():
        return cache.read_text()
    cache.parent.mkdir(parents=True, exist_ok=True)
    xml = _http_get(f"{API}/corpora/{corpus}/plays/{play_name}/tei", "application/xml")
    cache.write_text(xml)
    return xml


_SKIP_TAGS = frozenset({TEI + "stage", TEI + "speaker"})


def _iter_text_skip(elem, skip_tags):
    if elem.text:
        yield elem.text
    for child in elem:
        if child.tag in skip_tags:
            if child.tail:
                yield child.tail
            continue
        yield from _iter_text_skip(child, skip_tags)
        if child.tail:
            yield child.tail


def _flatten_text(elem):
    return re.sub(r"\s+", " ", "".join(_iter_text_skip(elem, _SKIP_TAGS))).strip()


def parse_play(xml):
    root = ET.fromstring(xml)
    title_raw = root.findtext(".//tei:titleStmt/tei:title", namespaces=NS) or "?"
    title = re.sub(r"\s+", " ", title_raw).strip()
    persname = root.find(".//tei:titleStmt/tei:author/tei:persName", namespaces=NS)
    if persname is not None:
        author = re.sub(r"\s+", " ", " ".join(persname.itertext())).strip()
    else:
        author_el = root.find(".//tei:titleStmt/tei:author", namespaces=NS)
        author = re.sub(r"\s+", " ", "".join(author_el.itertext())).strip() if author_el is not None else "?"
    chars = {}
    for person in root.findall(".//tei:particDesc//tei:person", namespaces=NS):
        xid = person.get("{http://www.w3.org/XML/1998/namespace}id")
        name_el = person.find("tei:persName", namespaces=NS)
        if xid:
            name = "".join(name_el.itertext()).strip() if name_el is not None else xid
            name = re.sub(r"\s+", " ", name).strip(".,;: ")
            chars[xid] = name
    turns = []
    for sp in root.iter(TEI + "sp"):
        who = sp.get("who")
        if not who:
            continue
        ids = who.lstrip("#").split()
        if len(ids) != 1:
            continue
        sid = ids[0].lstrip("#")
        text = _flatten_text(sp)
        if text:
            sents = split_sentences(text)
            turns.append({"sid": sid, "name": chars.get(sid, sid), "text": text, "sents": sents})
    return {"title": title, "author": author, "turns": turns}


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'‘“])")
_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|Mt|Lt|Capt|Col|Gen|Hon|Maj|Rev|Prof|Esq|Sgt|Cpl|"
    r"vs|etc|ie|eg|cf|No|Vol|pp|p|Sec|Art|Co|Inc|Ltd)\.\s*$",
    re.IGNORECASE,
)


def split_sentences(text):
    parts = _SENT_SPLIT.split(text)
    out = []
    buf = ""
    for p in parts:
        buf = (buf + " " + p) if buf else p
        if not _ABBREV_RE.search(buf):
            out.append(buf.strip())
            buf = ""
    if buf:
        out.append(buf.strip())
    return [s for s in out if s]


def parse_pattern(pattern):
    runs = []
    for c in pattern:
        if runs and runs[-1][0] == c:
            runs[-1] = (c, runs[-1][1] + 1)
        else:
            runs.append((c, 1))
    return runs


def _canonical(seq):
    """Relabel a sequence by order of first appearance: e.g. ['B','B','A'] -> [0,0,1]."""
    m = {}
    return [m.setdefault(x, len(m)) for x in seq]


def find_candidates(plays, pattern):
    """Windows whose per-turn speaker sequence is isomorphic to the value pattern.

    `pattern` is a string of value digits (arbitrary arity). Consecutive equal values
    collapse into one turn (the same speaker saying several sentence-lines). A window
    matches when its distinct-speaker count equals the pattern's distinct-value count and
    its speaker sequence, relabelled by first appearance, equals the value sequence
    relabelled the same way — so value i always maps to one speaker and different values
    to different speakers (binary is just the 2-value case).
    """
    runs = parse_pattern(pattern)
    n = len(runs)
    target = _canonical([c for c, _ in runs])
    k = (max(target) + 1) if target else 0
    out = []
    for play in plays:
        turns = play["turns"]
        if len(turns) < n:
            continue
        for start in range(len(turns) - n + 1):
            window = turns[start:start + n]
            sids = [t["sid"] for t in window]
            if len(set(sids)) != k or _canonical(sids) != target:
                continue
            if len({t["name"] for t in window}) != k:
                continue
            sent_lists = []
            ok = True
            for (_, length), turn in zip(runs, window):
                sents = turn["sents"]
                if len(sents) < length:
                    ok = False
                    break
                sent_lists.append(sents[:length])
            if ok:
                out.append((play, start, sent_lists))
    return out


def build_corpus(corpus, workers=8, verbose=False):
    """Fetch + parse a corpus, with a parsed-plays pickle cache for fast re-load."""
    parsed_cache = CACHE / f"{corpus}_parsed.pkl"
    if parsed_cache.exists():
        with open(parsed_cache, "rb") as f:
            return pickle.load(f)
    listing = list_plays(corpus)
    names = [p["name"] for p in listing["plays"]]
    parsed = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_tei, corpus, n): n for n in names}
        for fut in as_completed(futs):
            name = futs[fut]
            done += 1
            try:
                parsed.append(parse_play(fut.result()))
            except Exception as e:
                if verbose:
                    print(f"  skip {name}: {type(e).__name__} {e}", flush=True)
            if verbose and done % 50 == 0:
                print(f"  fetched/parsed {done}/{len(names)}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(parsed_cache, "wb") as f:
        pickle.dump(parsed, f)
    return parsed


def load_corpora(corpora, verbose=False):
    plays = []
    for c in corpora:
        plays.extend(build_corpus(c, verbose=verbose))
    return plays
