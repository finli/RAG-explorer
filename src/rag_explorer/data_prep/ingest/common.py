"""Shared helpers for ingesting Reddit data into a unified document schema.

Every cleaned doc is a dict with the same keys (see ``SCHEMA``):
id, title, post, source, links, doc_type, source_url,
time_utc, upvote, num_comments.
"""

import hashlib
import html
import re
import unicodedata

from ..schema import DOC_FIELDS, validate

# Remove markdown from before links [text](url) or [text] (url)
MD_LINK = re.compile(r"\[([^\]]+)\]\s?\(https?://[^)\s]+\)")
URL = re.compile(r"https?://[^\s)\]>,]+")


def clean_field(field: str):
    """Clean a Reddit text field.

    Unescapes HTML entities, normalizes unicode (NFKC), replaces smart quotes,
    bullets and zero-width/non-breaking spaces with plain equivalents, collapses
    blank lines, keeps the text of markdown links, and replaces bare URLs with
    "[URL]".

    Args:
        field (str): The raw text field from Reddit.

    Returns:
        str: The cleaned text.
    """
    field = html.unescape(field)

    # --- Unicode normalization (safe, no escape corruption)
    field = unicodedata.normalize("NFKC", field)

    # Replace reddit unicode
    field = (
        field.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2022", "-")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("&#x200B;", "")
    )

    # Collapse multiple newlines → single newline
    field = re.sub(r"\n\s*\n+", "\n", field)

    field = MD_LINK.sub(r"\1", field)  # keep link text, drop the URL
    field = re.sub(r"https?://\S+", "[URL]", field)  # take urls out of field

    return field


def _num(value, cast):
    """Cast a value with ``cast``, or return None if it is None or NaN.

    Args:
        value: The value to convert.
        cast (Callable): The type to convert to, e.g. float or int.

    Returns:
        The converted value, or None if missing.
    """
    return None if value is None or value != value else cast(value)


def make_doc(
    *,
    title: str,
    post: str,
    source: str,
    doc_type: str,
    source_url=None,
    time_utc=None,
    upvote=None,
    num_comments=None,
) -> dict:
    """Build a document in the unified schema.

    Pass the raw title and post; cleaning happens here. Links are extracted
    from the raw post before URLs are replaced. The id is a short SHA-1 hash of
    source_url, or of the cleaned title and time_utc if there is no source_url.

    Args:
        title (str): The raw title.
        post (str): The raw post body.
        source (str): The subreddit or origin; stored lowercased.
        doc_type (str): The kind of document, e.g. "post" or "guide".
        source_url (str, optional): The URL of the original document.
        time_utc (float, optional): The creation time as a UTC timestamp.
        upvote (float, optional): The number of upvotes.
        num_comments (int, optional): The number of comments.

    Returns:
        dict: A document with exactly the keys in ``SCHEMA``.
    """
    clean_title = clean_field(title)
    key = source_url or f"{clean_title}_{time_utc}"

    # extract links from the raw text, before URLs are replaced by [URL]
    links = list(dict.fromkeys(URL.findall(html.unescape(post))))  # deduped, in order

    doc = {
        "id": hashlib.sha1(key.encode()).hexdigest()[:12],
        "title": clean_title,
        "post": clean_field(post),
        "source": source.lower(),
        "links": links,
        "doc_type": doc_type,
        "source_url": source_url,
        "time_utc": _num(time_utc, float),
        "upvote": _num(upvote, float),
        "num_comments": _num(num_comments, int),
    }
    # assert list(doc) == DOC_FIELDS, "doc keys drifted from SCHEMA"
    validate(record=doc, fields=DOC_FIELDS)
    return doc
