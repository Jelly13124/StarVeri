import PyPDF2
from pydantic import BaseModel
import requests
import os
import pandas as pd
import re
from unidecode import unidecode
from scholarly import scholarly
import logging
from typing import List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
try:
    import google.genai as genai
    from google.genai.types import Tool, GoogleSearch, ThinkingConfig  # ← add these
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    Tool = None
    GENAI_AVAILABLE = False
    print("Warning: Google Generative AI library not available. Some features will be disabled.")
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from enum import Enum

# --- Configuration ---
GOOGLE_API_KEY = None

def set_google_api_key(api_key: str):
    """Set Google Gemini API key."""
    global GOOGLE_API_KEY
    GOOGLE_API_KEY = api_key


# --- Step 1: Read PDF and extract bibliography section ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from all pages of the PDF."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_bibliography_section(text: str, keywords: List[str] = [
    # English
    "Reference", "References", "Bibliography", "Works Cited",
    # Chinese
    "参考文献", "參考文獻",
    # Japanese
    "参考資料",
]) -> str:
    """
    Find the last occurrence of any keyword from 'keywords'
    and return the text from that point onward.
    """
    last_index = -1
    for keyword in keywords:
        index = text.lower().rfind(keyword.lower())
        if index > last_index:
            last_index = index
    if last_index == -1:
        raise ValueError("No bibliography section found using keywords: " + ", ".join(keywords))
    return text[last_index:]


# --- Step 2: Split the bibliography text into individual references ---
class ReferenceExtraction(BaseModel):
    title: str
    author: str
    DOI: str
    URL: str
    year: int
    type: str
    bib: str

class ReferenceStatus(Enum):
    VALIDATED = "validated"
    INVALID = "invalid"
    NOT_FOUND = "not_found"

class ReferenceCheckResult(BaseModel):
    status: ReferenceStatus
    explanation: str

class ReferenceReplacement(BaseModel):
    """Represents a suggested replacement for an invalid reference."""
    title: str
    author: str = ""
    year: str = ""
    doi: str = ""
    url: str = ""
    source: str = ""  # Where this replacement was found (e.g., "Google Scholar", "Crossref")
    confidence: float = 0.5  # Confidence score 0-1
    bib: str = ""  # Formatted bibliography entry

def split_references_fallback(bib_text):
    """
    Fallback method to split references using simple rule-based parsing.
    This is used when Google AI is not available.
    """
    import re
    
    # Split by common reference patterns
    # Look for patterns like [1], (1), 1., etc.
    reference_patterns = [
        r'\[\d+\]',  # [1], [2], etc.
        r'\(\d+\)',  # (1), (2), etc.
        r'^\d+\.',   # 1., 2., etc. at start of line
        r'^\d+\s',   # 1 , 2 , etc. at start of line
    ]
    
    # Try to split by the most common pattern first
    references = []
    lines = bib_text.split('\n')
    current_ref = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line starts a new reference
        is_new_ref = False
        for pattern in reference_patterns:
            if re.match(pattern, line):
                is_new_ref = True
                break
        
        if is_new_ref and current_ref:
            # Process the previous reference
            ref_text = ' '.join(current_ref).strip()
            if ref_text:
                ref = parse_single_reference_fallback(ref_text)
                if ref:
                    references.append(ref)
            current_ref = [line]
        else:
            current_ref.append(line)
    
    # Process the last reference
    if current_ref:
        ref_text = ' '.join(current_ref).strip()
        if ref_text:
            ref = parse_single_reference_fallback(ref_text)
            if ref:
                references.append(ref)
    
    return references

def parse_single_reference_fallback(ref_text):
    """
    Parse a single reference text using simple rules.
    """
    import re
    
    # Remove reference numbers at the beginning
    ref_text = re.sub(r'^[\[\(]?\d+[\]\)]?\s*', '', ref_text).strip()
    
    # Try to extract year (4-digit number)
    year_match = re.search(r'\b(19|20)\d{2}\b', ref_text)
    year = int(year_match.group()) if year_match else 2020
    
    # Try to extract author (usually first word or first few words before year)
    author = "Unknown"
    if year_match:
        before_year = ref_text[:year_match.start()].strip()
        # Look for common author patterns
        author_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # FirstName LastName
            r'^([A-Z][a-z]+)',  # Single name
        ]
        for pattern in author_patterns:
            match = re.match(pattern, before_year)
            if match:
                author = match.group(1).split()[-1]  # Take last name
                break
    
    # Extract title (look for text between author and year)
    title = ref_text
    if year_match:
        # Look for title between author and year
        before_year = ref_text[:year_match.start()].strip()
        
        # Try to find title after author name
        if author != "Unknown" and author in before_year:
            # Find the position after the author name
            author_pos = before_year.find(author)
            if author_pos != -1:
                after_author = before_year[author_pos + len(author):].strip()
                # Remove common prefixes and clean up
                after_author = re.sub(r'^[.,\s]+', '', after_author)  # Remove leading punctuation
                if len(after_author) > 10:
                    title = after_author
        else:
            # If no clear author, try to extract title from the beginning
            # Look for patterns like "Title. Journal" or "Title, Journal"
            title_match = re.match(r'^([^.]{10,}?)(?:\.|,)\s*(?:[A-Z][a-z]+|Journal|Proc|Proceedings)', before_year)
            if title_match:
                title = title_match.group(1).strip()
            elif len(before_year) > 10:
                title = before_year
    
    # Clean up title
    title = re.sub(r'^[.,\s]+', '', title)  # Remove leading punctuation
    title = re.sub(r'[.,\s]+$', '', title)  # Remove trailing punctuation
    if len(title) > 200:
        title = title[:200] + "..."
    
    # Determine type based on content
    ref_lower = ref_text.lower()
    if any(word in ref_lower for word in ['journal', 'j.', 'vol.', 'volume']):
        ref_type = "journal_article"
    elif any(word in ref_lower for word in ['conference', 'proc.', 'proceedings']):
        ref_type = "conference_paper"
    elif any(word in ref_lower for word in ['arxiv', 'preprint']):
        ref_type = "preprint"
    elif any(word in ref_lower for word in ['book', 'chapter']):
        ref_type = "book"
    else:
        ref_type = "journal_article"  # Default
    
    # Extract DOI if present
    doi_match = re.search(r'10\.\d+/[^\s]+', ref_text)
    doi = doi_match.group() if doi_match else ""
    
    # Extract URL if present
    url_match = re.search(r'https?://[^\s]+', ref_text)
    url = url_match.group() if url_match else ""
    
    try:
        return ReferenceExtraction(
            title=title,
            author=author,
            DOI=doi,
            URL=url,
            year=year,
            type=ref_type,
            bib=ref_text
        )
    except Exception as e:
        logging.warning(f"Failed to parse reference: {e}")
        return None

def split_references(bib_text):
    """Splits the bibliography text into individual references using the Google Gemini API."""

    prompt = """
    Process a reference list extracted from a PDF, where formatting may be corrupted.  
    For each reference, extract these fields:
    - title (full title case)
    - author: first author's family name (or organization name if applicable)
    - DOI (include if explicitly stated; otherwise leave blank)
    - URL (include if explicitly stated; otherwise leave blank)
    - year (4-digit publication year)
    - type (journal_article, preprint, conference_paper, book, book_chapter, OR non_academic_website)
    - bib: normalized bibliography entry (one line)

    Return the results as JSON array with keys: title, author, DOI, URL, year, type, bib
    """

    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt + bib_text,
        config={
            "response_mime_type": "application/json",
            "response_schema": list[ReferenceExtraction],  # 👈 SDK maps into Pydantic
            "temperature": 0,
            "thinking_config": ThinkingConfig(thinking_budget=0),
        },
    )

    # Structured output (auto-parsed into ReferenceExtraction models)
    references: list[ReferenceExtraction] = response.parsed or []
    return references


# --- Step 3: Verify each reference using crossref and compare title ---
def normalize_title(title: str) -> str:
    """Normalizes a title for comparison (case-insensitive, no punctuation, etc.)."""
    title = unidecode(title)  # Remove accents
    title = re.sub(r'[^\w\s]', '', title).lower()  # Remove punctuation
    title = re.sub(r'\band\b|\bthe\b', '', title)  # Remove 'and' and 'the'
    title = re.sub(r'\s+', '', title).strip()  # Remove extra whitespace
    return title


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_scholarly(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title using scholarly, with error handling and retries."""
    try:
        search_results = scholarly.search_pubs(ref.title)
        result = next(search_results, None)  # Safely get the first result, or None
        normalized_input_title = normalize_title(ref.title)

        # Check if the first author's family name and title match
        if result and 'bib' in result and 'author' in result['bib'] and 'title' in result['bib']:
            if result['bib']['author'][0].split()[-1] == ref.author:
                normalized_item_title = normalize_title(result['bib']['title'])
                if normalized_item_title == normalized_input_title:
                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author and title match Google Scholar (exact match).")
                if normalized_input_title in normalized_item_title or normalized_item_title in normalized_input_title:
                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author and title match Google Scholar (partial match).")
                if fuzz.ratio(normalized_item_title, normalized_input_title) > 85:
                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author and title match Google Scholar (fuzzy match).")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in Google Scholar.")
    except Exception as e:
        logging.warning(f"Scholarly search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Google Scholar search failed: {e}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_crossref(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title using the Crossref API, with retries and more robust matching. Returns ReferenceCheckResult."""
    params = {'query.title': ref.title, 'rows': 5}  # Increased rows
    response = requests.get("https://api.crossref.org/works", params=params)

    if response.status_code == 200:
        items = response.json().get('message', {}).get('items', [])
        normalized_input_title = normalize_title(ref.title)
        for item in items:
            # If DOI is provided in both reference and item, compare DOI first
            ref_doi = ref.DOI.strip().lower() if ref.DOI else ''
            item_doi = item.get('DOI', '').strip().lower() if 'DOI' in item else ''
            if ref_doi and item_doi:
                if ref_doi == item_doi:
                    if 'author' in item and item['author'] and 'family' in item['author'][0] and ref.author == item['author'][0]['family']:
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author, title and DOI match Crossref record.")
                    elif 'author' in item and item['author'] and 'family' in item['author'][0] and ref.author != item['author'][0]['family']:
                        return ReferenceCheckResult(status=ReferenceStatus.INVALID, explanation="Author does not match Crossref record.")
                else:
                    return ReferenceCheckResult(status=ReferenceStatus.INVALID, explanation="DOI does not match Crossref record.")
            # Check if the first author's family name matches
            if 'author' in item and item['author'] and 'family' in item['author'][0]:
                if ref.author == item['author'][0]['family']:
                    # Check if the title matches
                    if 'title' in item and item['title']:
                        item_title = item['title'][0]
                        normalized_item_title = normalize_title(item_title)
                        if normalized_item_title == normalized_input_title:
                            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author and title match Crossref record (exact match).")
                        if normalized_input_title in normalized_item_title or normalized_item_title in normalized_input_title:
                            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author and title match Crossref record (partial match).")
                        if fuzz.ratio(normalized_item_title, normalized_input_title) > 85:
                            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author and title match Crossref record (fuzzy match).")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in Crossref.")
    else:
        logging.warning(f"Crossref API request failed with status code: {response.status_code}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Crossref API request failed with status code: {response.status_code}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_arxiv(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title in arXiv, with error handling and retries."""
    try:
        # arXiv API endpoint
        url = "http://export.arxiv.org/api/query"
        
        # Search for the title - use double quotes around the title for exact match
        params = {
            'search_query': f'ti:"{ref.title}"',
            'max_results': 5
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            # Parse the XML response - use 'lxml' parser for better compatibility
            soup = BeautifulSoup(response.content, 'lxml-xml')
            entries = soup.find_all('entry')
            
            if not entries:
                # Try a more flexible search if no exact matches
                params['search_query'] = f'all:{ref.title}'
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'lxml-xml')
                    entries = soup.find_all('entry')
                
            if not entries:
                return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in arXiv.")
                
            normalized_input_title = normalize_title(ref.title)
            
            for entry in entries:
                title_tag = entry.find('title')
                if title_tag:
                    arxiv_title = title_tag.text.strip()
                    normalized_arxiv_title = normalize_title(arxiv_title)
                    
                    # More flexible title matching
                    if normalized_arxiv_title == normalized_input_title:
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Title match in arXiv (exact match).")
                    if normalized_input_title in normalized_arxiv_title or normalized_arxiv_title in normalized_input_title:
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Title match in arXiv (partial match).")
                    if fuzz.ratio(normalized_arxiv_title, normalized_input_title) > 85:
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Title match in arXiv (fuzzy match).")
                        
                    # Check authors if titles are somewhat similar
                    if fuzz.ratio(normalized_arxiv_title, normalized_input_title) > 70:
                        author_tags = entry.find_all('author')
                        for author_tag in author_tags:
                            name_tag = author_tag.find('name')
                            if name_tag:
                                author_name = name_tag.text.strip()
                                # Extract last name
                                last_name = author_name.split()[-1]
                                if last_name.lower() == ref.author.lower():
                                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author and similar title match in arXiv.")
                                
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in arXiv.")
        
    except Exception as e:
        logging.warning(f"arXiv search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"arXiv search failed: {e}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_workshop_paper(ref: ReferenceExtraction) -> ReferenceCheckResult:
    try:
        if not any(w in ref.bib.lower() for w in ["workshop", "symposium", "proc.", "proceedings"]):
            return ReferenceCheckResult(ReferenceStatus.NOT_FOUND, "Not a workshop paper.")

        if not GENAI_AVAILABLE or not GOOGLE_API_KEY:
            return ReferenceCheckResult(ReferenceStatus.NOT_FOUND, "Google AI not available for workshop search.")

        prompt = f"""
        Verify if this exact workshop paper exists. Return "True" if a site shows the same title and author; else "False".
        Only output True or False.

        Title: {ref.title}
        Author: {ref.author}
        Year: {ref.year}
        """

        client = genai.Client(api_key=GOOGLE_API_KEY)
        google_search_tool = Tool(google_search=GoogleSearch())
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"tools": [google_search_tool], "temperature": 0},
        )

        answer = resp.candidates[0].content.parts[0].text.strip().lower()
        return (
            ReferenceCheckResult(ReferenceStatus.VALIDATED, "Workshop paper found via Google search.")
            if "true" in answer else
            ReferenceCheckResult(ReferenceStatus.NOT_FOUND, "Workshop paper not found via Google search.")
        )

    except Exception as e:
        logging.warning(f"Workshop paper search failed for '{ref.title}': {e}")
        return ReferenceCheckResult(ReferenceStatus.NOT_FOUND, f"Workshop paper search failed: {e}")



@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_google(ref: ReferenceExtraction) -> ReferenceCheckResult:
    if not GENAI_AVAILABLE or not GOOGLE_API_KEY:
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Google AI not available for search.")

    prompt = f"""
    Search for this reference. Return "True" if you find a page with the exact title and author; otherwise "False".
    Only output True or False.

    Author: {ref.author}
    Title: {ref.title}
    """

    client = genai.Client(api_key=GOOGLE_API_KEY)
    google_search_tool = Tool(google_search=GoogleSearch())
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={"tools": [google_search_tool], "temperature": 0},
    )

    answer = resp.candidates[0].content.parts[0].text.strip().lower()
    return (
        ReferenceCheckResult(ReferenceStatus.VALIDATED, "Google search found matching reference.")
        if "true" in answer else
        ReferenceCheckResult(ReferenceStatus.NOT_FOUND, "Google search did not find matching reference.")
    )


def verify_url(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """
    Verifies if the title on the webpage at the given URL matches the reference title.
    """
    if not ref.URL:
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No URL provided.")

    try:
        response = requests.get(ref.URL, timeout=5)  # Set a timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find('title')

        if title_tag:
            webpage_title = title_tag.text.strip()
            normalized_webpage_title = normalize_title(webpage_title)
            normalized_input_title = normalize_title(ref.title)

            if normalized_webpage_title == normalized_input_title:
                return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Webpage title matches reference title (exact match).")
            elif normalized_input_title in normalized_webpage_title or normalized_webpage_title in normalized_input_title:  #robust matching
                return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Webpage title matches reference title (partial match).")
        else:
            logging.warning(f"No <title> tag found at URL: {ref.URL}")
            return search_title_google(ref)

    except requests.exceptions.RequestException as e:
        logging.warning(f"Error accessing URL {ref.URL}: {e}")
        return search_title_google(ref)  # Or consider raising the exception if you want to halt execution on URL errors.
    except Exception as e:
        logging.warning(f"Error processing URL {ref.URL}: {e}")
        return search_title_google(ref)

def search_title(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title using multiple methods."""
    if ref.type == "non_academic_website":
        return verify_url(ref)
    else:
        # First try Crossref
        crossref_result = search_title_crossref(ref)
        if crossref_result.status == ReferenceStatus.INVALID:
            return crossref_result
        if crossref_result.status == ReferenceStatus.VALIDATED:
            return crossref_result
        # For all academic papers, try arXiv as a fallback
        arxiv_result = search_title_arxiv(ref)
        if arxiv_result.status == ReferenceStatus.VALIDATED:
            return arxiv_result
        # Special check for workshop papers
        workshop_result = search_title_workshop_paper(ref)
        if workshop_result.status == ReferenceStatus.VALIDATED:
            return workshop_result
        # Fall back to Google Scholar
        scholar_result = search_title_scholarly(ref)
        if scholar_result.status == ReferenceStatus.VALIDATED:
            return scholar_result
        # If all fail, return the most informative NOT_FOUND
        for result in [crossref_result, arxiv_result, workshop_result, scholar_result]:
            if result.status == ReferenceStatus.NOT_FOUND:
                return result
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No evidence found in any source.")




def extract_topic_from_ref(invalid_ref: ReferenceExtraction) -> str:
    """
    Try to summarize the main topic from the invalid reference.
    Uses Gemini when available; falls back to a simple keyword heuristic.
    """
    title = (invalid_ref.title or "").strip()
    if GENAI_AVAILABLE and GOOGLE_API_KEY and title:
        try:
            client = genai.Client(api_key=GOOGLE_API_KEY)
            topic_prompt = f"""
            Summarize the research topic of this (possibly fake) reference in a short phrase (<=8 words):
            {invalid_ref.bib or title}
            """
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=topic_prompt,
                config={"temperature": 0},
            )
            topic = (resp.text or "").strip()
            return topic if topic else title
        except Exception:
            pass

    # Fallback: simple keyword extraction from title
    # Keep alphanumerics, drop stopwords, take first ~6 tokens
    words = re.findall(r"[A-Za-z0-9\-]+", title.lower())
    stop = {"the", "and", "for", "with", "of", "in", "on", "to", "a", "an", "by", "from"}
    keywords = [w for w in words if w not in stop]
    return " ".join(keywords[:6]) if keywords else title


def crossref_search_by_topic(topic: str, max_results: int = 3) -> List[ReferenceReplacement]:
    """
    Search Crossref for the topic; return ReferenceReplacement list.
    """
    results: List[ReferenceReplacement] = []
    try:
        params = {
            "query.bibliographic": topic,
            "rows": max_results * 3,   # overfetch, we’ll dedupe & trim later
        }
        r = requests.get("https://api.crossref.org/works", params=params, timeout=10)
        if r.status_code != 200:
            return results
        items = r.json().get("message", {}).get("items", [])
        for it in items:
            title = (it.get("title", [""]) or [""])[0].strip()
            if not title:
                continue
            authors = it.get("author", [])
            family = ""
            if authors:
                family = (authors[0].get("family") or authors[0].get("name") or "").strip()
            year = ""
            try:
                year = str(it.get("issued", {}).get("date-parts", [[None]])[0][0] or "")
            except Exception:
                year = ""
            doi = (it.get("DOI") or "").strip()
            url = (it.get("URL") or "").strip()
            bib = f"{family} ({year}). {title}"
            conf = 0.8 if doi else 0.6
            results.append(
                ReferenceReplacement(
                    title=title,
                    author=family,
                    year=year,
                    doi=doi,
                    url=url,
                    source="Crossref",
                    confidence=conf,
                    bib=bib
                )
            )
            if len(results) >= max_results * 3:
                break
    except Exception:
        pass
    return results[: max_results * 3]


def arxiv_search_by_topic(topic: str, max_results: int = 3) -> List[ReferenceReplacement]:
    """
    Search arXiv by topic; return ReferenceReplacement list.
    """
    results: List[ReferenceReplacement] = []
    try:
        q = f'all:"{topic}"'
        params = {"search_query": q, "max_results": max_results * 2}
        r = requests.get("http://export.arxiv.org/api/query", params=params, timeout=10)
        if r.status_code != 200:
            return results
        soup = BeautifulSoup(r.content, "lxml-xml")
        for entry in soup.find_all("entry"):
            title_tag = entry.find("title")
            if not title_tag:
                continue
            title = title_tag.text.strip()
            year = ""
            pub = entry.find("published")
            if pub and pub.text:
                year = pub.text.strip()[:4]
            author_family = ""
            a0 = entry.find("author")
            if a0:
                nm = a0.find("name")
                if nm and nm.text:
                    author_family = nm.text.strip().split()[-1]
            url = ""
            id_tag = entry.find("id")
            if id_tag and id_tag.text:
                url = id_tag.text.strip()
            doi = ""
            doi_tag = entry.find("arxiv:doi")
            if doi_tag and doi_tag.text:
                doi = doi_tag.text.strip()
            bib = f"{author_family} ({year}). {title}"
            results.append(
                ReferenceReplacement(
                    title=title,
                    author=author_family,
                    year=year,
                    doi=doi,
                    url=url,
                    source="arXiv",
                    confidence=0.6,
                    bib=bib
                )
            )
            if len(results) >= max_results * 2:
                break
    except Exception:
        pass
    return results[: max_results * 2]


def scholarly_search_by_topic(topic: str, max_results: int = 3) -> List[ReferenceReplacement]:
    """
    Reuse your existing Google Scholar fallback by creating a faux ReferenceExtraction
    whose 'title' is the topic phrase.
    """
    fake = ReferenceExtraction(
        title=topic,
        author="",
        DOI="",
        URL="",
        year=2020,
        type="conference_paper",
        bib=topic,
    )
    try:
        return search_similar_papers_scholarly(fake, max_results)
    except Exception:
        return []


def gemini_topic_replacements(topic: str, max_results: int = 3) -> List[ReferenceReplacement]:
    """
    Ask Gemini + GoogleSearch for replacements given a topic.
    Relies on response_schema=list[ReferenceReplacement] so we don't hand-parse JSON.
    """
    results: List[ReferenceReplacement] = []
    if not (GENAI_AVAILABLE and GOOGLE_API_KEY):
        return results
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        google_search_tool = Tool(google_search=GoogleSearch())
        prompt = f"""
        Suggest up to {max_results} real academic works about: "{topic}".
        Output strictly as JSON matching this schema: title, author, year, doi, url, source, confidence, bib.
        If metadata is missing, use an empty string for the field. Never return an empty list.
        """
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "tools": [google_search_tool],
                "response_mime_type": "application/json",
                "response_schema": list[ReferenceReplacement],
                "temperature": 0.6,
            },
        )
        results = resp.parsed or []
    except Exception:
        return []
    return results


def _merge_unique_by_title(candidates: List[ReferenceReplacement], limit: int) -> List[ReferenceReplacement]:
    """
    Deduplicate by normalized title; keep highest-confidence version.
    """
    best: dict[str, ReferenceReplacement] = {}
    for c in candidates:
        key = normalize_title(c.title or "")
        if not key:
            continue
        if key not in best or (c.confidence or 0) > (best[key].confidence or 0):
            best[key] = c
    uniq = list(best.values())
    uniq.sort(key=lambda x: x.confidence, reverse=True)
    return uniq[:limit]


def find_reference_replacements(invalid_ref: ReferenceExtraction, max_suggestions: int = 3) -> List[ReferenceReplacement]:
    """
    Optimal pipeline for replacement suggestions:
      0) Detect topic from the invalid reference
      1) Title-nearby (Google Scholar by the *invalid title*)
      2) Crossref by topic (DOI-rich, trustworthy)
      3) arXiv by topic (great recall for ML/CS/physics)
      4) Scholar by topic (broad recall)
      5) Gemini + GoogleSearch as last resort (strict JSON)

    Then:
      - Deduplicate (prefer DOI; else normalized title)
      - Score candidates and boost grounded sources/DOIs/near-title matches
      - Return top-N with updated confidence
    """

    # ---------------------------
    # Scoring helpers (local)
    # ---------------------------
    def _year_num(y: str | int) -> int | None:
        try:
            return int(str(y)[:4])
        except Exception:
            return None

    def _title_sim(a: str, b: str) -> float:
        # 0..1 fuzzy similarity between titles
        return fuzz.ratio(normalize_title(a), normalize_title(b)) / 100.0 if a and b else 0.0

    def _topic_overlap(topic: str, title: str) -> float:
        # fraction of topic tokens present in title (very simple)
        toks = [t for t in re.findall(r"[A-Za-z0-9\-]+", topic.lower()) if len(t) > 2]
        if not toks:
            return 0.0
        title_norm = normalize_title(title)
        hits = sum(1 for t in toks if t in title_norm)
        return hits / max(1, len(toks))

    def _source_weight(src: str) -> float:
        src = (src or "").lower()
        if "crossref" in src: return 1.0
        if "arxiv" in src: return 0.85
        if "scholar" in src: return 0.75
        if "publisher" in src: return 0.9
        if "ai search" in src: return 0.55
        return 0.6

    def _has_doi(doi: str) -> float:
        return 1.0 if (doi or "").strip() else 0.0

    def _year_proximity(inv_year: int | None, cand_year: str) -> float:
        if inv_year is None:
            return 0.0
        cy = _year_num(cand_year)
        if cy is None:
            return 0.0
        diff = abs(inv_year - cy)
        if diff <= 2: return 1.0
        if diff <= 5: return 0.6
        if diff <= 10: return 0.3
        return 0.0

    def _score(c: ReferenceReplacement, topic: str, invalid: ReferenceExtraction) -> float:
        # Weighted blend; adjust weights as needed
        sim = _title_sim(invalid.title, c.title)
        ovl = _topic_overlap(topic, c.title)
        yr = _year_proximity(invalid.year if isinstance(invalid.year, int) else _year_num(str(invalid.year)), c.year)
        doi_w = _has_doi(c.doi)
        src_w = _source_weight(c.source)

        # Final score in 0..1
        score = (0.40 * sim) + (0.20 * ovl) + (0.20 * doi_w) + (0.10 * src_w) + (0.10 * yr)
        # Never *lower* an existing model-provided confidence; keep the max
        return max(c.confidence or 0.0, round(min(1.0, score), 3))

    # ---------------------------
    # 0) Detect topic
    # ---------------------------
    try:
        topic = extract_topic_from_ref(invalid_ref)
    except Exception:
        topic = invalid_ref.title or ""

    # ---------------------------
    # 1..5) Gather candidates
    # ---------------------------
    candidates: List[ReferenceReplacement] = []

    # 1) Near-title (helps when fake ref is “close” to a real one)
    try:
        candidates += search_similar_papers_scholarly(invalid_ref, max_results=max_suggestions * 2)
    except Exception as e:
        logging.warning(f"Title-nearby (Scholar) failed: {e}")

    # 2) Crossref by topic
    try:
        candidates += crossref_search_by_topic(topic, max_results=max_suggestions * 2)
    except Exception as e:
        logging.warning(f"Crossref-by-topic failed: {e}")

    # 3) arXiv by topic
    try:
        candidates += arxiv_search_by_topic(topic, max_results=max_suggestions * 2)
    except Exception as e:
        logging.warning(f"arXiv-by-topic failed: {e}")

    # 4) Scholar by topic
    try:
        candidates += scholarly_search_by_topic(topic, max_results=max_suggestions * 2)
    except Exception as e:
        logging.warning(f"Scholar-by-topic failed: {e}")

    # 5) Gemini last
    try:
        candidates += gemini_topic_replacements(topic, max_results=max_suggestions * 2)
    except Exception as e:
        logging.warning(f"Gemini-topic replacements failed: {e}")

    # If everything failed, return empty (no fake “fallbacks”)
    if not candidates:
        return []

    # ---------------------------
    # Deduplicate (prefer DOI)
    # ---------------------------
    by_doi: dict[str, ReferenceReplacement] = {}
    for c in candidates:
        if c.doi:
            key = c.doi.strip().lower()
            prev = by_doi.get(key)
            if not prev or (c.confidence or 0) > (prev.confidence or 0):
                by_doi[key] = c

    # Collect non-DOI candidates and de-dup by normalized title
    rest: List[ReferenceReplacement] = [c for c in candidates if not (c.doi or "").strip()]
    by_title: dict[str, ReferenceReplacement] = {}
    for c in rest:
        key = normalize_title(c.title or "")
        if not key:
            continue
        prev = by_title.get(key)
        if not prev or (c.confidence or 0) > (prev.confidence or 0):
            by_title[key] = c

    deduped = list(by_doi.values()) + list(by_title.values())

    # ---------------------------
    # Score + normalize + sort
    # ---------------------------
    for c in deduped:
        # Normalize missing fields
        c.author = c.author or ""
        c.year = c.year or ""
        c.doi = c.doi or ""
        c.url = c.url or ""
        c.source = c.source or "AI Search"
        c.bib = c.bib or f"{c.author} ({c.year}). {c.title}"

        # Update confidence using scoring
        c.confidence = _score(c, topic, invalid_ref)

    deduped.sort(key=lambda x: x.confidence, reverse=True)

    return deduped[:max_suggestions]




def create_fallback_replacement(invalid_ref: ReferenceExtraction) -> List[ReferenceReplacement]:
    """
    Create a basic fallback replacement when all other methods fail.
    """
    logging.info("Creating fallback replacement suggestion")
    
    # Extract keywords from the title for a basic suggestion
    title_words = invalid_ref.title.split()[:3]  # First 3 words
    suggested_title = f"Research on {' '.join(title_words)}"
    
    replacement = ReferenceReplacement(
        title=suggested_title,
        author=invalid_ref.author,
        year=str(invalid_ref.year),
        source="Fallback Suggestion",
        confidence=0.3,
        bib=f"{invalid_ref.author} ({invalid_ref.year}). {suggested_title}. [Suggested replacement - please verify]"
    )
    
    return [replacement]

def search_similar_papers_scholarly(invalid_ref: ReferenceExtraction, max_results: int = 3) -> List[ReferenceReplacement]:
    """
    Search for similar papers using Google Scholar as a fallback method.
    """
    try:
        # Try multiple search strategies
        search_queries = [
            invalid_ref.title,  # Original title
            invalid_ref.title.split()[0] + " " + invalid_ref.title.split()[-1] if len(invalid_ref.title.split()) > 1 else invalid_ref.title,  # First and last words
            " ".join(invalid_ref.title.split()[:3]) if len(invalid_ref.title.split()) >= 3 else invalid_ref.title,  # First 3 words
        ]
        
        replacements = []
        seen_titles = set()
        
        for search_query in search_queries:
            if len(replacements) >= max_results:
                break
                
            try:
                search_results = scholarly.search_pubs(search_query)
                
                for result in search_results:
                    if len(replacements) >= max_results:
                        break
                        
                    if 'bib' in result and 'title' in result['bib'] and 'author' in result['bib']:
                        title = result['bib']['title']
                        
                        # Skip if we've already seen this title
                        if title.lower() in seen_titles:
                            continue
                        seen_titles.add(title.lower())
                        
                        # Calculate confidence score
                        title_similarity = fuzz.ratio(
                            normalize_title(invalid_ref.title),
                            normalize_title(title)
                        ) / 100.0
                        
                        # More lenient threshold for broader search
                        if title_similarity > 0.2 or len(replacements) == 0:  # Always include at least one result
                            author_name = ""
                            if result['bib']['author']:
                                author_parts = result['bib']['author'][0].split()
                                author_name = author_parts[-1] if author_parts else ""
                            
                            # Ensure year is converted to string
                            year_value = result['bib'].get('pub_year', result['bib'].get('year', ''))
                            year_str = str(year_value) if year_value else ''
                            
                            replacement = ReferenceReplacement(
                                title=title,
                                author=author_name,
                                year=year_str,
                                doi=result['bib'].get('doi', ''),
                                url=result.get('url', ''),
                                source="Google Scholar",
                                confidence=max(title_similarity, 0.3),  # Minimum confidence of 0.3
                                bib=result['bib'].get('bibtex', f"{author_name} ({year_str}). {title}")
                            )
                            replacements.append(replacement)
                            
            except Exception as e:
                logging.warning(f"Search query '{search_query}' failed: {e}")
                continue
        
        # Sort by confidence and return top results
        replacements.sort(key=lambda x: x.confidence, reverse=True)
        return replacements[:max_results]
        
    except Exception as e:
        logging.warning(f"Scholarly search for replacements failed: {e}")
        return []

# --- Main Workflow ---

def veriexcite(pdf_path: str) -> Tuple[int, int, List[str], List[str]]:
    """
    Check references in a PDF. Returns:
    - count_verified: number of validated references
    - count_warning: number of warnings (invalid or not found)
    - list_warning: list of bib entries with warnings
    - list_explanations: list of explanations for each reference
    """
    # 1. Extract text from PDF and find bibliography
    full_text = extract_text_from_pdf(pdf_path)
    bib_text = extract_bibliography_section(full_text)
    # print("Extracted Bibliography Section:\n", bib_text, "\n")

    # 2. Split into individual references
    references = split_references(bib_text)
    # print(f"Found {len(references)} references.")

    # 3. Verify each reference
    count_verified, count_warning = 0, 0
    list_warning = []
    list_explanations = []

    for idx, ref in enumerate(references):
        result = search_title(ref)
        list_explanations.append(f"Reference: {ref.bib}\nStatus: {result.status.value}\nExplanation: {result.explanation}\n")
        if result.status == ReferenceStatus.VALIDATED:
            count_verified += 1
        else:
            count_warning += 1
            list_warning.append(ref.bib)
    return count_verified, count_warning, list_warning, list_explanations

def process_pdf_file(pdf_path: str) -> None:
    """Check a single PDF file."""
    count_verified, count_warning, list_warning, list_explanations = veriexcite(pdf_path)
    print(f"{count_verified} references verified, {count_warning} warnings.")
    if count_warning > 0:
        print("\nWarning List:\n")
        for item in list_warning:
            print(item)
    print("\nExplanation:\n")
    for explanation in list_explanations:
        print(explanation)
    return count_verified, count_warning, list_warning, list_explanations

def process_folder(folder_path: str) -> None:
    """Check all PDF files in a folder."""
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    pdf_files.sort()
    print(f"Found {len(pdf_files)} PDF files in the folder.")

    results = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Checking file: {pdf_file}")
        count_verified, count_warning, list_warning, list_explanations = process_pdf_file(pdf_path)
        print("--------------------------------------------------")
        results.append({"File": pdf_file, "Found References": count_verified + count_warning, "Verified": count_verified,
                        "Warnings": count_warning, "Warning List": list_warning, "Explanation": list_explanations})
        pd.DataFrame(results).to_csv('VeriExCite results.csv', index=False)
    print("Results saved to VeriExCite results.csv")


if __name__ == "__main__":
    ''' Set your Google Gemini API key here '''
    # Apply for a key at https://ai.google.dev/aistudio with 1500 requests per day for FREE
    GOOGLE_API_KEY = "YOUR_API_KEY"
    set_google_api_key(GOOGLE_API_KEY)

    ''' Example usage #1: check a single PDF file '''
    # pdf_path = "path/to/your/paper.pdf"
    # process_pdf_file(pdf_path)

    ''' Example usage #2: check all PDF files in a folder '''
    # Please replace the folder path to your directory containing the PDF files.
    folder_path = "path/to/your/folder"
    process_folder(folder_path)