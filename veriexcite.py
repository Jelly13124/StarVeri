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
import google.generativeai as genai
from google.generativeai.types import Tool
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
    # French
    "Références", "Bibliographie",
    # German
    "Literaturverzeichnis", "Quellenverzeichnis",
    # Spanish
    "Referencias", "Bibliografía",
    # Russian
    "Список литературы",
    # Italian
    "Riferimenti", "Bibliografia",
    # Portuguese
    "Referências", "Bibliografia",
    # Korean
    "참고문헌"
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
    author: str
    year: str
    doi: str = ""
    url: str = ""
    source: str = ""  # Where this replacement was found (e.g., "Google Scholar", "Crossref")
    confidence: float = 0.0  # Confidence score 0-1
    bib: str = ""  # Formatted bibliography entry

def split_references(bib_text):
    """Splits the bibliography text into individual references using the Google Gemini API."""

    prompt = """
    Process a reference list extracted from a PDF, where formatting may be corrupted.  
    Follow these steps to clean and extract key information: 
    1. Normalisation: Fix spacing errors, line breaks, and punctuation.
    2. Extraction: For each reference, extract:
    - title (full title case)
    - author: First author's family name (If the author is an organization, use the organization name)
    - DOI (include if explicitly stated; otherwise leave blank)
    - URL (include if explicitly stated; otherwise leave blank)
    - year (4-digit publication year)
    - type (journal_article, preprint, conference_paper, book, book_chapter, OR non_academic_website. If the author is not a human but an organization, select non_academic_website)
    - bib: Normalised input bibliography (correct format, in one line)
    
    Return the results as a JSON array with lowercase field names: title, author, DOI, URL, year, type, bib\n\n
    """

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(
        prompt + bib_text,
        generation_config=genai.GenerationConfig(
            response_mime_type='application/json',
            temperature=0,
        )
    )

    # Parse the JSON response
    import json
    try:
        references_data = json.loads(response.text)
        
        # Normalize field names to lowercase
        normalized_refs = []
        for ref in references_data:
            normalized_ref = {}
            for key, value in ref.items():
                # Map common capitalized field names to lowercase
                field_mapping = {
                    'Title': 'title',
                    'Author': 'author', 
                    'DOI': 'DOI',
                    'URL': 'URL',
                    'Year': 'year',
                    'Type': 'type',
                    'Bib': 'bib'
                }
                normalized_key = field_mapping.get(key, key.lower())
                normalized_ref[normalized_key] = value
            normalized_refs.append(normalized_ref)
        
        references = [ReferenceExtraction(**ref) for ref in normalized_refs]
        return references
    except (json.JSONDecodeError, TypeError) as e:
        logging.error(f"Failed to parse references: {e}")
        return []


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
    """Searches for workshop papers using Google Search directly."""
    try:
        # Check if it's likely a workshop paper from the reference text
        workshop_indicators = ['workshop', 'symposium', 'proc.', 'proceedings']
        is_likely_workshop = any(indicator in ref.bib.lower() for indicator in workshop_indicators)
        
        if not is_likely_workshop:
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Not a workshop paper.")
            
        # Use Google search through the Google Gemini API with more specific prompt
        prompt = f"""
        Please search for this exact workshop paper and verify it exists:
        Title: {ref.title}
        Author: {ref.author}
        Year: {ref.year}
        
        This paper appears to be from a workshop or symposium. Check conferences, workshops, 
        and personal/university pages. Return 'True' only if you can find evidence this 
        specific workshop paper exists (exact title and author match). Return 'False' otherwise.
        Return only 'True' or 'False', without any additional explanation.
        """

        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0,
            )
        )

        answer = normalize_title(response.text)
        if answer.startswith('true') or answer.endswith('true'):
            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Workshop paper found via Google search.")
        else:
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Workshop paper not found via Google search.")
            
    except Exception as e:
        logging.warning(f"Workshop paper search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Workshop paper search failed: {e}")

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


def search_title_google(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title using Google Search and match using a LLM model."""

    prompt = f"""
    Please search for the reference on Google, compare with research results, and determine if it is genuine.\n
    Return 'True' only if a website with the the exact title and author is found. Otherwise, return 'False'.\n
    Return only 'True' or 'False', without any additional information.\n\n
    Author: {ref.author}\n
    Title: {ref.title}\n"""

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(
        prompt,
    )

    answer = normalize_title(response.text)
    if answer.startswith('true') or answer.endswith('true'):
        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Google search found matching reference.")
    else:
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Google search did not find matching reference.")

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

def find_reference_replacements(invalid_ref: ReferenceExtraction, max_suggestions: int = 3) -> List[ReferenceReplacement]:
    """
    Find real reference alternatives for an invalid reference using AI.
    This function searches for similar papers and suggests legitimate replacements.
    """
    if not GOOGLE_API_KEY:
        logging.warning("No Google API key available for replacement search")
        return []
    
    logging.info(f"Searching for replacements for: {invalid_ref.title}")
    
    # First try the fallback method using Google Scholar
    try:
        logging.info("Trying Google Scholar search...")
        scholarly_replacements = search_similar_papers_scholarly(invalid_ref, max_suggestions)
        if scholarly_replacements:
            logging.info(f"Found {len(scholarly_replacements)} replacements from Google Scholar")
            return scholarly_replacements
        else:
            logging.info("No replacements found from Google Scholar")
    except Exception as e:
        logging.warning(f"Scholarly fallback failed: {e}")
    
    # If scholarly search fails, try AI-powered search
    try:
        logging.info("Trying AI-powered search...")
        # Create a simpler, more focused prompt
        prompt = f"""
        Find {max_suggestions} real academic papers similar to this invalid reference:
        
        Title: {invalid_ref.title}
        Author: {invalid_ref.author}
        Year: {invalid_ref.year}
        
        Search for legitimate academic papers with similar topics. Return only the results in this exact JSON format:
        {{
            "replacements": [
                {{
                    "title": "Paper Title Here",
                    "author": "Author Last Name",
                    "year": "2023",
                    "doi": "10.1000/example",
                    "source": "Google Scholar",
                    "confidence": 0.8
                }}
            ]
        }}
        """
        
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type='application/json',
                temperature=0.1,  # Very focused
            )
        )
        
        logging.info(f"AI response received: {response.text[:200]}...")
        
        # Parse the response
        import json
        try:
            result = json.loads(response.text)
            replacements = []
            
            for item in result.get('replacements', []):
                if item.get('title') and item.get('author'):  # Only add if we have essential info
                    replacement = ReferenceReplacement(
                        title=item.get('title', ''),
                        author=item.get('author', ''),
                        year=str(item.get('year', '')),
                        doi=item.get('doi', ''),
                        url=item.get('url', ''),
                        source=item.get('source', 'AI Search'),
                        confidence=float(item.get('confidence', 0.5)),
                        bib=f"{item.get('author', '')} ({item.get('year', '')}). {item.get('title', '')}"
                    )
                    replacements.append(replacement)
            
            # Sort by confidence score (highest first)
            replacements.sort(key=lambda x: x.confidence, reverse=True)
            logging.info(f"Found {len(replacements)} replacements from AI search")
            return replacements[:max_suggestions]
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logging.warning(f"Failed to parse AI replacement suggestions: {e}")
            logging.info(f"Raw response: {response.text}")
            # Try to extract basic info from raw response
            return extract_replacements_from_text(response.text, invalid_ref)
            
    except Exception as e:
        logging.warning(f"AI search failed: {e}")
        # Return a basic fallback suggestion
        return create_fallback_replacement(invalid_ref)

def extract_replacements_from_text(text: str, invalid_ref: ReferenceExtraction) -> List[ReferenceReplacement]:
    """
    Fallback method to extract replacement info from raw AI response text.
    """
    replacements = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if 'title' in line.lower() and ':' in line:
            # Try to extract title
            title = line.split(':', 1)[1].strip().strip('"')
            if title and len(title) > 10:  # Reasonable title length
                replacement = ReferenceReplacement(
                    title=title,
                    author=invalid_ref.author,  # Use original author as fallback
                    year=str(invalid_ref.year),
                    source="AI Search (Fallback)",
                    confidence=0.6,
                    bib=f"{invalid_ref.author} ({invalid_ref.year}). {title}"
                )
                replacements.append(replacement)
                if len(replacements) >= 2:  # Limit to 2 suggestions
                    break
    
    return replacements

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
