import PyPDF2
from pydantic import BaseModel, Field
import requests
import os
import pandas as pd
import re
from unidecode import unidecode
from scholarly import scholarly
import logging
from typing import List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from google.genai.types import Tool, GoogleSearch, ThinkingConfig
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

# --- ADDED: Pydantic model for a structured replacement suggestion ---
class ReplacementSuggestion(BaseModel):
    """Defines the structure for a verified replacement suggestion."""
    found: bool = Field(description="Set to true if suitable replacements are found, otherwise false.")
    reasoning: str = Field(description="Brief reasoning for why these papers were chosen as replacements.")
    suggestion1_bib: str = Field(description="First replacement paper bibliography entry")
    suggestion1_url: str = Field(description="First replacement paper URL")
    suggestion1_score: int = Field(description="First replacement paper match score (1-100)")
    suggestion2_bib: str = Field(description="Second replacement paper bibliography entry")
    suggestion2_url: str = Field(description="Second replacement paper URL")
    suggestion2_score: int = Field(description="Second replacement paper match score (1-100)")
    suggestion3_bib: str = Field(description="Third replacement paper bibliography entry")
    suggestion3_url: str = Field(description="Third replacement paper URL")
    suggestion3_score: int = Field(description="Third replacement paper match score (1-100)")


def split_references(bib_text):
    """Splits the bibliography text into individual references using the Google Gemini API."""

    prompt = """
    Process a reference list extracted from a PDF, where formatting may be corrupted.  
    Follow these steps to clean and extract key information: 
    1. Normalisation: Fix spacing errors, line breaks, and punctuation.
    2. Extraction: For each reference, extract:
    - Title (full title case)
    - Author: First author's family name (If the author is an organization, use the organization name)
    - DOI (include if explicitly stated; otherwise leave blank)
    - URL (include if explicitly stated; otherwise leave blank)
    - Year (4-digit publication year)
    - Type (journal_article, preprint, conference_paper, book, book_chapter, OR non_academic_website. If the author is not a human but an organization, select non_academic_website)
    - Bib: Normalised input bibliography (correct format, in one line)\n\n
    """

    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt + bib_text,
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[ReferenceExtraction],
            'temperature': 0,
            'thinking_config': ThinkingConfig(thinking_budget=0),
        },
    )
    references: list[ReferenceExtraction] = response.parsed
    return references


# --- Step 3: Verify each reference using crossref and compare title ---
def normalize_title(title: str) -> str:
    """Normalizes a title for comparison (case-insensitive, no punctuation, etc.)."""
    title = unidecode(title)
    title = re.sub(r'[^\w\s]', '', title).lower()
    title = re.sub(r'\band\b|\bthe\b', '', title)
    title = re.sub(r'\s+', '', title).strip()
    return title


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_scholarly(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title using scholarly, with error handling and retries."""
    try:
        search_results = scholarly.search_pubs(ref.title)
        result = next(search_results, None)
        normalized_input_title = normalize_title(ref.title)

        if result and 'bib' in result and 'author' in result['bib'] and 'title' in result['bib']:
            # More flexible author matching
            result_author = result['bib']['author'][0].split()[-1].lower()
            ref_author = ref.author.lower()
            author_match = (
                ref_author == result_author or
                ref_author in result_author or
                result_author in ref_author or
                fuzz.ratio(ref_author, result_author) > 80
            )
            
            if author_match:
                normalized_item_title = normalize_title(result['bib']['title'])
                title_score = fuzz.ratio(normalized_item_title, normalized_input_title)
                
                if normalized_item_title == normalized_input_title:
                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author and title match Google Scholar (exact match).")
                if title_score >= 90:
                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation=f"Author and title match Google Scholar (high confidence: {title_score}%).")
            
            # Even if author doesn't match, check if title is very close (only 95%+ for title-only matches)
            normalized_item_title = normalize_title(result['bib']['title'])
            title_score = fuzz.ratio(normalized_item_title, normalized_input_title)
            if title_score >= 95:
                return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation=f"Title match Google Scholar (very high confidence: {title_score}%).")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in Google Scholar.")
    except Exception as e:
        logging.warning(f"Scholarly search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Google Scholar search failed: {e}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_crossref(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title using the Crossref API, with retries and more robust matching. Returns ReferenceCheckResult."""
    params = {'query.title': ref.title, 'rows': 10}  # Increased from 5 to 10 for more options
    response = requests.get("https://api.crossref.org/works", params=params)

    if response.status_code == 200:
        items = response.json().get('message', {}).get('items', [])
        normalized_input_title = normalize_title(ref.title)
        
        # First pass: Check DOI matches (highest confidence)
        ref_doi = ref.DOI.strip().lower() if ref.DOI else ''
        for item in items:
            item_doi = item.get('DOI', '').strip().lower() if 'DOI' in item else ''
            if ref_doi and item_doi and ref_doi == item_doi:
                if 'author' in item and item['author'] and 'family' in item['author'][0]:
                    if ref.author.lower() == item['author'][0]['family'].lower():
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author, title and DOI match Crossref record.")
                    else:
                        return ReferenceCheckResult(status=ReferenceStatus.INVALID, explanation="Author does not match Crossref record.")
                else:
                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="DOI matches Crossref record.")
        
        # Second pass: Check title and author matches with different confidence levels
        best_match_score = 0
        best_match_explanation = ""
        
        for item in items:
            if 'title' in item and item['title'] and 'author' in item and item['author']:
                item_title = item['title'][0]
                normalized_item_title = normalize_title(item_title)
                title_score = fuzz.ratio(normalized_item_title, normalized_input_title)
                
                # Check author match (more flexible)
                author_match = False
                if 'family' in item['author'][0]:
                    author_last_name = item['author'][0]['family'].lower()
                    ref_author = ref.author.lower()
                    # More flexible author matching
                    author_match = (
                        ref_author == author_last_name or  # Exact match
                        ref_author in author_last_name or  # Partial match
                        author_last_name in ref_author or  # Reverse partial match
                        fuzz.ratio(ref_author, author_last_name) > 80  # Fuzzy match
                    )
                
                # Calculate combined score
                combined_score = title_score
                if author_match:
                    combined_score += 20  # Bonus for author match
                
                # Update best match if this is better
                if combined_score > best_match_score:
                    best_match_score = combined_score
                    if title_score == 100:
                        best_match_explanation = "Author and title match Crossref record (exact match)."
                    elif title_score >= 90:
                        best_match_explanation = "Author and title match Crossref record (very high confidence)."
                    elif title_score >= 80:
                        best_match_explanation = "Author and title match Crossref record (high confidence)."
                    elif title_score >= 70:
                        best_match_explanation = "Author and title match Crossref record (good confidence)."
                    else:
                        best_match_explanation = f"Author and title match Crossref record (moderate confidence: {title_score}%)."
        
        # Return result based on best match - STRICT: Only >90% confidence or exact match
        if best_match_score >= 90:  # Very strict threshold - only high confidence matches
            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation=best_match_explanation)
        else:
            return ReferenceCheckResult(status=ReferenceStatus.INVALID, explanation=f"Reference found but confidence too low ({best_match_score}% < 90%).")
    else:
        logging.warning(f"Crossref API request failed with status code: {response.status_code}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Crossref API request failed with status code: {response.status_code}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_arxiv(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title in arXiv, with error handling and retries."""
    try:
        url = "http://export.arxiv.org/api/query"
        params = {'search_query': f'ti:"{ref.title}"', 'max_results': 5}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'lxml-xml')
            entries = soup.find_all('entry')
            
            if not entries:
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
                    
                    title_score = fuzz.ratio(normalized_arxiv_title, normalized_input_title)
                    
                    # Check for exact match first
                    if normalized_arxiv_title == normalized_input_title:
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Title match in arXiv (exact match).")
                    
                    # Check for very high confidence fuzzy match only
                    if title_score >= 90:
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation=f"Title match in arXiv (high confidence: {title_score}%).")
                    
                    # Check for high confidence match with author verification
                    if title_score >= 85:
                        author_tags = entry.find_all('author')
                        for author_tag in author_tags:
                            name_tag = author_tag.find('name')
                            if name_tag:
                                author_name = name_tag.text.strip()
                                last_name = author_name.split()[-1].lower()
                                ref_author = ref.author.lower()
                                
                                # Strict author matching
                                author_match = (
                                    ref_author == last_name or
                                    fuzz.ratio(ref_author, last_name) > 90
                                )
                                
                                if author_match:
                                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation=f"Author and similar title match in arXiv (confidence: {title_score}%).")
                                
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in arXiv.")
        
    except Exception as e:
        logging.warning(f"arXiv search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"arXiv search failed: {e}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_workshop_paper(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for workshop papers using Google Search directly."""
    try:
        workshop_indicators = ['workshop', 'symposium', 'proc.', 'proceedings']
        is_likely_workshop = any(indicator in ref.bib.lower() for indicator in workshop_indicators)
        
        if not is_likely_workshop:
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Not a workshop paper.")
            
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
        client = genai.Client(api_key=GOOGLE_API_KEY)
        google_search_tool = Tool(google_search=GoogleSearch())
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={'tools': [google_search_tool], 'temperature': 0},
        )
        answer = normalize_title(response.candidates[0].content.parts[0].text)
        if answer.startswith('true') or answer.endswith('true'):
            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Workshop paper found via Google search.")
        else:
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Workshop paper not found via Google search.")
            
    except Exception as e:
        logging.warning(f"Workshop paper search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Workshop paper search failed: {e}")

def verify_url(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Verifies if the title on the webpage at the given URL matches the reference title."""
    if not ref.URL:
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No URL provided.")
    try:
        response = requests.get(ref.URL, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find('title')
        if title_tag:
            webpage_title = title_tag.text.strip()
            normalized_webpage_title = normalize_title(webpage_title)
            normalized_input_title = normalize_title(ref.title)
            if normalized_webpage_title == normalized_input_title:
                return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Webpage title matches reference title (exact match).")
            elif normalized_input_title in normalized_webpage_title or normalized_webpage_title in normalized_input_title:
                return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Webpage title matches reference title (partial match).")
        else:
            logging.warning(f"No <title> tag found at URL: {ref.URL}")
            return search_title_google(ref)
    except requests.exceptions.RequestException as e:
        logging.warning(f"Error accessing URL {ref.URL}: {e}")
        return search_title_google(ref)
    except Exception as e:
        logging.warning(f"Error processing URL {ref.URL}: {e}")
        return search_title_google(ref)

def search_title_google(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title using Google Search and match using a LLM model."""
    try:
        prompt = f"""
        Please search for the reference on Google, compare with research results, and determine if it is genuine.\n
        Return 'True' only if a website with the the exact title and author is found. Otherwise, return 'False'.\n
        Return only 'True' or 'False', without any additional information.\n\n
        Author: {ref.author}\n
        Title: {ref.title}\n"""
        client = genai.Client(api_key=GOOGLE_API_KEY)
        google_search_tool = Tool(google_search=GoogleSearch())
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={'tools': [google_search_tool]},
        )
        answer = normalize_title(response.candidates[0].content.parts[0].text)
        if answer.startswith('true') or answer.endswith('true'):
            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Google search found matching reference.")
        else:
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Google search did not find matching reference.")
    except Exception as e:
        logging.warning(f"Google search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Google search failed: {e}")

# --- REFINED: find_replacement_reference method ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=30))
def find_replacement_reference(ref: ReferenceExtraction) -> ReplacementSuggestion:
    """
    Finds a plausible, verifiable replacement for an invalid reference using a structured approach.
    """
    prompt = f"""
    You are a helpful research assistant. A reference was found to be invalid or non-existent.
    Your task is to find 3 real, verifiable, and topically similar academic papers to suggest as replacements.

    Original (invalid) reference: "{ref.bib}"
    Reference type: {ref.type}

    Please follow these steps:
    1.  Analyze the probable topic from the invalid reference's title and author.
    2.  Find 3 well-regarded, real academic papers of the SAME TYPE ({ref.type}) on that same topic.
    3.  Rank them by relevance and quality (match_score: 1-100, where 100 is perfect match).
    4.  Provide the details of these replacement papers in the requested JSON format.
    5.  Ensure URLs are direct links to the papers (DOI, arXiv, or publisher links).
    6.  Make sure all suggestions are of the same type as the original reference.

    Return exactly 3 suggestions, ranked from best to worst match.
    """
    try:
        if not GOOGLE_API_KEY:
            return ReplacementSuggestion(
                found=False,
                reasoning="Google API key not set. Please set your API key first.",
                suggestion1_bib="", suggestion1_url="", suggestion1_score=0,
                suggestion2_bib="", suggestion2_url="", suggestion2_score=0,
                suggestion3_bib="", suggestion3_url="", suggestion3_score=0
            )
            
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-pro', # Using a more powerful model for better reasoning
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': ReplacementSuggestion,
                'temperature': 0.1,
            },
        )
        
        # Ensure we return a proper ReplacementSuggestion object
        if hasattr(response, 'parsed') and response.parsed:
            return response.parsed
        else:
            # Fallback: try to parse the response text
            import json
            try:
                parsed_data = json.loads(response.text)
                return ReplacementSuggestion(**parsed_data)
            except Exception as parse_error:
                logging.warning(f"Failed to parse replacement response: {parse_error}")
                return ReplacementSuggestion(
                    found=False,
                    reasoning="Failed to parse replacement response from API",
                    suggestion1_bib="", suggestion1_url="", suggestion1_score=0,
                    suggestion2_bib="", suggestion2_url="", suggestion2_score=0,
                    suggestion3_bib="", suggestion3_url="", suggestion3_score=0
                )
    except Exception as e:
        logging.error(f"Failed to find replacement for '{ref.title}': {e}")
        # Return a structured object indicating failure
        return ReplacementSuggestion(
            found=False,
            reasoning=f"Replacement search failed: {str(e)}",
            suggestion1_bib="", suggestion1_url="", suggestion1_score=0,
            suggestion2_bib="", suggestion2_url="", suggestion2_score=0,
            suggestion3_bib="", suggestion3_url="", suggestion3_score=0
        )

def search_title(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title using multiple methods with progressive fallback."""
    try:
        if ref.type == "non_academic_website":
            result = verify_url(ref)
            if result is None:
                return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="URL verification failed.")
            return result
        else:
            # Try Crossref first (most reliable for academic papers)
            crossref_result = search_title_crossref(ref)
            if crossref_result and crossref_result.status in [ReferenceStatus.VALIDATED, ReferenceStatus.INVALID]:
                return crossref_result  # Return result if found (valid or invalid)
            
            # Try arXiv for preprints
            arxiv_result = search_title_arxiv(ref)
            if arxiv_result and arxiv_result.status == ReferenceStatus.VALIDATED:
                return arxiv_result
                
            # Try Google Scholar (more comprehensive but less reliable)
            scholar_result = search_title_scholarly(ref)
            if scholar_result and scholar_result.status == ReferenceStatus.VALIDATED:
                return scholar_result
                
            # Try workshop papers last (most lenient)
            workshop_result = search_title_workshop_paper(ref)
            if workshop_result and workshop_result.status == ReferenceStatus.VALIDATED:
                return workshop_result
            
            # If all sources failed, return the best available result
            # Prefer Crossref result if available, otherwise return NOT_FOUND
            if crossref_result and crossref_result.status == ReferenceStatus.NOT_FOUND:
                return crossref_result
            elif arxiv_result and arxiv_result.status == ReferenceStatus.NOT_FOUND:
                return arxiv_result
            elif scholar_result and scholar_result.status == ReferenceStatus.NOT_FOUND:
                return scholar_result
            else:
                return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No evidence found in any source.")
    except Exception as e:
        logging.error(f"Search failed for reference '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Search failed: {e}")

# --- Main Workflow ---
def veriexcite(pdf_path: str) -> Tuple[int, int, List[str], List[str]]:
    """
    Check references in a PDF. Returns:
    - count_verified: number of validated references
    - count_warning: number of warnings (invalid or not found)
    - list_warning: list of bib entries with warnings
    - list_explanations: list of explanations for each reference
    """
    full_text = extract_text_from_pdf(pdf_path)
    bib_text = extract_bibliography_section(full_text)
    references = split_references(bib_text)

    count_verified, count_warning = 0, 0
    list_warning = []
    list_explanations = []

    for idx, ref in enumerate(references):
        try:
            result = search_title(ref)
            if result is None:
                result = ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Search returned no result.")
        except Exception as e:
            logging.error(f"Error processing reference {idx}: {e}")
            result = ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Processing error: {e}")
        
        if result.status == ReferenceStatus.VALIDATED:
            count_verified += 1
            list_explanations.append(f"Reference: {ref.bib}\nStatus: {result.status.value}\nExplanation: {result.explanation}\n")
        else:
            count_warning += 1
            list_warning.append(ref.bib)
            
            # --- MODIFIED: Handle the structured replacement suggestion ---
            suggestion = find_replacement_reference(ref)
            if suggestion.found:
                replacement_text = f"找到 3 个替换建议\n"
                replacement_text += f"推荐理由: {suggestion.reasoning}\n\n"
                
                # Add all 3 suggestions
                suggestions = [
                    (suggestion.suggestion1_bib, suggestion.suggestion1_url, suggestion.suggestion1_score),
                    (suggestion.suggestion2_bib, suggestion.suggestion2_url, suggestion.suggestion2_score),
                    (suggestion.suggestion3_bib, suggestion.suggestion3_url, suggestion.suggestion3_score)
                ]
                
                for i, (bib, url, score) in enumerate(suggestions, 1):
                    if bib and bib.strip():  # Only show non-empty suggestions
                        replacement_text += f"建议 {i} (匹配度: {score}/100):\n"
                        replacement_text += f"文献: {bib}\n"
                        replacement_text += f"链接: {url}\n\n"
            else:
                replacement_text = f"未找到合适的替换建议\n原因: {suggestion.reasoning}"

            list_explanations.append(
                f"Reference: {ref.bib}\n"
                f"Status: {result.status.value}\n"
                f"Explanation: {result.explanation}\n"
                f"{replacement_text}\n"
            )
            
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

def process_folder(folder_path: str) -> None:
    """Check all PDF files in a folder."""
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    pdf_files.sort()
    print(f"Found {len(pdf_files)} PDF files in the folder.")

    results = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Checking file: {pdf_file}")
        try:
            count_verified, count_warning, list_warning, list_explanations = process_pdf_file(pdf_path)
            print("--------------------------------------------------")
            results.append({"File": pdf_file, "Found References": count_verified + count_warning, "Verified": count_verified,
                            "Warnings": count_warning, "Warning List": list_warning, "Explanation": list_explanations})
        except Exception as e:
            print(f"Failed to process {pdf_file}: {e}")
    
    if results:
        pd.DataFrame(results).to_csv('VeriExCite results.csv', index=False)
        print("Results saved to VeriExCite results.csv")


if __name__ == "__main__":
    ''' Set your Google Gemini API key here '''
    # Apply for a key at https://ai.google.dev/
    GOOGLE_API_KEY = "YOUR_API_KEY"
    set_google_api_key(GOOGLE_API_KEY)

    ''' Example usage #1: check a single PDF file '''
    # pdf_path = "path/to/your/paper.pdf"
    # process_pdf_file(pdf_path)

    ''' Example usage #2: check all PDF files in a folder '''
    folder_path = "path/to/your/folder"
    process_folder(folder_path)