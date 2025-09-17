# VeriExCite - Academic Reference Verification System

A comprehensive AI-powered system for verifying academic references in research papers using Google Gemini AI and multiple academic databases.

## 🚀 Features

- **Smart Reference Parsing**: AI-powered extraction of titles, authors, DOIs, URLs, and publication years
- **Multi-Source Verification**: Cross-reference with Crossref, arXiv, and Google Scholar
- **Three-Tier Verification System**: Intelligent confidence-based verification workflow
- **AI-Powered Replacements**: Generate three replacement suggestions from different databases
- **Multi-Language Support**: Handle references in 8+ languages (Chinese, English, Japanese, French, German, Spanish, Russian, Italian, Portuguese, Korean)
- **Real-Time Progress**: Live updates during processing
- **Export Capabilities**: Download results as CSV

## 📋 Requirements

- Python 3.8+
- Google Gemini API key
- Required packages (see requirements.txt)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd VeriExCiting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Google Gemini API key:
```python
from veriexcite import set_google_api_key
set_google_api_key("your-api-key-here")
```

4. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## 🔄 Complete Verification Workflow

### **Step 1: Reference Parsing (AI)**
```
PDF Text → Gemini AI → Structured References
├── Title extraction
├── Author extraction  
├── DOI extraction
├── URL extraction
├── Year extraction
└── Type classification
```

### **Step 2: Three-Tier Verification System**

#### **Tier 1: High Confidence (< 70% invalid)**
- References with very low similarity scores
- Marked as INVALID immediately
- No further verification needed

#### **Tier 2: Medium Confidence (70-95% AI check)**
- References with moderate similarity scores
- Gemini AI performs DOI/URL verification
- AI checks title and year (not author names)
- Database search fallback if AI verification fails

#### **Tier 3: High Confidence (> 95% valid)**
- References with very high similarity scores
- Marked as VALIDATED immediately
- No additional verification needed

### **Step 3: Verification Methods**

#### **DOI/URL Verification (If Available)**
```
DOI/URL Verification:
├── Gemini DOI Verification (Primary)
│   ├── Search for DOI on web
│   ├── Access paper at DOI URL
│   ├── Extract paper metadata
│   ├── Compare title and year
│   └── Return True/False
├── Database Search Fallback (If Gemini says invalid)
│   ├── Search Crossref by title + author
│   ├── Search arXiv by title + author
│   ├── Search Google Scholar by title + author
│   └── Return best match result
└── Crossref Direct DOI Lookup (Fallback)
    ├── GET https://api.crossref.org/works/{DOI}
    ├── Check if DOI exists
    └── Return result
```

#### **Title + Author Verification (If No DOI/URL)**
```
Title + Author Verification:
├── Crossref API Search
│   ├── Search by title
│   ├── Fuzzy match title (90%+ threshold)
│   ├── Check author match (80%+ threshold)
│   └── Return combined score
├── arXiv API Search
│   ├── Search by title
│   ├── Exact title match OR 90%+ fuzzy match
│   ├── Check author match (90%+ threshold)
│   └── Return result
├── Google Scholar Search
│   ├── Search by title
│   ├── Exact title match OR 90%+ fuzzy match
│   ├── Check author match (80%+ threshold)
│   └── Return result
└── Workshop Paper Search (AI)
    ├── Check if likely workshop paper
    ├── Gemini + Google Search
    ├── Verify workshop paper exists
    └── Return result
```

### **Step 4: Replacement Generation (If Invalid/Not Found)**

```
AI-Powered Replacement Generation:
├── Gemini analyzes topic from invalid reference
├── Gemini searches three academic databases:
│   ├── arXiv (preprints and technical papers)
│   ├── Crossref (journal articles and conference papers)
│   └── Google Scholar (comprehensive academic search)
├── Gemini performs self-verification
│   ├── Verify DOI/URL is properly formatted
│   ├── Confirm paper exists and is related to topic
│   ├── Ensure bibliography information is accurate
│   └── Check paper is from assigned database
├── Return three suggestions with:
│   ├── Complete bibliography entry
│   ├── Valid DOI or direct URL
│   ├── Relevance score (1-100)
│   └── Source database name
└── No additional verification needed
```

## 🎯 Verification Priority Order

### **For References WITH DOI/URL:**
1. **Gemini DOI/URL Verification** (Most Reliable)
2. **Crossref Direct DOI Lookup** (Fallback)
3. **Title Search** (Final Fallback)

### **For References WITHOUT DOI/URL:**
1. **Crossref Title Search** (Most Reliable)
2. **arXiv Title Search** (Preprints)
3. **Google Scholar Search** (Comprehensive)
4. **Workshop Paper Search** (AI-powered)

## 🔧 Key Features

### **AI-Powered Verification:**
- **Gemini DOI Verification**: Visits DOI URLs and verifies content
- **Gemini URL Verification**: Checks if URLs lead to legitimate academic sources
- **Gemini Replacement Suggestions**: Finds three alternative papers using AI
- **Self-Verification**: AI ensures all suggestions are valid before returning

### **Database Verification:**
- **Crossref API**: Authoritative academic database
- **arXiv API**: Preprint and conference papers
- **Google Scholar**: Comprehensive academic search
- **Database Fallback**: Uses database search when AI verification fails

### **Smart Verification System:**
- **Three-Tier Classification**: Optimizes AI usage based on confidence scores
- **Fuzzy Matching**: 90%+ threshold for title validation
- **Author Matching**: Flexible matching (80%+ threshold)
- **Year Tolerance**: ±2 years acceptable for DOI verification

### **Error Handling:**
- **Retry Mechanisms**: 3 attempts with exponential backoff
- **Fallback Chains**: Multiple verification methods
- **Graceful Degradation**: System works even if some APIs fail
- **Unicode Support**: Handles international characters properly

## 📊 Success Criteria

### **VALIDATED Requirements:**
- **DOI/URL**: Must exist and match reference details
- **Title**: Must be substantially similar (90%+ match)
- **Author**: Must match (flexible matching allowed)
- **Year**: Must be close (±2 years for DOI verification)
- **Confidence**: Combined score above 95%

### **INVALID Triggers:**
- **DOI/URL**: Doesn't exist or doesn't match
- **Title**: Similarity below 90% threshold
- **Author**: No match found
- **Confidence**: Combined score below 70%

### **Replacement Requirements:**
- **Valid DOI/URL**: Must have accessible DOI or direct URL
- **Topic Relevance**: Must be related to original reference topic
- **Database Source**: One from each database (arXiv, Crossref, Google Scholar)
- **Self-Verified**: AI ensures validity before returning

## 🚀 Usage

1. **Upload PDF**: Upload your research paper PDF
2. **Set API Key**: Enter your Google Gemini API key
3. **Process**: Click "Process References" to start verification
4. **Review Results**: View verification status and replacement suggestions
5. **Export**: Download results as CSV for further analysis

## 📁 Project Structure

```
VeriExCiting/
├── veriexcite.py          # Core verification logic
├── streamlit_app.py       # Web interface
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── images/               # Screenshots and diagrams
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**VeriExCite** - Making academic reference verification intelligent, efficient, and reliable! 🎓✨