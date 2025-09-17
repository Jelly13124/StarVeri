# VeriExCite Reference Verification Workflow

## 🔄 Complete Verification Process

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


### **Step 2: Reference Type Classification**
```
Reference Type Check:
├── Website Reference (non_academic_website)
│   └── Go to Step 4A: URL Verification
└── Academic Reference (journal_article, preprint, etc.)
    └── Go to Step 4B: Academic Verification
```

### **Step 3A: Website Reference Verification**
```
URL Verification:
├── Gemini URL Verification (Primary)
│   ├── Visit URL with web search
│   ├── Check page accessibility
│   ├── Extract title and author
│   ├── Compare with reference
│   └── Return True/False
├── Database Search Fallback (If Gemini says invalid)
│   ├── Search Crossref by title + author
│   ├── Search arXiv by title + author
│   ├── Search Google Scholar by title + author
│   └── Return best match result
├── Traditional URL Verification (Fallback)
│   ├── HTTP request to URL
│   ├── Parse HTML title
│   ├── Compare titles
│   └── Return match result
└── Google Search Verification (Final Fallback)
    ├── Search for title + author
    ├── Check search results
    └── Return match result
```

### **Step 3B: Academic Reference Verification**

#### **3B.1: DOI/URL Verification (If Available)**
```
DOI/URL Verification:
├── Extract DOI from URL if present
├── Gemini DOI Verification (Primary)
│   ├── Search for DOI on web
│   ├── Access paper at DOI URL
│   ├── Extract paper metadata
│   ├── Compare title, author, year
│   └── Return True/False
├── Database Search Fallback (If Gemini says invalid)
│   ├── Search Crossref by title + author
│   ├── Search arXiv by title + author
│   ├── Search Google Scholar by title + author
│   └── Return best match result
├── Crossref Direct DOI Lookup (Fallback)
│   ├── GET https://api.crossref.org/works/{DOI}
│   ├── Check if DOI exists
│   ├── Compare author match
│   └── Return result
└── Title Search (Final Fallback)
    ├── Search Crossref by title
    ├── Fuzzy match title + author
    └── Return result
```

#### **3B.2: Title + Author Verification (If No DOI/URL)**
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

### **Step 4: Result Classification**
```
Verification Result:
├── VALIDATED ✅
│   ├── DOI/URL matches + title/author matches
│   ├── Title + author match (90%+ confidence)
│   ├── Workshop paper verified
│   └── Database search fallback successful
├── INVALID ❌
│   ├── DOI/URL doesn't match
│   ├── Title/author doesn't match
│   ├── Confidence too low (<90%)
│   └── Reference found but details don't match
└── NOT_FOUND ❓
    ├── No evidence in any source
    ├── API failures
    └── Reference not found anywhere
```

### **Step 5: Replacement Suggestions with Verification (If Invalid/Not Found)**
```
Replacement Generation (AI with Verification):
├── Analyze invalid reference
├── Identify topic and type
├── Find 1 best similar paper of same type
├── Rank by relevance (1-100 score)
├── Provide bibliography entry
├── Include URL and match score
├── AI Generation (no verification)
│   ├── Analyze topic from invalid reference
│   ├── Find any relevant academic paper on same topic
│   ├── Generate bibliography entry (any type acceptable)
│   ├── Provide URL and confidence score
│   └── Return suggestion directly
├── If generation fails:
│   ├── Retry with new suggestion (up to 3 attempts)
│   ├── Ask Gemini for different paper
│   └── Continue until replacement found
└── Return suggestion
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
- **Gemini DOI Verification**: Actually visits DOI URLs and verifies content
- **Gemini URL Verification**: Checks if URLs lead to legitimate academic sources
- **Gemini Replacement Suggestions**: Finds 1 best alternative paper using AI
- **Gemini Format Validation**: Validates citation formats (APA, Harvard, MLA, etc.)
- **Gemini Style Consistency**: Ensures all references use the same citation style

### **Database Verification:**
- **Crossref API**: Authoritative academic database
- **arXiv API**: Preprint and conference papers
- **Google Scholar**: Comprehensive academic search
- **Database Fallback**: Uses database search when AI says URL/DOI is invalid


### **Replacement Generation:**
- **AI Generation Only**: Gemini AI generates replacement suggestions without verification
- **Flexible Topic Matching**: AI finds papers on the same topic (any type acceptable)
- **No Verification**: Replacement suggestions are returned as-is from AI
- **Retry Mechanism**: Up to 3 attempts to find replacement
- **Direct Return**: AI suggestions are returned without any validation
- **Single Best Match**: Provides one topic-relevant replacement per invalid reference

### **Fuzzy Matching:**
- **Title Similarity**: 92.5%+ threshold for validation
- **Author Matching**: Flexible matching (80%+ threshold)
- **Year Tolerance**: ±2 years acceptable for DOI verification

### **Error Handling:**
- **Retry Mechanisms**: 3 attempts with exponential backoff
- **Fallback Chains**: Multiple verification methods
- **Graceful Degradation**: System works even if some APIs fail
- **Smart Fallbacks**: Database search when AI verification fails

## 📊 Success Criteria

### **VALIDATED Requirements:**
- **Format**: Must follow standard academic citation format
- **Style**: Must be consistent with other references in the paper
- **DOI/URL**: Must exist and match reference details
- **Title**: Must be substantially similar (90%+ match)
- **Author**: Must match (flexible matching allowed)
- **Year**: Must be close (±2 years for DOI verification)



### **INVALID Triggers:**
- **DOI/URL**: Doesn't exist or doesn't match
- **Title**: Similarity below 90% threshold
- **Author**: No match found
- **Confidence**: Combined score below 90%

### **Replacement Verification:**
- **URL Accessibility**: Replacement URL must be accessible
- **Academic Content**: Must appear to be academic content
- **Quality Check**: Must pass verification tests

This comprehensive workflow ensures accurate, consistent, and reliable reference verification using AI, database methods, and quality assurance checks!
