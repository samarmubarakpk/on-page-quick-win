# SEO Query Element Analyzer

A Streamlit application that analyzes Google Search Console reports against page SEO elements.

## Features
- Processes GSC reports (CSV/XLSX)
- Analyzes top queries by clicks/impressions
- Checks keyword usage in:
  - Page titles
  - Meta descriptions
  - H1/H2 headings
  - Body content (excluding navigation elements)
- Generates interactive CSV reports

## Requirements
- Python 3.8+
- Dependencies in `requirements.txt`

## Installation
```powershell
pip install -r requirements.txt
```

## Usage
1. Export GSC performance report
2. Run the app:
```powershell
streamlit run app.py
```
3. Upload your report and view analysis

## Output Columns
- URL
- Query
- Clicks
- Title/Meta/H1/H2s/Body content
- Boolean matches for each element
