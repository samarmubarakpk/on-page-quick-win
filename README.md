# On Page Quick Wins

A Streamlit application that analyzes SEO elements of web pages based on Google Search Console performance data.

## Features

- Analyzes top 10 queries from GSC data (prioritizing by clicks, then impressions)
- Checks keyword presence in:
  - Page Title
  - Meta Description
  - H1 Heading
  - Top 5 H2 Subheadings
  - Main content (excluding navigation, header, footer, etc.)
- Generates downloadable CSV report

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Export your Google Search Console performance report as CSV with the following columns:
   - Query
   - URL
   - Clicks
   - Impressions

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Upload your CSV file through the web interface
4. Wait for the analysis to complete
5. Download the generated report

## Notes

- The application analyzes the top 10 queries, prioritizing those with the most clicks
- If there are queries with 0 clicks, remaining slots will be filled with queries having the highest impressions
- The main content analysis excludes navigational elements, header, footer, and sidebar content
