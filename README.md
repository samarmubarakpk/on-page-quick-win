# SEO Keyword Check for Queries

This Streamlit application performs an SEO audit by cross-referencing a search query with the key elements of a webpage. It checks if the query is being used in key SEO elements such as the title, meta description, H1, H2 subheadings, and within the page content (ignoring header, footer, sidebar, popups, etc.).

## Features

- Upload a Google Search Console (GSC) performance report (CSV format) with queries, URLs, and clicks.
- Extracts SEO-related elements (Title, Meta Description, H1, H2, and body content) from the provided URLs.
- Check whether each query is used in the respective SEO elements.
- Displays a detailed report with "is being used?" checks for each query in key SEO elements.

## How It Works

1. **Input**: The user uploads a GSC performance report containing queries, URLs, clicks, impressions, average position, and URL CTR.
2. **Processing**: The script processes the top 10 queries with the highest clicks (or impressions if clicks are 0) and checks if each query is used in key SEO elements on each URL.
3. **Output**: The results are displayed in a table format showing whether the query is used in the Title, Meta Description, H1, H2, and the body content.

## Requirements

To run the application, you will need the following Python packages:
- `streamlit`: For building the Streamlit web application.
- `pandas`: For handling CSV data and data manipulation.
- `requests`: For fetching the webpage content.
- `beautifulsoup4`: For parsing HTML and extracting SEO elements.

You can install the required packages using the following command:

