import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Function to extract HTML content from the URL
def extract_html_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    except Exception as e:
        return None

# Function to check if the query is present in the given HTML element text
def check_query_in_element(query, element):
    if element and query.lower() in element.get_text().lower():
        return True
    return False

# Function to analyze SEO for a list of queries and URLs
def analyze_seo(gsc_data):
    results = []

    # Iterate through the data for the top 10 queries
    for _, row in gsc_data.iterrows():
        url = row['URL']
        query = row['Query']
        clicks = row['Clicks']
        
        # Fetch the HTML content of the URL
        soup = extract_html_content(url)
        if not soup:
            continue

        # Extract tags
        title = soup.find('title')
        meta_description = soup.find('meta', attrs={'name': 'description'})
        h1_tag = soup.find('h1')
        h2_tags = soup.find_all('h2')
        body_content = soup.find('body')

        # Check query in each element
        title_check = check_query_in_element(query, title)
        meta_check = check_query_in_element(query, meta_description)
        h1_check = check_query_in_element(query, h1_tag)
        h2_checks = [check_query_in_element(query, h2) for h2 in h2_tags]
        body_check = check_query_in_element(query, body_content)

        # Prepare row of results for this query
        result_row = {
            "Address/URLs": url,
            "Query": query,
            "Clicks": clicks,
            "Title": title.get_text() if title else '',
            "is being used?": title_check,
            "Meta Description": meta_description.get('content') if meta_description else '',
            "is being used?": meta_check,
            "H1 Heading": h1_tag.get_text() if h1_tag else '',
            "is being used?": h1_check,
            "H2-1": h2_tags[0].get_text() if len(h2_tags) > 0 else '',
            "is being used?": h2_checks[0] if len(h2_tags) > 0 else False,
            "H2-2": h2_tags[1].get_text() if len(h2_tags) > 1 else '',
            "is being used?": h2_checks[1] if len(h2_tags) > 1 else False,
            "H2-3": h2_tags[2].get_text() if len(h2_tags) > 2 else '',
            "is being used?": h2_checks[2] if len(h2_tags) > 2 else False,
            "H2-4": h2_tags[3].get_text() if len(h2_tags) > 3 else '',
            "is being used?": h2_checks[3] if len(h2_tags) > 3 else False,
            "H2-5": h2_tags[4].get_text() if len(h2_tags) > 4 else '',
            "is being used?": h2_checks[4] if len(h2_tags) > 4 else False,
            "Copy": body_content.get_text() if body_content else '',
            "is being used?": body_check
        }

        results.append(result_row)

    # Return the results as a DataFrame
    return pd.DataFrame(results)

# Streamlit UI elements
st.title('SEO Keyword Check for Queries')

# User input: Upload GSC report CSV file
uploaded_file = st.file_uploader("Upload GSC Performance Report CSV", type=["csv"])

if uploaded_file:
    gsc_data = pd.read_csv(uploaded_file)
    
    # Filter the top 10 queries based on Clicks
    sorted_gsc_data = gsc_data.sort_values(by=['Clicks'], ascending=False)
    top_queries = sorted_gsc_data.head(10)  # Get top 10 queries based on Clicks
    
    # If there are queries with 0 clicks, fill in with queries based on Impressions
    if top_queries['Clicks'].sum() == 0:
        top_queries = gsc_data.sort_values(by=['Impressions'], ascending=False).head(10)
    
    # Run SEO analysis
    seo_report = analyze_seo(top_queries)
    
    # Display the results in a table
    st.write(seo_report)
