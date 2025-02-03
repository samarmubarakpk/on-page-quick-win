import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def fetch_html(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Error fetching {url}: {str(e)}")
        return None

def extract_seo_elements(html):
    soup = BeautifulSoup(html, 'lxml')
    
    # Remove navigational elements
    for nav in soup.find_all(['header', 'footer', 'nav', 'aside', 'form']):
        nav.decompose()
    
    return {
        'title': soup.title.string if soup.title else '',
        'meta_description': soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else '',
        'h1': [h.get_text(strip=True) for h in soup.find_all('h1')],
        'h2': [h.get_text(strip=True) for h in soup.find_all('h2')][:5],
        'body_text': ' '.join(soup.find('body').stripped_strings) if soup.body else ''
    }

def check_keyword_usage(text, query):
    return str(query).lower() in text.lower()

# Streamlit UI
st.title('SEO Query Element Analyzer')
uploaded_file = st.file_uploader("Upload GSC Performance Report", type=['csv', 'xlsx'])

if uploaded_file:
    gsc_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    # Process queries
    top_queries = gsc_data.sort_values(
        by=['Clicks', 'Impressions'], 
        ascending=[False, False]
    ).head(10)
    
    results = []
    
    for _, row in top_queries.iterrows():
        if not is_valid_url(row['URL']):
            continue
            
        html = fetch_html(row['URL'])
        if not html:
            continue
            
        seo_data = extract_seo_elements(html)
        
        result = {
            'URL': row['URL'],
            'Query': row['Query'],
            'Clicks': row['Clicks'],
            'Title': seo_data['title'],
            'Title Match': check_keyword_usage(seo_data['title'], row['Query']),
            'Meta Description': seo_data['meta_description'],
            'Meta Match': check_keyword_usage(seo_data['meta_description'], row['Query']),
            'H1': seo_data['h1'][0] if seo_data['h1'] else '',
            'H1 Match': any(check_keyword_usage(h1, row['Query']) for h1 in seo_data['h1']),
            **{f'H2-{i+1}': h2 if i < len(seo_data['h2']) else '' for i, h2 in enumerate(seo_data['h2'])},
            **{f'H2-{i+1} Match': check_keyword_usage(h2, row['Query']) if i < len(seo_data['h2']) else False for i, h2 in enumerate(seo_data['h2'])},
            'Body Text': seo_data['body_text'][:500] + '...' if seo_data['body_text'] else '',
            'Body Match': check_keyword_usage(seo_data['body_text'], row['Query'])
        }
        
        results.append(result)
    
    if results:
        report_df = pd.DataFrame(results)
        st.dataframe(report_df)
        st.download_button(
            label="Download Report",
            data=report_df.to_csv(index=False),
            file_name='seo_element_analysis.csv',
            mime='text/csv'
        )
    else:
        st.warning("No valid URLs processed")
