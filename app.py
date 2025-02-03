import streamlit as st
import pandas as pd
import requests
from html.parser import HTMLParser
import re
from typing import List, Dict

class SEOParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.meta_description = ""
        self.h1s = []
        self.h2s = []
        self.current_tag = None
        self.main_content = []
        self.in_nav = False
        self.in_header = False
        self.in_footer = False
        self.in_sidebar = False

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        attrs_dict = dict(attrs)
        
        if tag == 'meta' and attrs_dict.get('name') == 'description':
            self.meta_description = attrs_dict.get('content', '')
        elif tag in ['nav', 'header', 'footer', 'aside']:
            setattr(self, f'in_{tag}', True)

    def handle_endtag(self, tag):
        if tag in ['nav', 'header', 'footer', 'aside']:
            setattr(self, f'in_{tag}', False)
        self.current_tag = None

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_nav or self.in_header or self.in_footer or self.in_sidebar:
            return

        if self.current_tag == 'title':
            self.title = data
        elif self.current_tag == 'h1':
            self.h1s.append(data)
        elif self.current_tag == 'h2':
            self.h2s.append(data)
        elif self.current_tag in ['p', 'div', 'span', 'article']:
            self.main_content.append(data)

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and newlines"""
    return ' '.join(text.split())

def check_keyword_presence(text: str, keyword: str) -> bool:
    """Check if keyword is present in text (case insensitive)"""
    if not text or not keyword:
        return False
    return keyword.lower() in text.lower()

def analyze_url(url: str, keyword: str) -> Dict:
    """Analyze a URL for SEO elements and keyword presence"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        parser = SEOParser()
        parser.feed(response.text)
        
        # Get first 5 H2s or pad with empty strings
        h2s = parser.h2s[:5]
        h2s.extend([''] * (5 - len(h2s)))
        
        main_content = ' '.join(parser.main_content)
        
        return {
            'title': clean_text(parser.title),
            'title_contains': check_keyword_presence(parser.title, keyword),
            'meta_description': clean_text(parser.meta_description),
            'meta_contains': check_keyword_presence(parser.meta_description, keyword),
            'h1': clean_text(parser.h1s[0] if parser.h1s else ''),
            'h1_contains': check_keyword_presence(parser.h1s[0] if parser.h1s else '', keyword),
            'h2_1': clean_text(h2s[0]),
            'h2_1_contains': check_keyword_presence(h2s[0], keyword),
            'h2_2': clean_text(h2s[1]),
            'h2_2_contains': check_keyword_presence(h2s[1], keyword),
            'h2_3': clean_text(h2s[2]),
            'h2_3_contains': check_keyword_presence(h2s[2], keyword),
            'h2_4': clean_text(h2s[3]),
            'h2_4_contains': check_keyword_presence(h2s[3], keyword),
            'h2_5': clean_text(h2s[4]),
            'h2_5_contains': check_keyword_presence(h2s[4], keyword),
            'main_content': clean_text(main_content[:1000]),  # Limit content length for display
            'content_contains': check_keyword_presence(main_content, keyword)
        }
    except Exception as e:
        st.error(f"Error analyzing {url}: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="On Page Quick Wins", layout="wide")
    
    st.title("On Page Quick Wins")
    st.write("Upload your GSC performance report to analyze keyword usage in your pages.")
    
    uploaded_file = st.file_uploader("Upload GSC Performance Report (CSV)", type=['csv'])
    
    if uploaded_file:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            required_columns = ['Query', 'URL', 'Clicks', 'Impressions']
            
            if not all(col in df.columns for col in required_columns):
                st.error("CSV file must contain: Query, URL, Clicks, and Impressions columns")
                return
            
            # Sort by clicks first, then impressions for remaining slots
            top_queries = pd.concat([
                df.nlargest(10, 'Clicks'),
                df[df['Clicks'] == 0].nlargest(10 - len(df[df['Clicks'] > 0]), 'Impressions')
            ]).drop_duplicates().head(10)
            
            results = []
            progress_bar = st.progress(0)
            
            for idx, row in top_queries.iterrows():
                progress = (idx + 1) / len(top_queries)
                progress_bar.progress(progress)
                
                analysis = analyze_url(row['URL'], row['Query'])
                if analysis:
                    results.append({
                        'URL': row['URL'],
                        'Query': row['Query'],
                        'Clicks': row['Clicks'],
                        'Title': analysis['title'],
                        'Title Contains': analysis['title_contains'],
                        'Meta Description': analysis['meta_description'],
                        'Meta Contains': analysis['meta_contains'],
                        'H1': analysis['h1'],
                        'H1 Contains': analysis['h1_contains'],
                        'H2-1': analysis['h2_1'],
                        'H2-1 Contains': analysis['h2_1_contains'],
                        'H2-2': analysis['h2_2'],
                        'H2-2 Contains': analysis['h2_2_contains'],
                        'H2-3': analysis['h2_3'],
                        'H2-3 Contains': analysis['h2_3_contains'],
                        'H2-4': analysis['h2_4'],
                        'H2-4 Contains': analysis['h2_4_contains'],
                        'H2-5': analysis['h2_5'],
                        'H2-5 Contains': analysis['h2_5_contains'],
                        'Copy': analysis['main_content'],
                        'Copy Contains': analysis['content_contains']
                    })
            
            if results:
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Export option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Analysis Report",
                    data=csv,
                    file_name="seo_analysis_report.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
