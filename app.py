import streamlit as st
import pandas as pd
import requests
import bs4
BeautifulSoup = bs4.BeautifulSoup
import re

def clean_text(text):
    """Clean text by removing extra whitespace and newlines"""
    return ' '.join(text.split())

def check_keyword_presence(text, keyword):
    """Check if keyword is present in text (case insensitive)"""
    if not text or not keyword:
        return False
    return keyword.lower() in text.lower()

def extract_main_content(soup):
    """Extract main content from body, excluding navigation, header, footer, etc."""
    # Remove unwanted elements
    for unwanted in soup.select('nav, header, footer, sidebar, .nav, .header, .footer, .sidebar, [role="navigation"]'):
        unwanted.decompose()
    
    # Get remaining text from body
    body = soup.find('body')
    if body:
        return clean_text(' '.join(body.stripped_strings))
    return ""

def analyze_url(url, keyword):
    """Analyze a URL for SEO elements and keyword presence"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract SEO elements
        title = soup.title.string if soup.title else ""
        meta_desc = soup.find('meta', {'name': 'description'})
        meta_desc = meta_desc['content'] if meta_desc else ""
        h1 = soup.find('h1').get_text() if soup.find('h1') else ""
        h2s = [h2.get_text() for h2 in soup.find_all('h2')][:5]  # Get first 5 H2s
        h2s.extend([""] * (5 - len(h2s)))  # Pad with empty strings if less than 5 H2s
        
        # Get main content
        main_content = extract_main_content(soup)
        
        return {
            'title': clean_text(title),
            'title_contains': check_keyword_presence(title, keyword),
            'meta_description': clean_text(meta_desc),
            'meta_contains': check_keyword_presence(meta_desc, keyword),
            'h1': clean_text(h1),
            'h1_contains': check_keyword_presence(h1, keyword),
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
            'main_content': main_content[:1000],  # Limit content length for display
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
