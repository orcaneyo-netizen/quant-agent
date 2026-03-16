import requests
from bs4 import BeautifulSoup
import urllib.parse
import time

def scrape_finviz_headlines(ticker: str) -> list:
    """
    Scrapes the latest headlines for a ticker from Finviz.
    Returns: list of dicts with keys 'title', 'link', 'date'
    """
    url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch Finviz for {ticker}: {response.status_code}")
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table:
            print(f"No news table found for {ticker}")
            return []
            
        headlines = []
        rows = news_table.find_all('tr')
        
        for row in rows:
            # Finviz places date inside a td, inside row
            cols = row.find_all('td')
            if len(cols) != 2:
                continue
            
            date_col = cols[0].get_text().strip()
            link_col = cols[1].find('a')
            if link_col:
                title = link_col.get_text().strip()
                link = link_col['href']
                headlines.append({
                    'title': title,
                    'link': link,
                    'date': date_col # Contains date/time
                })
        
        return headlines
        
    except Exception as e:
        print(f"Error scraping Finviz for {ticker}: {e}")
        return []

def scrape_google_news(ticker: str) -> list:
    """
    Alternative: Scrape Google News RSS.
    Returns: list of dicts with keys 'title', 'link'
    """
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'xml') # Google News uses XML
        items = soup.find_all('item')
        
        headlines = []
        for item in items[:10]: # Top 10
            title = item.find('title').get_text()
            link = item.find('link').get_text()
            headlines.append({
                'title': title,
                'link': link
            })
        return headlines
    except Exception as e:
        print(f"Error reading Google News for {ticker}: {e}")
        return []

if __name__ == "__main__":
    # Test
    print(scrape_finviz_headlines("NVDA")[:3])
