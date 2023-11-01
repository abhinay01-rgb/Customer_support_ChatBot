
import requests
import re
import json
from bs4 import BeautifulSoup
from duckduckgo_search import ddg

x = None  # Initialize x as a global variable

def link(link):
    global x  # Declare x as a global variable
    x = link
    print(x)


def get_search_results(query):
    global x  # Declare x as a global variable
    print(f"x is {x}")
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
  
    results = ddg(f"site:{x} {query}")
    s1=json.dumps(results)
    data = json.loads(s1)

    url = []
    count = 0
    text =""
    
    for item in data:
        href = item.get('href', 'No href found')
        url.append(href)
        count += 1
        if count == 2:
            break

    for u in url:
        res=requests.get(u,headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')

        text += re.sub(r'\s+', ' ', soup.get_text().strip())
        print(text)
    return text


