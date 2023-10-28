import json
from duckduckgo_search import ddg

#query = "address of ihub?"

def search_results(query,link):
    # search DuckDuckGo and scrape the results
    #results = ddg(f"site:ihubiitmandi.in {query}")
    results = ddg(f"{link}{query}")
    s1=json.dumps(results)
    data = json.loads(s1)#return a list of dictionary

    text = []
    for item in data:
        text.append(item.get('title', 'No title found'))
        text.append(item.get('body', 'No body found'))

    #print("Title:", text)
    result = ", ".join(text)
    return result