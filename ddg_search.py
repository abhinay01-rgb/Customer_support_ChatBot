import json
from duckduckgo_search import ddg

x = None  # Initialize x as a global variable

def link(link):
    global x  # Declare x as a global variable
    x = link
    print(x)

def search_results(query):
    global x  # Declare x as a global variable
    print(f"x is {x}")
    # Search DuckDuckGo and scrape the results
    results = ddg(f"site:{x} {query}")
    print(results)
    s1 = json.dumps(results)
    data = json.loads(s1)  # Return a list of dictionaries

    text = []
    for item in data:
        text.append(item.get('title', 'No title found'))
        text.append(item.get('body', 'No body found'))

    # print("Title:", text)
    result = ", ".join(text)
    return result
