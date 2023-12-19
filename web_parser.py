from requests import get
from bs4 import BeautifulSoup
from wikipedia import summary,search


# Scrapes content of given link and parse it to find relevant content
def parser(url):
    res=get(url)
    soup = BeautifulSoup(res.content, 'html.parser') 
    if 'spaceplace.nasa.gov' in url:  # if given url is of nasa.com
        s=soup.find('div',id='main')
        s=s.get_text()
        lines = (line.strip() for line in s.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text

    elif 'en.wikipedia.org' in url:  # if given url is of wikipedia
        results = search(url.split('/')[-1])
        text=summary(results[0],auto_suggest=False)
        return text

    elif 'byjus.com' in url:  # if given url of byjus.com
        s=soup.find('div',class_='bgc-white p30 mb20 pm15')
        s=s.find_all('p')
        text=""
        for i in s:
            text+=i.get_text()+" "
        return text
    return ""


if __name__=="__main__":
    url="https://en.wikipedia.org/wiki/Gravity"
    print(parser(url))