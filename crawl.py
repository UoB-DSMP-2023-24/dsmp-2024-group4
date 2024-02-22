# Use this script to crawl the CDR3 chains split into V and J from HTML files

from bs4 import BeautifulSoup
import csv

colors = {
    "rgb(77, 175, 74)": None,
    "black": None,
    "rgb(55, 126, 184)": None,
    "rgb(215, 25, 28)": None,
}

html_path='saved_pages//'
with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    for i in range(928): # 928
        # 读取 HTML 内容
        with open(html_path+str(i+1)+'.html', 'r', encoding='utf-8') as file:
            html_content = file.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        for tr in soup.find_all('tr', class_="center aligned fade element"):
            row = []
            for td in tr.find_all('td'):
                if td.has_attr('search-table-entry-cdr'):
                    for color in colors.keys():
                        colors[color] = ''
                    for span in td.find_all('span'):
                        color = span.get('style', '').split(':')[-1].strip(';').strip()
                        text = span.get_text(strip=True)
                        if color in colors:
                            colors[color] += text
                    for color, text in colors.items():
                        row.append(text)
                else:
                    text = td.get_text(strip=True)
                    row.append(text)
            csvwriter.writerow(row)


