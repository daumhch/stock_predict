from bs4 import BeautifulSoup
import requests
from pprint import pprint

class Demo:
    def read_kospi(self, path):
        html = requests.get(path)
        soup = BeautifulSoup(html.text, 'html5lib')
        data1 = soup.find('div', {'class':'quotient up'})
        now_kospi = data1.find('em', {'id':'now_value'}).text.replace(',', '')
        return float(now_kospi)
        

if __name__ == "__main__":
    path = 'https://finance.naver.com/sise/sise_index.nhn?code=KOSPI'

    demo = Demo()
    print( demo.read_kospi(path) )





