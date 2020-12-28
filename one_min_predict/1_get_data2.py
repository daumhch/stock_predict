# pykiwoom 설치 하고 실행 안 되면 참고
# https://sangjjang.tistory.com/342

# 증권사 API는 32bit 여야 한다고 한다
# 그래서 아나콘다도 32bit 환경도 설치해야 한다
# https://chancoding.tistory.com/91
# https://lazyquant.tistory.com/5

# 또한 실행하는 인터프리터도 32bit로 설정해야 한다

from pykiwoom.kiwoom import *

kiwoom = Kiwoom()
kiwoom.CommConnect(block=True)
print('블로킹 로그인 완료')