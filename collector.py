import os
import json
import requests
import psycopg2
from urllib.parse import urlencode, quote_plus
from dotenv import load_dotenv
from datetime import datetime, timedelta

# 한국 시간대로 변경(한국 시간대(KST)는 UTC+9)
now_kst = datetime.utcnow() + timedelta(hours=9)

# .env 로드
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
API_KEY = os.getenv("G2B_API_KEY")

# DB 연결
conn = psycopg2.connect(SUPABASE_DB_URL)
cur = conn.cursor()

# 수집 시점 관리
def load_last_time(file, default):
    try:
        with open(file, 'r') as f:
            return json.load(f)['last_collected_time']
    except:
        return default

last_file = "last_time.json"
start_time = load_last_time(last_file, (now_kst - timedelta(minutes=10)).strftime("%Y%m%d%H%M"))
end_time = (now_kst).strftime("%Y%m%d%H%M")

# 나라장터 API 호출
BASE_URL = "http://apis.data.go.kr/1230000/ao/PubDataOpnStdService/getDataSetOpnStdBidPblancInfo"

page = 1
info = []

while True:
    params = {
        'serviceKey': API_KEY,
        'pageNo': page,
        'numOfRows': 100,
        'inqryDiv': 1,
        'type': 'json',
        'bidNtceBgnDt': int(start_time),
        'bidNtceEndDt': int(end_time),
    }
    url = f"{BASE_URL}?{urlencode(params, quote_via=quote_plus)}"

    try:
        data = requests.get(url).json()
        items = data['response']['body']['items']
    except:
        break

    if not items:
        break
    info.extend(items)
    page += 1

# Supabase에 삽입
for item in info:
    try:
        cur.execute("""
            INSERT INTO bids_live (bidNtceNo, raw)
            VALUES (%s, %s)
            ON CONFLICT (bidNtceNo) DO UPDATE SET raw = EXCLUDED.raw
        """, (item.get("bidNtceNo"), json.dumps(item, ensure_ascii=False)))
    except Exception as e:
        print("저장 오류:", e)

conn.commit()
conn.close()

# 마지막 수집 시간 기록: 마지막 공고의 게시일 기준
if info:
    last_time = info[-1].get("bidNtceBgnDt", None)
    if last_time:
        with open(last_file, 'w') as f:
            json.dump({"last_collected_time": last_time}, f)

        # GitHub에 자동 커밋
        try:
            os.system("git config --global user.email 'dbwoals137@gmail.com'")
            os.system("git config --global user.name 'Min-Jae717'")
            os.system("git pull")
            os.system("git add last_time.json")
            os.system("git commit -m 'update last_time.json [skip ci]' || echo 'nothing to commit'")
            os.system("git push")
        except Exception as e:
            print("자동 커밋 오류:", e)

print(f"{len(info)}건 저장 완료")
