import os
import json
import requests
import psycopg2
from urllib.parse import urlencode, quote_plus
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

# 한국 시간대로 변경(한국 시간대(KST)는 UTC+9)
now_kst = datetime.now(timezone.utc) + timedelta(hours=9)

# .env 로드
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
API_KEY = os.getenv("G2B_API_KEY")

# DB 연결
conn = psycopg2.connect(SUPABASE_DB_URL)
cur = conn.cursor()

# Supabase에서 마지막 수집 시각 불러오기
try:
    cur.execute("""
    SELECT raw
    FROM bids_live
    ORDER BY 
        raw->>'bidNtceDt' DESC, 
        raw->>'bidNtceBgn' DESC LIMIT 1                
""")
    result = cur.fetchone()
    raw = json.loads(result[0]) if result else {}
    start_time = raw.get("bidNtceBgn", (now_kst - timedelta(minutes=10)).strftime("%Y%m%d%H%M"))
    
except Exception as e:
    print("start_time 불러오기 오류:", e)
    start_time = (now_kst - timedelta(minutes=10)).strftime("%Y%m%d%H%M")

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

print(f"{len(info)}건 저장 완료")
