import os
import zipfile
import tempfile
import requests
from bs4 import BeautifulSoup
from contextlib import closing
from hwp5.hwp5html import HTMLTransform
from hwp5.xmlmodel import Hwp5File
import fitz  # PyMuPDF
import re
import pandas as pd
from docx import Document
import openpyxl
import json
import psycopg2
from urllib.parse import urlencode, quote_plus
from datetime import datetime, timedelta
import openai
from tqdm import tqdm

# 한국 시간대로 변경(한국 시간대(KST)는 UTC+9)
now_kst = datetime.utcnow() + timedelta(hours=9)

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

# ====== 나라장터 공고서 파일 다운로드 ======
# 나라장터 API 호출
def nara_api() :
    BASE_URL = ["http://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoCnstwk",
                "http://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoServc",
                "http://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoFrgcpt",
                "http://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoThng"]

    tag = ["공사", "용역", "외자", "물품"]

    page = 1
    info = []

    for t, u in zip(tag, BASE_URL) :
        while True :
            params = {
                'serviceKey': API_KEY,
                'pageNo': page,
                'numOfRows': 100,
                'inqryDiv': 1,
                'type': 'json',
                'inqryBgnDt': start_time,
                'inqryEndDt': end_time
            }
            url = f"{u}?{urlencode(params, quote_via=quote_plus)}"

            try:
                response = requests.get(url)
                data = response.json()
                items = data['response']['body']['items']
                
                if not items :
                    break
                else :
                    for item in items :
                        item["bsnsDivNm"] = t        
                        info_dict = {}
                        info_dict["bidNtceNm"] = item["bidNtceNm"]
                        info_dict["bidNtceNo"] = item["bidNtceNo"]
                        info_dict["bsnsDivNm"] = item["bsnsDivNm"]
                        info_dict["stdNtceDocUrl"] = item["stdNtceDocUrl"]
                        info_dict["bidNtceDtlUrl"] = item["bidNtceDtlUrl"]
                        info.append(info_dict)
                    page += 1
            except Exception as e:
                print(e)
    return info


# ====== 첨부파일 텍스트 추출 ======
class DocumentTextExtractor:
    def __init__(self):
        pass

    @staticmethod
    def is_hwp5_file(file_path):
        try:
            with open(file_path, "rb") as f:
                header = f.read(8)
                return header.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1')
        except:
            return False

    @staticmethod
    def is_pdf_file(file_path):
        try:
            with open(file_path, "rb") as f:
                header = f.read(4)
                return header.startswith(b"%PDF")
        except:
            return False

    @staticmethod
    def extract_text_from_hwp5(file_path):
        import sys
        class DummyFile(object):
            def write(self, x): pass
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = DummyFile()
        sys.stderr = DummyFile()
        try:
            with closing(Hwp5File(file_path)) as hwp_file:
                transformer = HTMLTransform()
                with tempfile.TemporaryDirectory() as tmpdir:
                    transformer.transform_hwp5_to_dir(hwp_file, tmpdir)
                    html_path = os.path.join(tmpdir, "index.xhtml")
                    with open(html_path, "r", encoding="utf-8") as f:
                        soup = BeautifulSoup(f, "xml")
                    text_elements = soup.find_all(["p", "td", "th"])
                    return "\n".join(el.get_text(strip=True) for el in text_elements if el.get_text(strip=True))
        except Exception as e:
            print(f"⚠️ HWP5 변환 중 오류 발생 (스타일 오류 포함 가능): {e}")
            return ""
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    @staticmethod
    def extract_text_from_hwpx(file_path):
        with zipfile.ZipFile(file_path, 'r') as z:
            try:
                with z.open("Contents/section0.xml") as f:
                    soup = BeautifulSoup(f.read(), "xml")
            except KeyError:
                raise ValueError("HWPX 본문(section0.xml)을 찾을 수 없습니다.")
        parts = []
        paragraphs = soup.find_all("hp:p")
        parts.extend(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        table_cells = soup.find_all(["hp:cell", "hp:th"])
        parts.extend(cell.get_text(strip=True) for cell in table_cells if cell.get_text(strip=True))
        headers = soup.find_all("hp:header")
        footers = soup.find_all("hp:footer")
        parts.extend(el.get_text(strip=True) for el in headers + footers if el.get_text(strip=True))
        footnotes = soup.find_all("hp:footnote")
        parts.extend(fn.get_text(strip=True) for fn in footnotes if fn.get_text(strip=True))
        captions = soup.find_all("hp:caption")
        parts.extend(cp.get_text(strip=True) for cp in captions if cp.get_text(strip=True))
        return "\n".join(parts)

    @staticmethod
    def extract_text_from_pdf(file_path):
        extracted_text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                extracted_text += page.get_text().strip() + "\n"
        return extracted_text

    @staticmethod
    def extract_text_from_docx(file_path):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    @staticmethod
    def extract_text_from_xlsx(file_path):
        wb = openpyxl.load_workbook(file_path, data_only=True)
        texts = []
        for sheet in wb.worksheets:
            for row in sheet.iter_rows():
                row_text = [str(cell.value).strip() for cell in row if cell.value is not None]
                if row_text:
                    texts.append("\t".join(row_text))
        return "\n".join(texts)

    @staticmethod
    def extract_text_from_html(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        texts = []
        for tag in soup.find_all(["p", "td", "th", "div", "span"]):
            t = tag.get_text(strip=True)
            if t:
                texts.append(t)
        return "\n".join(texts)

    def extract_text_from_zip(self, file_path):
        texts = []
        with zipfile.ZipFile(file_path, 'r') as z:
            for name in z.namelist():
                if name.endswith(('.hwp', '.hwpx', '.pdf', '.docx', '.xlsx', '.htm', '.html')):
                    with z.open(name) as extracted:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[-1]) as tmpf:
                            tmpf.write(extracted.read())
                            tmpf.flush()
                            try:
                                text = self.extract_text_auto(tmpf.name)
                                texts.append(f"[{name}]\n{text}")
                            except Exception as e:
                                print(f"ZIP 내부 파일 오류: {name}, {e}")
                        os.unlink(tmpf.name)
        return "\n".join(texts)

    def extract_text_auto(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        ext = os.path.splitext(file_path)[-1].lower()
        if ext in [".htm", ".html"]:
            return self.extract_text_from_html(file_path)
        elif ext == ".docx":
            return self.extract_text_from_docx(file_path)
        elif ext == ".xlsx":
            return self.extract_text_from_xlsx(file_path)
        elif ext == ".zip":
            return self.extract_text_from_zip(file_path)
        elif self.is_pdf_file(file_path):
            return self.extract_text_from_pdf(file_path)
        elif zipfile.is_zipfile(file_path):
            return self.extract_text_from_hwpx(file_path)
        elif self.is_hwp5_file(file_path):
            return self.extract_text_from_hwp5(file_path)
        else:
            raise ValueError("지원하지 않는 파일 형식입니다.")

    def extract_text_from_url(self, file_url):
        with tempfile.TemporaryDirectory() as tmpdir:
            response = requests.get(file_url)
            print("📥 다운로드 응답 크기:", len(response.content), "bytes", "URL:", file_url)
            response.raise_for_status()
            content_disposition = response.headers.get("Content-Disposition", "").lower()
            filename = None
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].split(';')[0].strip('"')
            if filename:
                ext = os.path.splitext(filename)[-1].lower()
            else:
                ext = os.path.splitext(file_url)[-1].lower()
                if ext not in ['.pdf', '.hwp', '.hwpx', '.docx', '.xlsx', '.htm', '.html', '.zip']:
                    ext = '.hwp'
            file_path = os.path.join(tmpdir, "temp_file" + ext)
            with open(file_path, "wb") as f:
                f.write(response.content)
            return self.extract_text_auto(file_path)

    @staticmethod
    def preprocess_text(raw_text):
        lines = raw_text.splitlines()
        deduped_lines = []
        prev = None
        for line in lines:
            clean = line.strip()
            if clean and clean != prev:
                deduped_lines.append(clean)
                prev = clean
        joined_text = "\n".join(deduped_lines)
        joined_text = re.sub(r"[\uf000-\uf8ff]", "", joined_text)
        chunks = re.split(r"\n{2,}|(?<=\.)\n", joined_text)
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 20]
        return chunks

    @staticmethod
    def refine_chunks_to_sentences(chunks):
        refined = []
        for chunk in chunks:
            chunk = re.sub(r"^\s*[-•·∙●○※]?\s*", "", chunk)
            chunk = re.sub(r"(?P<key>[가-힣a-zA-Z0-9]+)\s*[:：]\s*(?P<val>.+)", r"\g<key>은(는) \g<val>입니다.", chunk)
            chunk = chunk.strip()
            if chunk and not chunk.endswith(("다", ".", "요")):
                chunk += "입니다."
            refined.append(chunk)
        return refined

    def process_info_list(self, info):
        table = []
        for i in info:
            table_dict = {}
            url = i.get("stdNtceDocUrl", "").strip()
            re_text = ""
            try:
                if url and url.startswith("http"):
                    raw_text = self.extract_text_from_url(url)
                    pre_text = self.preprocess_text(raw_text)
                    re_text = self.refine_chunks_to_sentences(pre_text)
                elif url == "":
                    re_text = ""
                else:
                    print("📛 유효하지 않은 URL:", url, i.get("bidNtceDtlUrl", ""))
            except Exception as e:
                print("⚠️ 오류 발생:", e, url, i.get("bidNtceDtlUrl", ""))
            table_dict["bidNtceNo"] = i.get("bidNtceNo")
            table_dict["bidNtceNm"] = i.get("bidNtceNm")
            table_dict["bsnsDivNm"] = i.get("bsnsDivNm")
            table_dict["contents"] = re_text
            table.append(table_dict)
        return pd.DataFrame(table)
        # return table

# ====== 텍스트 임베딩 및 정리 ======

# OpenAI API 키 설정 (환경변수 또는 직접 입력 가능)
openai.api_key = os.getenv("OPENAI_API_KEY")

# 벡터 임베딩 함수 (OpenAI text-embedding-3-small 모델 사용)
def get_embedding(text):
    res = openai.Embedding.create(input=[text], model="text-embedding-3-small")
    return res["data"][0]["embedding"]

# 텍스트 추출기 초기화
extractor = DocumentTextExtractor()

# 데이터프레임 생성
info = nara_api()
df = extractor.process_info_list(info)

# ======GPT기반 QA생성기======
def generate_qa_gpt(chunk):
    prompt = f"""
당신은 나라장터 입찰 공고문 전문 분석가입니다. 정부 조달 및 공공기관 입찰에 대한 깊은 전문 지식을 보유하고 있으며,
복잡한 공고문을 체계적으로 분석하여 사업자가 이해하기 쉽게 핵심 정보를 제공합니다.

주어진 "문단"을 분석하여 다음 작업을 수행해주세요:
1.  문단이 무슨 내용을 설명하는지 나타내는 'keyword'를 단어로 정의합니다.
2.  문단의 내용에 기반하여, 사용자가 궁금해할 만한 다양한 질문과 그에 대한 답변 쌍들을 'qa_pairs' 리스트 형태로 작성합니다.
    - 각 질문은 문단 내의 구체적인 정보를 바탕으로 해야 합니다.
    - 각 답변은 해당 질문에 대해 문단에서 찾을 수 있는 정확한 정보를 제공해야 합니다.
    - 문단에서 추출할 수 있는 정보가 있다면, 가능한 한 많은 질문-답변 쌍을 만들어주세요.

문단: "{chunk}"

**⚠️ 반드시 아래 JSON 형식만 출력하세요. 설명, 주석, 마크다운 등의 추가 텍스트는 절대 포함하지 마세요.**
**⚠️ JSON 블록을 감싸는 ```json 등의 Markdown 코드블록을 절대 사용하지 마세요. JSON 객체만 출력하세요.**


JSON 형식 예시:
{{
  "keyword": "주제어",
  "qa_pairs": [
    {{
      "question": "첫 번째 예상 질문",
      "answer": "첫 번째 질문에 대한 답변"
    }},
    {{
      "question": "두 번째 예상 질문",
      "answer": "두 번째 질문에 대한 답변"
    }}
  ]
}}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # 안전한 JSON 파싱 (오류 발생 시 예외 처리)
    content = response['choices'][0]['message']['content']
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("❌ JSON 파싱 실패\n응답:", content)
        raise e

# DB 연결
chunk_id_counter = 1  # 문단 기준 ID 수동 증가

for _, row in tqdm(df.iterrows(), total=len(df)):
    sentence = row["contents"]
    qa_result = generate_qa_gpt(sentence)
    keyword = qa_result["keyword"]
    qa_pairs = qa_result["qa_pairs"]

    metadata_base = {
        "chunk_id": chunk_id_counter,
        "bidNtceNo": row.get("bidNtceNo", "unknown"),
        "bidNtceNm": row.get("bidNtceNm", "unknown"),
        "bsnsDivNm": row.get("bsnsDivNm", "unknown")
    }

    # sentence 저장
    s_embedding = get_embedding(sentence)
    cur.execute("""
        INSERT INTO semantic_chunks (type, text, embedding, metadata)
        VALUES (%s, %s, %s, %s)
    """, ("sentence", sentence, s_embedding, json.dumps(metadata_base)))

    # keyword 저장
    k_embedding = get_embedding(keyword)
    cur.execute("""
        INSERT INTO semantic_chunks (type, text, embedding, metadata)
        VALUES (%s, %s, %s, %s)
    """, ("keyword", keyword, k_embedding, json.dumps(metadata_base)))

    # question/answer 저장
    for qa in qa_pairs:
        q_embedding = get_embedding(qa["question"])
        a_embedding = get_embedding(qa["answer"])

        cur.execute("""
            INSERT INTO semantic_chunks (type, text, embedding, metadata)
            VALUES (%s, %s, %s, %s)
        """, ("question", qa["question"], q_embedding, json.dumps(metadata_base)))

        cur.execute("""
            INSERT INTO semantic_chunks (type, text, embedding, metadata)
            VALUES (%s, %s, %s, %s)
        """, ("answer", qa["answer"], a_embedding, json.dumps(metadata_base)))

    chunk_id_counter += 1

conn.commit()
cur.close()
conn.close()