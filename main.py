import streamlit as st
import pandas as pd
from supabase import create_client, Client
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import numpy as np
from typing import List, Dict, Tuple, TypedDict, Optional, Annotated
import operator
import time

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Union

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# API 및 웹 요청 관련
from urllib.parse import urlencode, quote_plus
import requests

# ========== 초기 설정 및 API 키 ==========
# API 키 및 설정값 초기화
OPENAI_API_KEY = ""  # OpenAI API 키
SUPABASE_URL = ""    # 예: "https://your-project.supabase.co"
SUPABASE_ANON_KEY = ""  # Supabase Anonymous Key
SUPABASE_SERVICE_ROLE_KEY = ""  # Supabase Service Role Key (관리자 권한)

# 나라장터 API 키
NARATANG_API_KEY = "6FAWdycqkHj1fAb/TpeNQLlEzjIB+7eozDneMjTwZPUWDmva0FamSPT1uGtzrxVKuub/vADLVft2bCZ+hkL5YA=="

# ========== Supabase 클라이언트 초기화 ==========
@st.cache_resource
def init_supabase_client():
    """Supabase 클라이언트 초기화"""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.error("Supabase URL과 API 키를 설정해주세요.")
        return None
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        return supabase
    except Exception as e:
        st.error(f"Supabase 연결 실패: {e}")
        return None

# ========== 임베딩 및 벡터 스토어 초기화 ==========
@st.cache_resource
def init_embeddings():
    """OpenAI 임베딩 초기화"""
    if not OPENAI_API_KEY:
        st.error("OpenAI API 키가 설정되지 않았습니다. 임베딩 기능을 사용할 수 없습니다.")
        return None
    
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

@st.cache_resource
def init_vector_store():
    """Supabase 벡터 스토어 초기화"""
    supabase = init_supabase_client()
    embeddings = init_embeddings()
    
    if not supabase or not embeddings:
        return None
    
    try:
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="semantic_chunks",
            query_name="match_documents"
        )
        return vector_store
    except Exception as e:
        st.error(f"벡터 스토어 초기화 실패: {e}")
        return None

# ========== OpenAI 클라이언트 초기화 ==========
@st.cache_resource
def init_openai_client():
    """OpenAI 클라이언트 초기화"""
    if not OPENAI_API_KEY:
        st.warning("OpenAI API 키가 설정되지 않았습니다. GPT 기능이 제한됩니다.")
        return None
    
    return OpenAI(api_key=OPENAI_API_KEY)

# ========== LangChain ChatOpenAI 초기화 ==========
@st.cache_resource
def init_langchain_llm():
    """LangChain ChatOpenAI 초기화"""
    if not OPENAI_API_KEY:
        return None
    
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=OPENAI_API_KEY
    )

# ========== 데이터 액세스 함수들 ==========
@st.cache_data(ttl=300)  # 5분 캐시
def get_live_bids_data():
    """실시간 입찰 공고 데이터 조회"""
    supabase = init_supabase_client()
    if not supabase:
        return pd.DataFrame()
    
    try:
        # 최근 30일 데이터만 조회 (raw JSONB에서 bidNtceDate 추출)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        response = supabase.table('bids_live').select(
            "bidNtceNo, raw, created_at, updated_at"
        ).gte('raw->>bidNtceDate', thirty_days_ago).order('raw->>bidNtceDate', desc=True).execute()
        
        if response.data:
            # JSONB 데이터를 DataFrame으로 변환
            processed_data = []
            for item in response.data:
                bid_data = {
                    'bidNtceNo': item['bidNtceNo'],
                    **item['raw']  # raw JSONB 데이터를 펼쳐서 추가
                }
                processed_data.append(bid_data)
            
            df = pd.DataFrame(processed_data)
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"입찰 공고 데이터 조회 실패: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)  # 10분 캐시
def get_bid_summary(bid_no: str):
    """특정 공고의 GPT 요약 조회"""
    supabase = init_supabase_client()
    if not supabase:
        return "요약 정보를 가져올 수 없습니다.", None, None
    
    try:
        # hwp_based 우선, 없으면 basic_info
        response = supabase.table('bid_summaries').select(
            "summary, summary_type, created_at"
        ).eq('bidNtceNo', bid_no).order('summary_type', desc=True).limit(1).execute()
        
        if response.data:
            summary_doc = response.data[0]
            summary = summary_doc.get('summary', '요약이 없습니다.')
            summary_type = summary_doc.get('summary_type', 'unknown')
            created_at = summary_doc.get('created_at', '')
            
            if created_at:
                created_at = pd.to_datetime(created_at).strftime("%Y-%m-%d %H:%M")
            
            return summary, created_at, summary_type
        else:
            return "이 공고에 대한 요약이 아직 생성되지 않았습니다.", None, None
            
    except Exception as e:
        return f"요약 조회 중 오류 발생: {e}", None, None

def search_bids_by_text(query: str, limit: int = 50):
    """텍스트 기반 입찰 공고 검색 (JSONB 컬럼 검색)"""
    supabase = init_supabase_client()
    if not supabase:
        return []
    
    try:
        # PostgreSQL JSONB 연산자를 사용한 텍스트 검색
        response = supabase.table('bids_live').select("bidNtceNo, raw").or_(
            f"raw->>bidNtceNm.ilike.%{query}%,"
            f"raw->>ntceInsttNm.ilike.%{query}%,"
            f"raw->>dmndInsttNm.ilike.%{query}%,"
            f"raw->>bidprcPsblIndstrytyNm.ilike.%{query}%"
        ).limit(limit).execute()
        
        # 결과 처리: JSONB raw 데이터를 펼쳐서 반환
        processed_results = []
        if response.data:
            for item in response.data:
                bid_data = {
                    'bidNtceNo': item['bidNtceNo'],
                    **item['raw']  # raw JSONB 데이터를 펼쳐서 추가
                }
                processed_results.append(bid_data)
        
        return processed_results
        
    except Exception as e:
        st.error(f"텍스트 검색 중 오류: {e}")
        return []

def get_bid_detail_by_no(bid_no: str):
    """공고번호로 상세 정보 조회"""
    supabase = init_supabase_client()
    if not supabase:
        return None
    
    try:
        response = supabase.table('bids_live').select(
            "bidNtceNo, raw, created_at, updated_at"
        ).eq('bidNtceNo', bid_no).single().execute()
        
        if response.data:
            # JSONB raw 데이터를 펼쳐서 반환
            bid_data = {
                'bidNtceNo': response.data['bidNtceNo'],
                'created_at': response.data.get('created_at'),
                'updated_at': response.data.get('updated_at'),
                **response.data['raw']  # raw JSONB 데이터를 펼쳐서 추가
            }
            return bid_data
        else:
            return None
            
    except Exception as e:
        st.error(f"공고 상세 조회 실패: {e}")
        return None

def search_semantic_chunks(query: str, similarity_threshold: float = 0.3, limit: int = 30):
    """시맨틱 검색 (벡터 유사도 검색)"""
    vector_store = init_vector_store()
    if not vector_store:
        return []
    
    try:
        # 벡터 유사도 검색
        docs = vector_store.similarity_search_with_score(query, k=limit)
        
        results = []
        for doc, score in docs:
            similarity = 1 / (1 + score)  # 거리를 유사도로 변환
            
            if similarity >= similarity_threshold:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": similarity,
                    "bidNtceNo": doc.metadata.get('공고번호', 'N/A'),
                    "bidNtceNm": doc.metadata.get('공고명', 'N/A'),
                    "ntceInsttNm": doc.metadata.get('기관명', 'N/A')
                })
        
        return results
        
    except Exception as e:
        st.error(f"시맨틱 검색 중 오류: {e}")
        return []

# ========== 유틸리티 함수들 ==========
def convert_to_won_format(amount):
    """금액을 한국 원화 형식으로 변환"""
    try:
        if not amount or pd.isna(amount):
            return "공고 참조"
        
        amount = float(str(amount).replace(",", ""))

        if amount >= 100000000:  # 1억 이상
            amount_in_100m = amount / 100000000
            return f"{amount_in_100m:.1f}억원"
        elif amount >= 10000:  # 1만 이상
            amount_in_10k = amount / 10000
            return f"{amount_in_10k:.1f}만원"
        else:
            return f"{int(amount):,}원"
        
    except Exception as e:
        return "공고 참조"

def format_won(amount):
    """금액 포맷팅"""
    try:
        if isinstance(amount, str):
            amount = amount.replace(",", "")
        amount = int(float(amount))
        return f"{amount:,}원"
    except (ValueError, TypeError):
        return "공고 참조"

def format_joint_contract(value):
    """공동수급 정보 포맷팅"""
    if value and str(value).strip():
        return f"허용 [{str(value).strip()}]"
    return "공고서 참조"

# ========== 나라장터 API 설정 ==========
BASE_URL_COMMON = "http://apis.data.go.kr/1230000/ad/BidPublicInfoService"
API_ENDPOINTS = {
    "공사": f"{BASE_URL_COMMON}/getBidPblancListInfoCnstwk",
    "용역": f"{BASE_URL_COMMON}/getBidPblancListInfoServc", 
    "물품": f"{BASE_URL_COMMON}/getBidPblancListInfoThng",
    "외자": f"{BASE_URL_COMMON}/getBidPblancListInfoFrgcpt",
}

# ========== 챗봇 클래스 (Supabase 버전) ==========
class BidSearchChatbot:
    """입찰 공고 검색 챗봇 클래스 - Supabase 버전"""
    
    def __init__(self):
        """챗봇 초기화"""
        self.supabase = init_supabase_client()
        self.vector_store = init_vector_store()
        self.openai_client = init_openai_client()
        self.embeddings = init_embeddings()

    def search_vector_db(self, query: str, n_results: int = 10) -> List[Dict]:
        """벡터 DB에서 관련 문서 검색"""
        if not self.vector_store:
            st.error("벡터 스토어가 초기화되지 않았습니다.")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=n_results)
            
            documents = []
            for doc, score in results:
                # 시맨틱 검색 결과를 Supabase에서 상세 정보와 매칭
                bid_no = doc.metadata.get('공고번호', '')
                if bid_no:
                    bid_detail = get_bid_detail_by_no(bid_no)
                    if bid_detail:
                        document = {
                            "content": doc.page_content,
                            "metadata": {**doc.metadata, **bid_detail},
                            "score": 1 / (1 + score),  # 거리를 유사도로 변환
                            "bidNtceNo": bid_no
                        }
                        documents.append(document)
            
            return documents
        except Exception as e:
            st.error(f"검색 오류: {e}")
            return []

    def search_supabase_text(self, query: str, n_results: int = 10) -> List[Dict]:
        """Supabase에서 텍스트 기반 검색"""
        if not self.supabase:
            return []
        
        try:
            # JSONB 필드에서 텍스트 검색
            results = search_bids_by_text(query, limit=n_results)
            
            documents = []
            for item in results:
                document = {
                    "content": f"공고명: {item.get('bidNtceNm', '')}\n기관: {item.get('ntceInsttNm', '')}\n분류: {item.get('bsnsDivNm', '')}",
                    "metadata": item,
                    "score": 0.8,  # 텍스트 매칭 기본 점수
                    "bidNtceNo": item.get('bidNtceNo', '')
                }
                documents.append(document)
            
            return documents
        except Exception as e:
            st.error(f"Supabase 텍스트 검색 오류: {e}")
            return []

    def get_combined_search_results(self, query: str, n_results: int = 10) -> List[Dict]:
        """벡터 검색과 텍스트 검색 결과를 결합"""
        # 벡터 검색
        vector_results = self.search_vector_db(query, n_results//2)
        
        # 텍스트 검색
        text_results = self.search_supabase_text(query, n_results//2)
        
        # 결과 결합 및 중복 제거
        combined_dict = {}
        
        # 벡터 검색 결과 추가
        for doc in vector_results:
            bid_no = doc.get('bidNtceNo', '')
            if bid_no:
                combined_dict[bid_no] = doc
        
        # 텍스트 검색 결과 추가 (중복되면 점수 증가)
        for doc in text_results:
            bid_no = doc.get('bidNtceNo', '')
            if bid_no:
                if bid_no in combined_dict:
                    combined_dict[bid_no]['score'] += 0.3  # 중복 발견 시 점수 증가
                    combined_dict[bid_no]['source'] = 'vector+text'
                else:
                    doc['source'] = 'text'
                    combined_dict[bid_no] = doc
        
        # 점수 순으로 정렬
        results = list(combined_dict.values())
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return results[:n_results]

    def get_simple_response(self, question: str, search_results: List[Dict]) -> str:
        """간단한 응답 생성"""
        if search_results:
            # 가장 관련성이 높은 첫 번째 검색 결과를 선택
            best_result = search_results[0]
            metadata = best_result.get("metadata", {})
            bid_name = metadata.get("bidNtceNm", "공고명 없음")
            org_name = metadata.get("ntceInsttNm", "기관명 없음")
            return f"질문: {question}\n\n가장 관련성 높은 공고:\n- 공고명: {bid_name}\n- 기관: {org_name}"
        else:
            return "관련된 공고를 찾을 수 없습니다."

    def get_gpt_response(self, question: str, search_results: List[Dict]) -> str:
        """GPT를 사용한 응답 생성"""
        if not search_results:
            return "관련된 공고를 찾을 수 없습니다."
        
        if not self.openai_client:
            return self.get_simple_response(question, search_results)
        
        # 컨텍스트 구성
        context_parts = []
        for i, doc in enumerate(search_results[:5]):  # 상위 5개만 사용
            metadata = doc.get('metadata', {})
            context_parts.append(f"""
        [문서 {i+1}] (유사도: {doc.get('score', 0):.2f})
        공고번호: {metadata.get('bidNtceNo', 'N/A')}
        공고명: {metadata.get('bidNtceNm', 'N/A')}
        기관명: {metadata.get('ntceInsttNm', 'N/A')}
        분류: {metadata.get('bsnsDivNm', 'N/A')}
        금액: {convert_to_won_format(metadata.get('asignBdgtAmt', 0))}
        내용: {doc.get('content', '')[:300]}...
        """)
        
        context = "\n".join(context_parts)
        
        system_prompt = """당신은 공공입찰 정보 검색 도우미입니다.
        사용자의 질문에 대해 제공된 검색 결과를 바탕으로 답변해주세요.

        답변 지침:
        1. 검색된 공고들 중 가장 관련성 높은 것들을 우선적으로 언급
        2. 각 공고의 핵심 정보(공고명, 기관, 마감일 등)를 간결하게 정리
        3. 사용자가 추가로 확인해야 할 사항이 있다면 안내
        4. 검색 결과가 충분하지 않다면 다른 검색어를 제안
        """
        
        user_prompt = f"""
        사용자 질문: {question}

검색된 관련 공고:
{context}

위 검색 결과를 바탕으로 사용자 질문에 답변해주세요.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"응답 생성 중 오류 발생: {e}"

# ========== 시맨틱 검색 함수들 (Supabase 버전) ==========
def semantic_search_supabase(query: str, k: int = 10) -> List[Tuple[Dict, float]]:
    """
    Supabase 벡터 DB에서 시맨틱 검색 수행
    """
    vector_store = init_vector_store()
    if not vector_store:
        st.error("벡터 스토어가 초기화되지 않았습니다.")
        return []
    
    try:
        # 유사도 검색 수행
        results = vector_store.similarity_search_with_score(query, k=k)
        
        # 결과 정리 및 Supabase에서 상세 정보 조회
        formatted_results = []
        for doc, score in results:
            # 벡터 검색 결과의 메타데이터에서 공고번호 추출
            bid_no = doc.metadata.get('공고번호', '')
            
            if bid_no:
                # Supabase에서 해당 공고의 전체 정보 조회
                bid_detail = get_bid_detail_by_no(bid_no)
                if bid_detail:
                    # 점수를 0-1 범위로 정규화 (낮을수록 유사)
                    similarity_score = 1 / (1 + score)
                    
                    # 메타데이터 결합
                    combined_metadata = {**doc.metadata, **bid_detail}
                    formatted_results.append((combined_metadata, similarity_score))
        
        return formatted_results
    except Exception as e:
        st.error(f"시맨틱 검색 중 오류 발생: {e}")
        return []

def generate_rag_response_supabase(query: str, context_docs: List[Tuple[Dict, float]]) -> str:
    """검색된 문서를 기반으로 RAG 응답 생성 - Supabase 버전"""
    
    openai_client = init_openai_client()
    if not openai_client:
        return "OpenAI 클라이언트가 초기화되지 않았습니다."
    
    # 컨텍스트 구성
    context_parts = []
    for i, (metadata, score) in enumerate(context_docs[:5]):  # 상위 5개만 사용
        context_parts.append(f"""
    [문서 {i+1}] (유사도: {score:.2f})
    공고번호: {metadata.get('bidNtceNo', 'N/A')}
    공고명: {metadata.get('bidNtceNm', 'N/A')}
    기관명: {metadata.get('ntceInsttNm', 'N/A')}
    분류: {metadata.get('bsnsDivNm', 'N/A')}
    금액: {convert_to_won_format(metadata.get('asignBdgtAmt', 0))}
    마감일: {metadata.get('bidClseDate', 'N/A')}
""")
    
    context = "\n".join(context_parts)
    
    system_prompt = """당신은 공공입찰 정보 검색 도우미입니다.
    사용자의 질문에 대해 제공된 검색 결과를 바탕으로 답변해주세요.

    답변 지침:
    1. 검색된 공고들 중 가장 관련성 높은 것들을 우선적으로 언급
    2. 각 공고의 핵심 정보(공고명, 기관, 마감일 등)를 간결하게 정리
    3. 사용자가 추가로 확인해야 할 사항이 있다면 안내
    4. 검색 결과가 충분하지 않다면 다른 검색어를 제안
    """
    
    user_prompt = f"""
    사용자 질문: {query}

    검색된 관련 공고:
    {context}

    위 검색 결과를 바탕으로 사용자 질문에 답변해주세요.
    """
        
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"응답 생성 중 오류 발생: {e}"

# ========== 초기화 함수들 (캐시 적용) ==========
@st.cache_resource
def init_chatbot():
    """챗봇 인스턴스를 초기화하고 캐싱"""
    return BidSearchChatbot()

@st.cache_resource
def init_resources():
    """리소스 초기화 - Supabase 버전"""
    # Supabase 연결
    supabase = init_supabase_client()
    
    # 벡터 DB 초기화
    vector_store = init_vector_store()
    
    # LangChain LLM
    llm = init_langchain_llm()
    
    return supabase, vector_store, llm

# ========== 챗봇 기능 ==========
def process_question(question: str, chatbot: BidSearchChatbot):
    """질문 처리 및 응답 생성"""
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_messages.append({"role": "user", "content": question})
    
    # 응답 생성
    with st.spinner("입찰 공고를 검색하고 분석 중입니다..."):
        # 하이브리드 검색 (벡터 + 텍스트)
        search_results = chatbot.get_combined_search_results(question)
        
        # GPT 응답 또는 간단한 응답
        if OPENAI_API_KEY:
            response = chatbot.get_gpt_response(question, search_results)
        else:
            response = chatbot.get_simple_response(question, search_results)
    
    # 응답 표시
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_messages.append({"role": "assistant", "content": response})

    # ========== LangGraph 상태 정의 (Supabase 버전) ==========
class BidSearchState(TypedDict):
    """입찰 공고 검색 상태 - Supabase 버전"""
    query: str
    search_type: str  # 'semantic' or 'keyword'
    category_filter: Optional[List[str]]
    date_range: Optional[Tuple[datetime, datetime]]
    supabase_results: List[Dict]  # MongoDB -> Supabase
    vector_results: List[Dict]
    combined_results: List[Dict]
    final_answer: str
    error: Optional[str]
    status_messages: Annotated[List[str], operator.add]
    quality_score: float

class HybridSearchState(TypedDict):
    """하이브리드 검색 프로세스의 상태 - Supabase 버전"""
    query: str
    search_method: List[str]  # 사용된 검색 방법들
    supabase_results: List[Dict]  # MongoDB -> Supabase
    vector_results: List[Dict]
    api_results: Dict[str, List[Dict]]
    combined_results: List[Dict]
    final_results: List[Dict]
    summary: str
    messages: Annotated[List[Union[HumanMessage, AIMessage]], operator.add]
    error: Optional[str]
    total_count: int
    need_api_search: bool

# ========== LangGraph 노드 함수들 (Supabase 버전) ==========
def preprocess_query_node(state: BidSearchState) -> BidSearchState:
    """쿼리 전처리 노드"""
    try:
        # 쿼리 확장 및 동의어 처리
        query = state["query"].lower()
        
        # 입찰 관련 키워드 확장
        keyword_expansions = {
            "ai": ["인공지능", "AI", "머신러닝", "딥러닝"],
            "서버": ["서버", "서버구축", "서버시스템", "인프라"],
            "sw": ["소프트웨어", "SW", "S/W", "프로그램"],
            "hw": ["하드웨어", "HW", "H/W", "장비"],
            "시스템": ["시스템", "시스템구축", "정보시스템"],
            "개발": ["개발", "구축", "제작", "개발사업"]
        }
        
        expanded_terms = [query]
        for key, synonyms in keyword_expansions.items():
            if key in query:
                expanded_terms.extend(synonyms)
        
        state["expanded_query"] = " ".join(set(expanded_terms))
        state["status_messages"] = [f"✅ 쿼리 전처리 완료: {len(expanded_terms)}개 검색어"]
        
    except Exception as e:
        state["error"] = f"쿼리 전처리 오류: {str(e)}"
        
    return state

def search_supabase_node(state: BidSearchState) -> BidSearchState:
    """Supabase 검색 노드"""
    try:
        supabase = init_supabase_client()
        if not supabase:
            state["error"] = "Supabase 연결 실패"
            state["supabase_results"] = []
            return state
        
        # 검색 쿼리 구성
        query_text = state.get("expanded_query", state["query"])
        
        # 최근 30일 데이터만 검색
        from datetime import datetime, timedelta
        date_30_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        # JSONB 필드에서 텍스트 검색
        response = supabase.table('bids_live').select("bidNtceNo, raw").or_(
            f"raw->>bidNtceNm.ilike.%{query_text}%,"
            f"raw->>ntceInsttNm.ilike.%{query_text}%,"
            f"raw->>dmndInsttNm.ilike.%{query_text}%,"
            f"raw->>bidprcPsblIndstrytyNm.ilike.%{query_text}%"
        ).gte('raw->>bidNtceDate', date_30_days_ago).limit(50).execute()
        
        # 결과 처리
        results = []
        if response.data:
            for item in response.data:
                bid_data = {
                    'bidNtceNo': item['bidNtceNo'],
                    **item['raw']  # raw JSONB 데이터를 펼쳐서 추가
                }
                results.append(bid_data)
        
        # 카테고리 필터
        if state.get("category_filter"):
            results = [r for r in results if r.get('bsnsDivNm') in state["category_filter"]]
        
        # 날짜 필터
        if state.get("date_range"):
            start_date, end_date = state["date_range"]
            filtered_results = []
            for r in results:
                bid_date = r.get('bidNtceDate', '')
                if bid_date and start_date.strftime('%Y%m%d') <= bid_date <= end_date.strftime('%Y%m%d'):
                    filtered_results.append(r)
            results = filtered_results
        
        state["supabase_results"] = results
        state["status_messages"] = [f"✅ Supabase에서 {len(results)}개 공고 검색 완료"]
        
    except Exception as e:
        state["error"] = f"Supabase 검색 오류: {str(e)}"
        state["supabase_results"] = []
        
    return state

def search_vector_db_node(state: BidSearchState) -> BidSearchState:
    """벡터 DB 시맨틱 검색 노드"""
    try:
        vector_store = init_vector_store()
        if not vector_store:
            state["error"] = "벡터 스토어 초기화 실패"
            state["vector_results"] = []
            return state
        
        # 벡터 검색 실행
        results = vector_store.similarity_search_with_score(
            state["query"], 
            k=30
        )
        
        # 결과 포맷팅
        vector_results = []
        for doc, score in results:
            metadata = doc.metadata
            similarity_score = 1 / (1 + score)  # 거리를 유사도로 변환
            
            # Supabase에서 상세 정보 조회
            bid_no = metadata.get('공고번호', '')
            if bid_no:
                bid_detail = get_bid_detail_by_no(bid_no)
                if bid_detail:
                    vector_results.append({
                        "content": doc.page_content,
                        "metadata": {**metadata, **bid_detail},
                        "similarity": similarity_score,
                        "bidNtceNo": bid_no,
                        "bidNtceNm": bid_detail.get('bidNtceNm', metadata.get('공고명', 'N/A')),
                        "ntceInsttNm": bid_detail.get('ntceInsttNm', metadata.get('기관명', 'N/A'))
                    })
        
        # 유사도 임계값 필터링
        filtered_results = [r for r in vector_results if r["similarity"] >= 0.3]
        
        state["vector_results"] = filtered_results
        state["status_messages"] = [f"✅ 벡터 DB에서 {len(filtered_results)}개 관련 문서 검색 완료"]
        
    except Exception as e:
        state["error"] = f"벡터 검색 오류: {str(e)}"
        state["vector_results"] = []
        
    return state

def combine_results_node(state: BidSearchState) -> BidSearchState:
    """검색 결과 통합 노드"""
    try:
        supabase_results = state.get("supabase_results", [])
        vector_results = state.get("vector_results", [])
        combined_dict = {}

        # Supabase 결과 추가
        for supabase_item in supabase_results:
            bid_no = supabase_item.get("bidNtceNo")
            if bid_no:
                combined_dict[bid_no] = {
                    **supabase_item,
                    "source": "supabase",
                    "relevance_score": 0.5
                }

        # 벡터 결과 추가/병합
        for vector_item in vector_results:
            bid_no = vector_item.get("bidNtceNo")
            if bid_no:
                if bid_no in combined_dict:
                    combined_dict[bid_no]["relevance_score"] += vector_item["similarity"]
                    combined_dict[bid_no]["vector_content"] = vector_item["content"]
                    combined_dict[bid_no]["similarity"] = vector_item["similarity"]
                    combined_dict[bid_no]["source"] = "supabase+vector"
                else:
                    combined_dict[bid_no] = {
                        **vector_item["metadata"],
                        "source": "vector",
                        "relevance_score": vector_item["similarity"],
                        "vector_content": vector_item["content"],
                        "similarity": vector_item["similarity"]
                    }

        combined_results = list(combined_dict.values())
        combined_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        state["combined_results"] = combined_results[:20]

        # 검색 품질 점수 계산
        if combined_results:
            avg_score = np.mean([r["relevance_score"] for r in combined_results[:10]])
            state["quality_score"] = round(float(avg_score), 3)
        else:
            state["quality_score"] = 0.0

        state["status_messages"] = [f"✅ {len(state['combined_results'])}개 공고로 통합 완료"]

    except Exception as e:
        state["error"] = f"결과 통합 오류: {str(e)}"
        state["combined_results"] = []
        state["quality_score"] = 0.0

    return state

def enrich_with_summaries_node(state: BidSearchState) -> BidSearchState:
    """Supabase에서 GPT 요약 추가 노드"""
    try:
        supabase = init_supabase_client()
        if not supabase:
            return state
        
        # 각 공고에 대해 요약 정보 추가
        for result in state["combined_results"]:
            bid_no = result.get("bidNtceNo")
            if bid_no:
                # 요약 정보 조회
                summary, created_at, summary_type = get_bid_summary(bid_no)
                result["summary"] = summary if summary != "이 공고에 대한 요약이 아직 생성되지 않았습니다." else ""
                result["summary_type"] = summary_type or "none"
                result["summary_created_at"] = created_at
        
        state["status_messages"] = ["✅ GPT 요약 정보 추가 완료"]
        
    except Exception as e:
        state["error"] = f"요약 정보 추가 오류: {str(e)}"
        
    return state

def generate_answer_node(state: BidSearchState) -> BidSearchState:
    """AI 답변 생성 노드 (RAG 적용) - Supabase 버전"""
    try:
        llm = init_langchain_llm()
        if not llm:
            state["final_answer"] = "AI 답변 생성을 위한 OpenAI 연결이 필요합니다."
            return state

        if not state["combined_results"]:
            state["final_answer"] = f"'{state['query']}'에 대한 입찰 공고를 찾을 수 없습니다."
            return state
        
        # RAG (Retrieval-Augmented Generation) 답변 생성
        contexts = []
        for i, result in enumerate(state["combined_results"][:5]):
            bid_no = result.get("bidNtceNo", "N/A")
            title = result.get("bidNtceNm", "제목 없음")
            org = result.get("ntceInsttNm", "기관명 없음")
            summary = result.get("summary", "요약 없음")
            amount = convert_to_won_format(result.get("asignBdgtAmt", 0))
            source = result.get("source", "unknown")

            context_item = f"""
공고 {i+1} (출처: {source}):
- 공고번호: {bid_no}
- 공고명: {title}
- 기관: {org}
- 금액: {amount}
- 요약: {summary}
"""
            contexts.append(context_item)

        context_text = "\n".join(contexts)

        # LangChain 프롬프트 템플릿을 사용하여 RAG 적용
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 입찰 공고 분석 전문가입니다. 제공된 검색 결과를 바탕으로 정확하고 유용한 답변을 제공해주세요."),
            ("human", f"다음은 입찰 공고 검색 결과입니다:\n\n{context_text}\n\n위 정보를 바탕으로 다음 질문에 답해주세요: {state['query']}")
        ])

        # LangChain 체인 실행
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "query": state["query"],
            "context": context_text
        })

        state["final_answer"] = answer
        state["status_messages"] = ["✅ AI 분석 답변 생성 완료!"]

    except Exception as e:
        state["error"] = f"AI 답변 생성 오류: {str(e)}"
        state["final_answer"] = "답변 생성에 실패했습니다."
    return state

def check_error(state: BidSearchState) -> str:
    """에러 체크 노드"""
    if state.get("error"):
        return "error"
    return "continue"

# ========== LangGraph 워크플로우 구성 ==========
def create_bid_search_workflow():
    """입찰 공고 검색 워크플로우 생성 - Supabase 버전"""
    workflow = StateGraph(BidSearchState)
    
    # 노드 추가
    workflow.add_node("preprocess_query", preprocess_query_node)
    workflow.add_node("search_supabase", search_supabase_node)  # MongoDB -> Supabase
    workflow.add_node("search_vector_db", search_vector_db_node)
    workflow.add_node("combine_results", combine_results_node)
    workflow.add_node("enrich_summaries", enrich_with_summaries_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    # 엣지 추가
    workflow.set_entry_point("preprocess_query")
    
    # 조건부 엣지
    workflow.add_conditional_edges(
        "preprocess_query",
        check_error,
        {
            "continue": "search_supabase",
            "error": END
        }
    )
    
    # 병렬 검색 후 결합
    workflow.add_edge("search_supabase", "search_vector_db")
    workflow.add_edge("search_vector_db", "combine_results")
    workflow.add_edge("combine_results", "enrich_summaries")
    workflow.add_edge("enrich_summaries", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # 체크포인터 추가
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# ========== LangGraph 하이브리드 검색 노드들 (Supabase 버전) ==========
def search_supabase_hybrid_node(state: HybridSearchState) -> HybridSearchState:
    """Supabase에서 검색 - 하이브리드 버전"""
    query = state["query"]
    
    state["messages"].append(
        HumanMessage(content=f"Supabase 검색 시작: {query}")
    )
    
    try:
        # Supabase 텍스트 검색
        results = search_bids_by_text(query, limit=50)
        
        # 최근 30일 데이터 필터링
        from datetime import datetime, timedelta
        date_30_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        filtered_results = []
        for result in results:
            bid_date = result.get('bidNtceDate', '')
            if bid_date and bid_date >= date_30_days_ago:
                filtered_results.append(result)
        
        state["supabase_results"] = filtered_results
        state["search_method"].append("Supabase")
        state["messages"].append(
            AIMessage(content=f"Supabase에서 {len(filtered_results)}건 검색됨")
        )
        
    except Exception as e:
        state["messages"].append(
            AIMessage(content=f"Supabase 검색 오류: {str(e)}")
        )
        state["supabase_results"] = []
    
    return state

def search_vector_hybrid_node(state: HybridSearchState) -> HybridSearchState:
    """VectorDB에서 시맨틱 검색 - 하이브리드 버전"""
    query = state["query"]
    
    state["messages"].append(
        HumanMessage(content=f"VectorDB 시맨틱 검색 시작")
    )
    
    try:
        vector_store = init_vector_store()
        if not vector_store:
            state["messages"].append(
                AIMessage(content="벡터 스토어 초기화 실패")
            )
            state["vector_results"] = []
            return state
        
        # 벡터 검색
        results = vector_store.similarity_search_with_score(query, k=30)
        
        vector_results = []
        for doc, score in results:
            metadata = doc.metadata
            similarity = 1 / (1 + score)
            
            if similarity >= 0.03:  # 유사도 임계값
                # Supabase에서 상세 정보 조회
                bid_no = metadata.get('공고번호', '')
                if bid_no:
                    bid_detail = get_bid_detail_by_no(bid_no)
                    if bid_detail:
                        vector_results.append({
                            "content": doc.page_content[:500],
                            "metadata": {**metadata, **bid_detail},
                            "similarity": similarity,
                            "bidNtceNo": bid_no,
                            "bidNtceNm": bid_detail.get('bidNtceNm', metadata.get('공고명', 'N/A')),
                            "ntceInsttNm": bid_detail.get('ntceInsttNm', metadata.get('기관명', 'N/A'))
                        })
        
        state["vector_results"] = vector_results
        state["search_method"].append("VectorDB")
        state["messages"].append(
            AIMessage(content=f"VectorDB에서 {len(vector_results)}건 검색됨 (유사도 0.03 이상)")
        )
        
    except Exception as e:
        state["messages"].append(
            AIMessage(content=f"VectorDB 검색 오류: {str(e)}")
        )
        state["vector_results"] = []
    
    return state

def check_need_api_node(state: HybridSearchState) -> HybridSearchState:
    """API 검색이 필요한지 판단"""
    supabase_count = len(state["supabase_results"])
    vector_count = len(state["vector_results"])
    total_count = supabase_count + vector_count
    
    state["messages"].append(
        HumanMessage(content=f"검색 결과 확인: Supabase {supabase_count}건, VectorDB {vector_count}건")
    )
    
    # 결과가 10개 미만이면 API 검색 필요
    if total_count < 10:
        state["need_api_search"] = True
        state["messages"].append(
            AIMessage(content=f"검색 결과 부족 ({total_count}건), API 검색 필요")
        )
    else:
        state["need_api_search"] = False
        state["messages"].append(
            AIMessage(content=f"충분한 검색 결과 ({total_count}건), API 검색 불필요")
        )
    
    return state

def fetch_naratang_api_node(state: HybridSearchState) -> HybridSearchState:
    """나라장터 API 호출 (필요한 경우만)"""
    if not state["need_api_search"]:
        state["api_results"] = {}
        return state
    
    query = state["query"]
    
    state["messages"].append(
        HumanMessage(content=f"나라장터 API 실시간 검색 시작")
    )
    
    # 날짜 설정
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    all_results = {}
    api_total = 0
    
    for category, endpoint in API_ENDPOINTS.items():
        try:
            state["messages"].append(
                AIMessage(content=f"{category} 카테고리 API 호출 중...")
            )
            
            params = {
                'serviceKey': NARATANG_API_KEY,
                'pageNo': 1,
                'numOfRows': 20,
                'inqryDiv': 1,
                'type': 'json',
                'inqryBgnDt': start_date.strftime('%Y%m%d') + '0000',
                'inqryEndDt': end_date.strftime('%Y%m%d') + '2359',
                'bidNtceNm': query
            }
            
            query_string = urlencode(params, quote_via=quote_plus)
            url = f"{endpoint}?{query_string}"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('response', {}).get('body', {}).get('items', [])
                
                if items:
                    all_results[category] = items
                    api_total += len(items)
                    state["messages"].append(
                        AIMessage(content=f"{category}: {len(items)}건 검색 완료")
                    )
                
        except Exception as e:
            state["messages"].append(
                AIMessage(content=f"{category} API 오류: {str(e)}")
            )
    
    state["api_results"] = all_results
    state["search_method"].append("API")
    state["messages"].append(
        AIMessage(content=f"API 검색 완료: 총 {api_total}건")
    )
    
    return state

def combine_hybrid_results_node(state: HybridSearchState) -> HybridSearchState:
    """모든 검색 결과 통합 및 중복 제거 - Supabase 버전"""
    state["messages"].append(
        HumanMessage(content="검색 결과 통합 중...")
    )
    
    combined_dict = {}
    
    # Supabase 결과 추가
    for item in state["supabase_results"]:
        bid_no = item.get("bidNtceNo")
        if bid_no:
            combined_dict[bid_no] = {
                **item,
                "source": "Supabase",
                "relevance_score": 5  # 기본 점수
            }
    
    # VectorDB 결과 추가/병합
    for item in state["vector_results"]:
        bid_no = item.get("bidNtceNo")
        if bid_no:
            if bid_no in combined_dict:
                # 이미 있으면 점수 증가
                combined_dict[bid_no]["relevance_score"] += item["similarity"] * 10
                combined_dict[bid_no]["vector_content"] = item.get("content", "")
                combined_dict[bid_no]["source"] += ", VectorDB"
            else:
                # Supabase에서 추가 정보 조회
                bid_detail = get_bid_detail_by_no(bid_no)
                
                if bid_detail:
                    combined_dict[bid_no] = {
                        **bid_detail,
                        "source": "VectorDB",
                        "relevance_score": item["similarity"] * 10,
                        "vector_content": item.get("content", "")
                    }
    
    # API 결과 추가
    for category, items in state["api_results"].items():
        for item in items:
            bid_no = item.get("bidNtceNo")
            if bid_no:
                if bid_no in combined_dict:
                    combined_dict[bid_no]["relevance_score"] += 3
                    combined_dict[bid_no]["source"] += ", API"
                else:
                    combined_dict[bid_no] = {
                        **item,
                        "source": f"API({category})",
                        "relevance_score": 3,
                        "category": category
                    }
    
    # 리스트로 변환 및 정렬
    combined_results = list(combined_dict.values())
    combined_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    state["combined_results"] = combined_results
    state["total_count"] = len(combined_results)
    
    state["messages"].append(
        AIMessage(content=f"통합 완료: 총 {len(combined_results)}개 (중복 제거됨)")
    )
    
    return state

def generate_hybrid_summary_node(state: HybridSearchState) -> HybridSearchState:
    """AI 요약 생성 - 하이브리드 버전"""
    if not state["combined_results"]:
        state["summary"] = "검색 결과가 없습니다."
        state["final_results"] = []
        return state
    
    state["messages"].append(
        HumanMessage(content="AI 요약 생성 중...")
    )
    
    # 상위 20개만 최종 결과로
    state["final_results"] = state["combined_results"][:20]
    
    # 소스별 통계
    source_stats = {}
    for item in state["final_results"]:
        sources = item.get("source", "").split(", ")
        for source in sources:
            source_stats[source] = source_stats.get(source, 0) + 1
    
    # 요약 생성
    summary_parts = [
        f"🔍 '{state['query']}' 검색 결과: 총 {state['total_count']}건",
        f"\n📊 검색 소스:"
    ]
    
    for source, count in source_stats.items():
        summary_parts.append(f"  • {source}: {count}건")
    
    # 상위 3개 공고 하이라이트
    summary_parts.append(f"\n🏆 상위 공고:")
    for i, item in enumerate(state["final_results"][:3], 1):
        title = item.get('bidNtceNm', '제목 없음')
        org = item.get('ntceInsttNm', 'N/A')
        amount = item.get('asignBdgtAmt', 0)
        summary_parts.append(
            f"{i}. {title[:40]}..."
            f" ({org}, {convert_to_won_format(amount)})"
        )
    
    state["summary"] = "\n".join(summary_parts)
    
    state["messages"].append(
        AIMessage(content="요약 생성 완료")
    )
    
    return state

def should_search_api(state: HybridSearchState) -> str:
    """API 검색 필요 여부 판단"""
    if state["need_api_search"]:
        return "search_api"
    else:
        return "combine"

# ========== 하이브리드 워크플로우 생성 ==========
def create_hybrid_search_workflow():
    """하이브리드 검색 워크플로우 생성 - Supabase 버전"""
    workflow = StateGraph(HybridSearchState)
    
    # 노드 추가
    workflow.add_node("search_supabase", search_supabase_hybrid_node)
    workflow.add_node("search_vector", search_vector_hybrid_node)
    workflow.add_node("check_need_api", check_need_api_node)
    workflow.add_node("search_api", fetch_naratang_api_node)
    workflow.add_node("combine", combine_hybrid_results_node)
    workflow.add_node("generate_summary", generate_hybrid_summary_node)
    
    # 엣지 설정
    workflow.set_entry_point("search_supabase")
    workflow.add_edge("search_supabase", "search_vector")
    workflow.add_edge("search_vector", "check_need_api")
    
    # 조건부 엣지: API 검색 필요 여부
    workflow.add_conditional_edges(
        "check_need_api",
        should_search_api,
        {
            "search_api": "search_api",
            "combine": "combine"
        }
    )
    
    workflow.add_edge("search_api", "combine")
    workflow.add_edge("combine", "generate_summary")
    workflow.add_edge("generate_summary", END)
    
    return workflow.compile()

# ========== UI 탭 함수들 ==========

# tab1: 실시간 입찰 공고
def show_live_bids_tab(df_live):
    """실시간 입찰 공고 탭 UI 구성 - Supabase 버전"""
    st.markdown("""
    <style>
        .main-header-2 {
            text-align: center;
            padding: 3.5rem 0;
            background: linear-gradient(135deg, #ff9a8b 0%, #ff6a88 100%);
            color: white;
            border-radius: 20px;
            margin-bottom: 3rem;
            box-shadow: 0 10px 20px rgba(0,0,0,0.25);
            overflow: hidden;
            position: relative;
            animation: fadeIn 1.5s ease-out;
        }
        .main-header-2::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: rotateBg 20s linear infinite;
            opacity: 0.7;
        }
        .main-header-2 h1 {
            font-size: 3.8rem;
            font-weight: 900;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
            margin-bottom: 0.8rem;
        }
        .main-header-2 p {
            font-size: 1.5rem;
            font-weight: 400;
            opacity: 0.95;
            line-height: 1.6;
            max-width: 900px;
            margin: 0.8rem auto 0 auto;
            position: relative;
            z-index: 1;
        }
        @keyframes rotateBg {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header-2">
        <h1>🚀 당신의 입찰 성공 파트너, AI 입찰 도우미!</h1>
        <p>
            매일 업데이트되는 실시간 공고 확인부터<br>
            인공지능 기반의 정확한 검색과 스마트한 질의응답까지,<br>
            복잡한 입찰 과정을 쉽고 빠르게 경험하세요.
        </p>
    </div>
    """, unsafe_allow_html=True) 
         
    st.subheader("📢 현재 진행 중인 입찰 목록")
    
    # DataFrame이 비어있는 경우 처리
    if df_live.empty:
        st.warning("현재 표시할 입찰 공고가 없습니다.")
        return
    
    # 데이터 준비 - 컬럼명 한글로 변경
    df_live_display = df_live.copy()
    
    # 필요한 컬럼만 선택 (JSONB에서 추출된 데이터)
    display_columns = {
        "bidNtceNo": "공고번호",
        "bidNtceNm": "공고명", 
        "ntceInsttNm": "공고기관",
        "bsnsDivNm": "분류",
        "asignBdgtAmt": "금액",
        "bidNtceDate": "게시일",
        "bidClseDate": "마감일",
        "bidClseTm": "마감시간",
        "bidNtceUrl": "url",
        "bidNtceBgn": "게시시간",
        "bidNtceSttusNm": "입찰공고상태명",
        "dmndInsttNm": "수요기관",
        "bidprcPsblIndstrytyNm": "투찰가능업종명",
        "cmmnReciptMethdNm": "공동수급",
        "rgnLmtYn": "지역제한",
        "prtcptPsblRgnNm": "참가가능지역명",
        "presmptPrce": "추정가격"
    }
    
    # 컬럼명 변경
    available_columns = [col for col in display_columns.keys() if col in df_live_display.columns]
    df_live_display = df_live_display[available_columns]
    df_live_display.columns = [display_columns[col] for col in available_columns]

    # 날짜 형식 변환
    if "마감일" in df_live_display.columns:
        df_live_display["마감일"] = pd.to_datetime(df_live_display["마감일"], errors='coerce')
    if "게시일" in df_live_display.columns:
        df_live_display["게시일"] = pd.to_datetime(df_live_display["게시일"], errors='coerce')
    
    df_live_display = df_live_display.sort_values(by=['게시일','게시시간'], ascending=False)

    # 필터 UI
    search_keyword = st.text_input("🔎 공고명 또는 공고기관 검색")
    unique_categories = ["공사", "용역", "물품", "외자"]
    selected_cls = st.multiselect("📁 분류 선택", 
                                 options=unique_categories, 
                                 default=unique_categories)

    col2, col3, col4 = st.columns(3)        
    with col2:
        if not df_live_display["게시일"].empty:
            start_date = st.date_input("📅 게시일 기준 시작일", value=df_live_display["게시일"].min().date())
        else:
            start_date = st.date_input("📅 게시일 기준 시작일", value=datetime.now().date())
    with col3:
        if not df_live_display["게시일"].empty:
            end_date = st.date_input("📅 게시일 기준 종료일", value=df_live_display["게시일"].max().date())
        else:
            end_date = st.date_input("📅 게시일 기준 종료일", value=datetime.now().date())
    with col4:
        sort_col = st.selectbox("정렬기준",options=["실시간","게시일","마감일","금액"])
        if sort_col == "실시간" :
            sort_order = "내림차순"
            st.empty()
        else :
            sort_order = st.radio("정렬 방향", options=["오름차순", "내림차순"], horizontal=True,
                                label_visibility="collapsed")
   
    # 필터링 적용
    filtered = df_live_display.copy()

    if selected_cls and "분류" in filtered.columns:
        filtered = filtered[filtered["분류"].isin(selected_cls)]

    if search_keyword:
        mask = pd.Series([False] * len(filtered))
        if "공고명" in filtered.columns:
            mask |= filtered["공고명"].str.contains(search_keyword, case=False, na=False, regex=False)
        if "공고기관" in filtered.columns:
            mask |= filtered["공고기관"].str.contains(search_keyword, case=False, na=False, regex=False)
        if "공고번호" in filtered.columns:
            mask |= filtered["공고번호"].str.contains(search_keyword, case=False, na=False, regex=False)
        filtered = filtered[mask]

    if "게시일" in filtered.columns:
        filtered = filtered[
            (filtered["게시일"].dt.date >= start_date) & 
            (filtered["게시일"].dt.date <= end_date)
        ]

    # 정렬 적용
    ascending = True if sort_order == "오름차순" else False

    if sort_col == "실시간" and "게시일" in filtered.columns:
        filtered = filtered.sort_values(by=["게시일", "게시시간"], ascending=False)
    elif sort_col == "게시일" and "게시일" in filtered.columns:
        filtered = filtered.sort_values(by=["게시일", "게시시간"], ascending=ascending)
    elif sort_col == "마감일" and "마감일" in filtered.columns:
        filtered = filtered.sort_values(by="마감일", ascending=ascending)
    elif sort_col == "금액" and "금액" in filtered.columns:
        filtered = filtered.sort_values(by="금액", ascending=ascending)

    st.markdown(f"<div style='text-align: left; margin-bottom: 10px;'>검색 결과 {len(filtered)}건</div>", unsafe_allow_html=True)

    # 페이지네이션
    PAGE_SIZE = 10
    def paginate_dataframe(df, page_num, page_size):
        start_index = page_num * page_size
        end_index = (page_num + 1) * page_size
        return df.iloc[start_index:end_index]

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = 0

    total_pages = (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE
    paginated_df = paginate_dataframe(filtered, st.session_state["current_page"], PAGE_SIZE)
    
    st.write("")
    st.write("")
    
    # 테이블 헤더
    header_cols = st.columns([1,1.5, 3.5, 2.5, 1, 1.5,1.5, 1.5, 1])
    headers = ['공고번호',"구분",'공고명','공고기관','분류','금액','게시일','마감일','상세정보']
    for col, head in zip(header_cols, headers):
        col.markdown(f"**{head}**")

    st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)

    # 행 렌더링
    for i, (idx, row) in enumerate(paginated_df.iterrows()):
        cols = st.columns([1,1.5, 3.5, 2.5, 1, 1.5,1.5, 1.5, 1])
        cols[0].write(row.get("공고번호", ""))
        cols[1].write(row.get("입찰공고상태명", ""))
        cols[2].markdown(row.get("공고명", ""))
        cols[3].write(row.get("공고기관", ""))
        cols[4].write(row.get("분류", ""))            
        
        # 금액 표시 (추정가격이 있으면 추정가격, 없으면 금액)
        if row.get("분류") == "공사" and row.get("추정가격"):
            금액 = row.get("추정가격")
        else:
            금액 = row.get("금액")
        cols[5].write(convert_to_won_format(금액))
        
        if pd.notna(row.get("게시일")):
            cols[6].write(row["게시일"].strftime("%Y-%m-%d"))
        else:
            cols[6].write("")
            
        if pd.notna(row.get("마감일")):
            cols[7].write(row["마감일"].strftime("%Y-%m-%d"))
        else:
            cols[7].write("공고 참조")
            
        if cols[8].button("보기", key=f"live_detail_{i}_{idx}"):
            st.session_state["page"] = "detail"
            st.session_state["selected_live_bid"] = row.to_dict()
            st.rerun()

        st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)

    # 페이지 이동 버튼
    cols_pagination = st.columns([1, 3, 1])
    with cols_pagination[0]:
        if st.session_state["current_page"] > 0:
            if st.button("이전"):
                st.session_state["current_page"] -= 1
                st.rerun()

    with cols_pagination[2]:
        if st.session_state['current_page'] < total_pages -1:
            if st.button("다음"):
                st.session_state["current_page"] += 1
                st.rerun()

    st.markdown(f"<div style='text-align: center;'> {st.session_state['current_page'] + 1} / {total_pages}</div>", unsafe_allow_html=True)

# tab2: 시맨틱 검색
def show_semantic_search_tab():
    """시맨틱 검색 탭 UI 구성 - Supabase 버전"""
    st.markdown("""
    <style>
        .main-header-1 {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border-radius: 20px;
            margin-bottom: 2.5rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            overflow: hidden;
            position: relative;
            animation: fadeIn 1.5s ease-out;
        }        
        .main-header-1 h1 {
            font-size: 3.5rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .main-header-1 p {
            font-size: 1.4rem;
            opacity: 0.9;
            line-height: 1.6;
            max-width: 800px;
            margin: 0.8rem auto 0 auto;
            position: relative;
            z-index: 1;
        }      
        .main-header-1::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, rgba(255,255,255,0.1) 0%, transparent 20%, transparent 80%, rgba(255,255,255,0.1) 100%);
            animation: scanLine 4s infinite linear;
            z-index: 0;
        }
        @keyframes scanLine {
            0% { transform: translateX(-100%); }
            50% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header-1">
        <h1>🔍 스마트 AI 검색 엔진</h1>
        <p>
            원하는 입찰 정보를 키워드 대신 자연어로 검색하고<br>
            AI가 요약해주는 핵심 내용을 확인하세요.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 검색 UI
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input("검색어를 입력하세요", 
                                    placeholder="예: 서버 구축, 소프트웨어 개발, 건설 공사 등",
                                    key="semantic_search_input")
    with col2:
        search_button = st.button("🔍 검색", key="semantic_search_btn", type="primary")
    
    # 검색 옵션
    with st.expander("🔧 검색 옵션"):
        num_results = st.slider("검색 결과 수", min_value=5, max_value=30, value=10, step=5)
        similarity_threshold = st.slider("유사도 임계값", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    
    # 검색 실행
    if search_button and search_query:
        with st.spinner("검색 중..."):
            # 시맨틱 검색 수행
            search_results = semantic_search_supabase(search_query, k=num_results)
            
            if search_results:
                # 유사도 임계값 필터링
                filtered_results = [(meta, score) for meta, score in search_results if score >= similarity_threshold]
                
                st.markdown(f"### 검색 결과: {len(filtered_results)}건")
                
                if filtered_results:
                    # RAG 응답 생성
                    rag_response = generate_rag_response_supabase(search_query, filtered_results)
                    
                    # AI 응답 표시
                    st.markdown("#### 🤖 AI 검색 요약")
                    st.info(rag_response)
                    
                    st.markdown("---")
                    st.markdown("#### 📋 검색된 공고 목록")
                    
                    # 검색 결과를 표시
                    for idx, (metadata, score) in enumerate(filtered_results):
                        with st.container():
                            col1, col2, col3, col4 = st.columns([0.8, 3, 2, 1])
                            
                            # 유사도 표시 (색상 코딩)
                            if score >= 0.7:
                                col1.markdown(f"<span style='color: green; font-weight: bold;'>{score:.1%}</span>", 
                                            unsafe_allow_html=True)
                            elif score >= 0.5:
                                col1.markdown(f"<span style='color: orange; font-weight: bold;'>{score:.1%}</span>", 
                                            unsafe_allow_html=True)
                            else:
                                col1.markdown(f"<span style='color: red;'>{score:.1%}</span>", 
                                            unsafe_allow_html=True)
                            
                            col2.write(f"**{metadata.get('bidNtceNm', 'N/A')}**")
                            col3.write(metadata.get('ntceInsttNm', 'N/A'))
                            
                            # 상세보기 버튼
                            if col4.button("상세", key=f"semantic_detail_{idx}"):
                                st.session_state["page"] = "detail"
                                st.session_state["selected_live_bid"] = metadata
                                st.rerun()
                                
                            # 추가 정보 표시
                            with st.expander(f"더보기 - {metadata.get('bidNtceNo', 'N/A')}"):
                                st.write(f"**마감일:** {metadata.get('bidClseDate', 'N/A')}")
                                st.write(f"**예산:** {convert_to_won_format(metadata.get('asignBdgtAmt', 0))}")
                                st.write(f"**분류:** {metadata.get('bsnsDivNm', 'N/A')}")
                                
                                # GPT 요약 표시
                                bid_no = metadata.get('bidNtceNo')
                                if bid_no:
                                    summary, created_at, summary_type = get_bid_summary(bid_no)
                                    if summary and summary != "이 공고에 대한 요약이 아직 생성되지 않았습니다.":
                                        st.markdown("**📝 요약:**")
                                        st.write(summary)
                                
                                st.divider()
                else:
                    st.warning(f"유사도 {similarity_threshold:.1%} 이상인 검색 결과가 없습니다. 임계값을 낮춰보세요.")
            else:
                st.warning("검색 결과가 없습니다. 다른 검색어를 시도해보세요.")

# tab3: LangGraph AI 검색
def add_langgraph_search_tab():
    """LangGraph AI 검색 탭 UI - Supabase 버전"""   
    st.markdown("""
    <style>
        .langgraph-header-option2 {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #3a1c71 0%, #d76d77 50%, #ffaf7b 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 2.5rem;
            box-shadow: 0 8px 20px rgba(0,0,0,0.35);
            overflow: hidden;
            position: relative;
            animation: fadeIn 1.5s ease-out;
        }
        .langgraph-header-option2 h1 {
            font-size: 3.5rem;
            font-weight: 900;
            text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
            position: relative;
            z-index: 1;
            margin-bottom: 0.8rem;
        }
        .langgraph-header-option2 p {
            font-size: 1.4rem;
            font-weight: 400;
            opacity: 0.95;
            line-height: 1.6;
            max-width: 900px;
            margin: 0.8rem auto 0 auto;
            position: relative;
            z-index: 1;
        }
        .langgraph-header-option2::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 300%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3) 50%, transparent);
            transform: translateX(-100%);
            animation: lightSweep 10s infinite ease-in-out;
            z-index: 0;
        }
        @keyframes lightSweep {
            0% { transform: translateX(-100%); }
            50% { transform: translateX(0%); }
            100% { transform: translateX(100%); }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="langgraph-header-option2">
        <h1>✨ LangGraph 기반 고급 분석</h1>
        <p>
            AI 워크플로우로 입찰 데이터의 숨겨진 패턴을 파악하고<br>
            전략적인 의사결정을 위한 깊이 있는 통찰을 얻으세요.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 워크플로우 인스턴스 생성
    hybrid_workflow = create_hybrid_search_workflow()
    
    # 검색 UI
    col1, col2 = st.columns([4, 1])
    
    with col1:
        search_query = st.text_input("검색어를 입력하세요", 
                                   placeholder="예: AI 개발, 서버 구축, 소프트웨어 개발 등",
                                   key="langgraph_hybrid_search_input")
    with col2:
        search_button = st.button("🔍 검색", key="langgraph_hybrid_search_btn", type="primary")
    
    # 검색 옵션
    with st.expander("🔧 검색 옵션"):
        col1, col2 = st.columns(2)
        with col1:
            min_results_for_api = st.slider("API 검색 기준 (최소 결과 수)", 
                                           min_value=5, max_value=20, value=10,
                                           help="DB 검색 결과가 이 수치 미만일 때 API 검색 실행")
        with col2:
            show_process = st.checkbox("검색 프로세스 표시", value=True)
    
    # 검색 실행
    if search_button and search_query:
        # 프로세스 컨테이너
        if show_process:
            process_container = st.container()
        
        with st.spinner("하이브리드 검색 실행 중..."):
            # 초기 상태 설정
            initial_state = {
                "query": search_query,
                "search_method": [],
                "supabase_results": [],
                "vector_results": [],
                "api_results": {},
                "combined_results": [],
                "final_results": [],
                "summary": "",
                "messages": [],
                "error": None,
                "total_count": 0,
                "need_api_search": False
            }
            
            # 워크플로우 실행
            try:
                final_state = hybrid_workflow.invoke(initial_state)
                
                # 프로세스 로그 표시
                if show_process:
                    with process_container:
                        st.markdown("### 🔄 검색 프로세스")
                        for msg in final_state["messages"]:
                            if isinstance(msg, HumanMessage):
                                st.markdown(f'<div style="color: #0066cc;">👤 {msg.content}</div>', 
                                          unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div style="color: #009900;">🤖 {msg.content}</div>', 
                                          unsafe_allow_html=True)
                
                # 결과 표시
                if final_state["final_results"]:
                    st.markdown("---")
                    
                    # 요약 표시
                    st.success(final_state["summary"])
                    
                    # 검색 소스 태그
                    st.markdown("### 🏷️ 사용된 검색 방법")
                    cols = st.columns(len(final_state["search_method"]))
                    for idx, method in enumerate(final_state["search_method"]):
                        with cols[idx]:
                            if method == "Supabase":
                                st.info(f"📚 {method}")
                            elif method == "VectorDB":
                                st.warning(f"🔍 {method}")
                            elif method == "API":
                                st.success(f"🌐 {method}")
                    
                    # 상세 결과
                    st.markdown("### 📋 검색 결과")

                    for idx, item in enumerate(final_state["final_results"], 1):
                        with st.container():
                            col1, col2, col3, col4 = st.columns([0.5, 3, 1.5, 1])

                            with col1:
                                st.markdown(f"**{idx}**")

                            with col2:
                                st.markdown(f"**{item.get('bidNtceNm', '제목 없음')}**")
                                st.caption(f"{item.get('ntceInsttNm', '기관명 없음')} | "
                                        f"공고번호: {item.get('bidNtceNo', 'N/A')}")

                            with col3:
                                sources = item.get('source', 'Unknown').split(', ')
                                source_tags = []
                                for source in sources:
                                    if 'Supabase' in source:
                                        source_tags.append("📚")
                                    elif 'VectorDB' in source:
                                        source_tags.append("🔍")
                                    elif 'API' in source:
                                        source_tags.append("🌐")
                                st.write(" ".join(source_tags) + f" (점수: {item.get('relevance_score', 0):.1f})")

                            with col4:
                                st.write(convert_to_won_format(item.get('asignBdgtAmt', 0)))
                                if st.button("상세", key=f"detail_hybrid_{idx}"):
                                    st.session_state["page"] = "detail"
                                    st.session_state["selected_live_bid"] = item
                                    st.rerun()

                            # 상세 정보 표시
                            with st.expander("더보기"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**마감일:** {item.get('bidClseDate', 'N/A')}")
                                    st.write(f"**분류:** {item.get('bsnsDivNm', 'N/A')}")
                                    st.write(f"**검색 소스:** {item.get('source', 'N/A')}")
                                with col2:
                                    st.write(f"**입찰방법:** {item.get('bidMthdNm', 'N/A')}")
                                    st.write(f"**지역제한:** {item.get('rgnLmtYn', 'N/A')}")
                                    st.write(f"**관련도 점수:** {item.get('relevance_score', 0):.1f}")

                                # 벡터 콘텐츠가 있으면 미리보기 표시
                                if 'vector_content' in item:
                                    st.markdown("**📄 문서 내용 미리보기:**")
                                    st.text(item['vector_content'][:300] + "...")
                                
                                # AI 요약이 있을 경우 추가
                                bid_no = item.get('bidNtceNo')
                                if bid_no:
                                    summary, created_at, summary_type = get_bid_summary(bid_no)
                                    if summary and summary != "이 공고에 대한 요약이 아직 생성되지 않았습니다.":
                                        created_info = f"({summary_type} - 생성일: {created_at})" if created_at else ""
                                        st.markdown(f"**📝 AI 요약:** {summary} {created_info}")

                            st.markdown("---")
                else:
                    st.warning(f"'{search_query}'에 대한 검색 결과가 없습니다.")
                    
            except Exception as e:
                st.error(f"검색 중 오류 발생: {str(e)}")
    
    # LangGraph 정보
    with st.expander("🚀 LangGraph 하이브리드 검색 특징", expanded=False):
        st.markdown("""
        **지능형 하이브리드 검색 시스템 (Supabase 버전):**
        
        **🔄 3단계 순차 검색:**
        1. **Supabase 검색** - JSONB 기반 정확한 매칭
        2. **VectorDB 검색** - AI 시맨틱 유사도 검색  
        3. **나라장터 API** - 실시간 최신 데이터 (필요시만)
        
        **🎯 스마트 검색 전략:**
        - DB 검색 결과가 충분하면 API 호출 생략 (성능 최적화)
        - 결과가 부족할 때만 실시간 API 검색 실행
        - 중복 제거 및 관련도 기반 정렬
        
        **📊 통합 점수 시스템:**
        - 여러 소스에서 발견된 공고는 높은 점수
        - Supabase 기본 점수: 5점
        - VectorDB 유사도 점수: 0~10점
        - API 추가 점수: 3점
        
        **✨ Supabase 특화 장점:**
        - JSONB 기반 유연한 데이터 구조
        - PostgreSQL의 강력한 검색 기능
        - pgvector를 통한 벡터 검색
        - 실시간 확장성과 관리 편의성
        """)

# tab4: AI 도우미
def add_chatbot_to_streamlit():
    """Streamlit 앱에 챗봇 기능 추가 - Supabase 버전"""
    st.markdown("""
    <style>
        .chatbot-header-4 {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(90deg, #ff7e5f 0%, #feb47b 100%);
            color: white;
            border-radius: 20px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            overflow: hidden;
            position: relative;
            animation: fadeIn 1.5s ease-out;
        }
        .chatbot-header-4 h1 {
            font-size: 3.5rem;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .chatbot-header-4 p {
            font-size: 1.4rem;
            opacity: 0.8;
            line-height: 1.6;
            max-width: 900px;
            margin: 0.8rem auto 0 auto;
            position: relative;    
            z-index: 1;
        }  
        .chatbot-header-4::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url('data:image/svg+xml;utf8,<svg width="100%" height="100%" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="10" cy="10" r="7" fill="rgba(255,255,255,0.2)"/><circle cx="30" cy="50" r="10" fill="rgba(255,255,255,0.18)"/><circle cx="70" cy="20" r="8" fill="rgba(255,255,255,0.17)"/><circle cx="90" cy="80" r="9" fill="rgba(255,255,255,0.19)"/><circle cx="50" cy="90" r="6" fill="rgba(255,255,255,0.16)"/></svg>') repeat;
            background-size: 20% 20%;
            animation: bubbleFloat 20s infinite linear;
            z-index: 0;
        }
        @keyframes bubbleFloat {
            0% { transform: translateY(0) translateX(0); opacity: 0.8; }
            25% { transform: translateY(-5%) translateX(5%); opacity: 0.9; }
            50% { transform: translateY(-10%) translateX(0); opacity: 0.8; }
            75% { transform: translateY(-5%) translateX(-5%); opacity: 0.9; }
            100% { transform: translateY(0) translateX(0); opacity: 0.8; }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="chatbot-header-4">
        <h1>🤖 AI 챗봇 도우미</h1>
        <p>
            궁금한 점이 있다면 언제든지 질문하세요.<br>
            AI가 친절하게 답변해 드릴게요!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 예시 질문
    example_questions = [
        "AI 개발 관련 입찰 공고를 찾아주세요",
        "서버 구축 입찰 현황은 어떤가요?",
        "최근 공고된 소프트웨어 개발 입찰은?",
        "1억원 이상 IT 입찰 공고가 있나요?",
        "오늘 마감되는 입찰 공고는?",
        "특정 기관의 입찰 공고를 보여주세요"
    ]

    cols = st.columns(6)
    for idx, question in enumerate(example_questions):
        if cols[idx % 6].button(question, key=f"example_{question}"):
            st.session_state.pending_question = question
            st.rerun()
    
    if st.button("🔄 대화 초기화"):
        st.session_state.chat_messages = []
        if 'pending_question' in st.session_state:
            del st.session_state.pending_question
        st.rerun()

    chatbot = init_chatbot()

    # 세션 상태 초기화
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # 이전 대화 내용 표시
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 예시 질문 처리
    if 'pending_question' in st.session_state:
        question = st.session_state.pending_question
        del st.session_state.pending_question
        process_question(question, chatbot)
        st.rerun()
    
    # 사용자 입력 받기
    if prompt := st.chat_input("질문을 입력하세요 (예: AI 관련 입찰 공고를 찾아주세요)"):
        process_question(prompt, chatbot)

# ========== 상세 페이지 ==========
def show_detail_page():
    """상세 페이지 표시 - Supabase 버전"""
    if st.button("⬅️ 목록으로 돌아가기"):
        st.session_state["page"] = "home"
        st.rerun()

    if "selected_live_bid" in st.session_state:
        row = st.session_state["selected_live_bid"]
        
        # 날짜 처리
        마감일 = row.get('마감일') or row.get('bidClseDate')
        마감시간 = row.get('마감시간') or row.get('bidClseTm')
        
        if isinstance(마감일, str):
            try:
                마감일 = pd.to_datetime(마감일)
            except:
                마감일 = None
        
        마감일_표시 = 마감일.strftime("%Y년 %m월 %d일") if pd.notna(마감일) else "공고 참조"
        마감시간_표시 = 마감시간 if pd.notna(마감시간) else "공고 참조"

        게시일 = row.get('게시일') or row.get('bidNtceDate')
        if isinstance(게시일, str):
            try:
                게시일 = pd.to_datetime(게시일)
            except:
                게시일 = None
        게시일_표시 = 게시일.strftime("%Y년 %m월 %d일") if pd.notna(게시일) else "정보 없음"

        # 상단 정보 카드
        st.markdown(
            f"""
            <div style="
                background-color: #e0f2f7; 
                padding: 25px 20px; 
                border-radius: 15px; 
                margin-bottom: 30px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            ">
                <h2 style="color: #0056b3; margin-top: 0px; margin-bottom: 10px; font-weight: bold; font-size: 2.2em;">
                    {row.get('공고명') or row.get('bidNtceNm', '공고명 없음')}
                </h2>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 1.2em; font-weight: 600; color: #333;">
                        📊 구분: {row.get('입찰공고상태명') or row.get('bidNtceSttusNm', '정보 없음')}
                    </span>
                    <span style="font-size: 1.2em; font-weight: 600; color: #333;">
                        🏢 수요기관: {row.get('수요기관') or row.get('dmndInsttNm', '정보 없음')}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 1.2em; font-weight: 600; color: #333;">
                        📅 게시일: {게시일_표시}
                    </span>                   
                </div>
                    <span style="font-size: 1.2em; font-weight: 600; color: #333;">
                            ⏳ 공고마감일: {마감일_표시} {마감시간_표시}
                    </span>
                    <div style="font-size: 1.5em; font-weight: bold; color: #007bff; text-align: right;">
                        💰 금액: {format_won(str(row.get('금액') or row.get('asignBdgtAmt', '0')))}
                    </div>
            </div>
            """, unsafe_allow_html=True
        )

        # 상세 정보 카드들
        col1, col2, col3 = st.columns([1,1,1])       
        
        with col1:
            공동수급 = row.get('공동수급') or row.get('cmmnReciptMethdNm')
            지역제한 = row.get('지역제한') or row.get('rgnLmtYn')
            참가가능지역 = row.get('참가가능지역명') or row.get('prtcptPsblRgnNm')
            
            st.markdown(
                f"""
                <div style="
                background-color: #f0fdf4;
                border: 1px solid #e5e5e5;
                padding: 25px 20px; 
                border-radius: 15px; 
                margin-bottom: 30px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                height: 300px; 
            ">
                    <h4 style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">공동수급 • 지역제한</h4>
                    <hr style="border: 1px solid #e5e5e5; margin-top: 10px; margin-bottom: 10px;">
                    <div style="margin-bottom: 10px;">
                        <span style="font-size: 16px; font-weight: bold; color: #333;">🤝 공동수급</span><br>
                        <span style="font-size: 18px; font-weight: 500; color: #000;">
                        {format_joint_contract(공동수급)}</span>
                    </div>
                    <div>
                        <span style="font-size: 16px; font-weight: bold; color: #333;">📍 지역제한</span><br>
                        <span style="font-size: 18px; font-weight: 500; color: #000;">
                            {참가가능지역 if 지역제한 == 'Y' and pd.notna(참가가능지역) else '없음'}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )          
            
        with col2:
            업종명 = row.get('투찰가능업종명') or row.get('bidprcPsblIndstrytyNm')
            st.markdown(
                f"""
                <div style="
                background-color: #fff9e6; 
                border: 1px solid #e5e5e5;
                padding: 25px 20px; 
                border-radius: 15px; 
                margin-bottom: 30px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                height: 300px; 
            ">
                    <h4 style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">🚫업종 제한</h4>
                    <hr style="border: 1px solid #e5e5e5; margin-top: 10px; margin-bottom: 10px;">
                    <p style="font-size: 18px; font-weight: bold; overflow-y: auto; max-height: 90px;">
                        {"<br>".join([f"{i+1}. {item.strip()}" for i,
                                    item in enumerate(str(업종명).split(',')) if str(item).strip()]) 
                                    if 업종명 and str(업종명).strip() != "" else '공문서참조'}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div style="
                background-color: #f0f8ff; 
                border: 1px solid #e5e5e5;
                padding: 25px 20px; 
                border-radius: 15px; 
                margin-bottom: 30px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                height: 300px; 
                ">
                    <h4 style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">💡 기타 정보</h4>
                    <hr style="border: 1px solid #e5e5e5; margin-top: 10px; margin-bottom: 10px;">
                    <p style="font-size: 16px;">
                        <strong>공고번호:</strong> {row.get('공고번호') or row.get('bidNtceNo', 'N/A')}<br>
                        <strong>공고기관:</strong> {row.get('공고기관') or row.get('ntceInsttNm', 'N/A')}<br>
                        <strong>분류:</strong> {row.get('분류') or row.get('bsnsDivNm', 'N/A')}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # GPT 요약
        bid_no = row.get('공고번호') or row.get('bidNtceNo')
        summary_text, created_at, summary_type = get_bid_summary(bid_no)

        created_info = ""
        if created_at:
            type_label = "상세문서 기반" if summary_type == "hwp_based" else "기본정보 기반"
            created_info = f" ({type_label}, 생성일: {created_at})"

        st.markdown(
            f"""
            <div style="
                background-color: #f0f8ff; 
                border-left: 5px solid #4682b4; 
                padding: 15px;
                margin-top: 10px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            ">
            <div>
                <span style="font-size: 16px; font-weight: bold; color: #333;">AI 상세요약{created_info}</span><br>   
                <hr style="border: 1px solid #e5e5e5; margin-top: 10px; margin-bottom: 10px;">         
            </div>
                <p style="font-size: 16px; font-weight: 500;">{summary_text}</p>
            </div>
            """, unsafe_allow_html=True
        )

# ========== 메인 애플리케이션 ==========
def main():
    """메인 애플리케이션 - Supabase 버전"""
    st.set_page_config(page_title="입찰 공고 서비스 (Supabase)", layout="wide")

    # 설정 확인
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.error("⚠️ Supabase 설정이 필요합니다. SUPABASE_URL과 SUPABASE_ANON_KEY를 설정해주세요.")
        st.stop()
    
    if not OPENAI_API_KEY:
        st.error("⚠️ OpenAI API 키가 필요합니다. 임베딩 및 GPT 기능을 사용할 수 없습니다.")
        st.stop()

    # 연결 테스트
    supabase = init_supabase_client()
    if not supabase:
        st.error("❌ Supabase 연결 실패")
        st.stop()

    # 데이터 로드
    df_live = get_live_bids_data()

    # 페이지 상태 관리
    page = st.session_state.get("page", "home")

    if page == "home":
        # 탭 생성
        tab1, tab2, tab3, tab4 = st.tabs(["📢 실시간 입찰 공고", "🔍 AI 검색", "🚀 LangGraph AI 검색", "🤖 AI 도우미"])
        
        with tab1:        
            show_live_bids_tab(df_live)
        
        with tab2:       
            show_semantic_search_tab()
            
        with tab3:
            add_langgraph_search_tab()
        
        with tab4:
            add_chatbot_to_streamlit()

    elif page == "detail":
        show_detail_page()

if __name__ == "__main__":
    main()