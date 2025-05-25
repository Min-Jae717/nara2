import streamlit as st
import pandas as pd
import psycopg2
import numpy as np
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# 컬럼 매핑용 딕셔너리
simple_info = {
    "bidNtceNo": "입찰공고번호",
    "bidNtceOrd": "입찰공고차수",
    "bidNtceNm": "입찰공고명",
    "bidNtceSttusNm": "입찰공고상태명",
    "bidNtceDate": "입찰공고일자",
    "bidNtceBgn": "입찰공고시각",
    "bsnsDivNm": "업무구분명",
    "intrntnlBidYn": "국제입찰여부",
    "cmmnCntrctYn": "공동계약여부",
    "cmmnReciptMethdNm": "공동수급방식명",
    "elctrnBidYn": "전자입찰여부",
    "cntrctCnclsSttusNm": "계약체결형태명",
    "cntrctCnclsMthdNm": "계약체결방법명",
    "bidwinrDcsnMthdNm": "낙찰자결정방법명",
    "ntceInsttNm": "공고기관명",
    "ntceInsttCd": "공고기관코드",
    "ntceInsttOfclDeptNm": "공고기관담당자부서명",
    "ntceInsttOfclNm": "공고기관담당자명",
    "ntceInsttOfclTel": "공고기관담당자전화번호",
    "ntceInsttOfclEmailAdrs": "공고기관담당자이메일주소",
    "dmndInsttNm": "수요기관명",
    "dmndInsttCd": "수요기관코드",
    "dmndInsttOfclDeptNm": "수요기관담당자부서명",
    "dmndInsttOfclNm": "수요기관담당자명",
    "dmndInsttOfclTel": "수요기관담당자전화번호",
    "dmndInsttOfclEmailAdrs": "수요기관담당자이메일주소",
    "presnatnOprtnYn": "설명회실시여부",
    "presnatnOprtnDate": "설명회실시일자",
    "presnatnOprtnTm": "설명회실시시각",
    "presnatnOprtnPlce": "설명회실시장소",
    "bidPrtcptQlfctRgstClseDate": "입찰참가자격등록마감일자",
    "bidPrtcptQlfctRgstClseTm": "입찰참가자격등록마감시각",
    "cmmnReciptAgrmntClseDate": "공동수급협정마감일자",
    "cmmnReciptAgrmntClseTm": "공동수급협정마감시각",
    "bidBeginDate": "입찰개시일자",
    "bidBeginTm": "입찰개시시각",
    "bidClseDate": "입찰마감일자",
    "bidClseTm": "입찰마감시각",
    "opengDate": "개찰일자",
    "opengTm": "개찰시각",
    "opengPlce": "개찰장소",
    "asignBdgtAmt": "배정예산금액",
    "presmptPrce": "추정가격",
    "rsrvtnPrceDcsnMthdNm": "예정가격결정방법명",
    "rgnLmtYn": "지역제한여부",
    "prtcptPsblRgnNm": "참가가능지역명",
    "indstrytyLmtYn": "업종제한여부",
    "bidprcPsblIndstrytyNm": "투찰가능업종명",
    "bidNtceUrl": "입찰공고URL",
    "dataBssDate": "데이터기준일자"
}

st.set_page_config(page_title="입찰 공고 서비스", layout="wide")

# 캐싱 데이터 로드
@st.cache_data
def load_all_data():
    conn = psycopg2.connect(st.secrets["SUPABASE_DB_URL"])
    df = pd.read_sql("SELECT raw FROM bids_live ORDER BY raw->>'bidNtceDate' DESC, raw->>'bidNtceBgn' DESC", conn)
    conn.close()
    live_data = [(l[0]) for l in df.values]
    df_live = pd.json_normalize(live_data)
    return df_live

def load_new_data(last_date, last_time):
    conn = psycopg2.connect(st.secrets["SUPABASE_DB_URL"])
    sql = """
        SELECT raw FROM bids_live
        WHERE (raw->>'bidNtceDate' > %s)
           OR (raw->>'bidNtceDate' = %s AND raw->>'bidNtceBgn' > %s)
        ORDER BY raw->>'bidNtceDate' DESC, raw->>'bidNtceBgn' DESC
    """
    df = pd.read_sql(sql, conn, params=[str(last_date), str(last_date), str(last_time)])
    conn.close()
    live_data = [(l[0]) for l in df.values]
    new_df = pd.json_normalize(live_data)
    return new_df

# 캐시 및 컬럼명 변환
if "cached_df" not in st.session_state:
    st.session_state["cached_df"] = load_all_data()
# 무조건 컬럼명 한글화(중복해도 안전)
st.session_state["cached_df"].rename(columns=simple_info, inplace=True)

# 마지막 날짜/시간 구하기
if not st.session_state["cached_df"].empty:
    last_row = st.session_state["cached_df"].iloc[0]
    last_date = last_row["입찰공고일자"] if "입찰공고일자" in last_row else last_row.get("bidNtceDate")
    last_time = last_row["입찰공고시각"] if "입찰공고시각" in last_row else last_row.get("bidNtceBgn")
else:
    last_date, last_time = "2000-01-01", "00:00"

# 신규 데이터 불러오고 컬럼명 한글로 변환
new_df = load_new_data(str(last_date), str(last_time))
new_df.rename(columns=simple_info, inplace=True)
if not new_df.empty:
    st.session_state["cached_df"] = pd.concat([new_df, st.session_state["cached_df"]], ignore_index=True)
    st.session_state["cached_df"].rename(columns=simple_info, inplace=True) # 병합 후에도 컬럼명 강제

# 최종 데이터프레임 사용
df_live = st.session_state["cached_df"]

# 메인 페이지 금액 억단위
def convert_to_won_format(amount):
    try:
        if not amount or pd.isna(amount):
            return "공고서 참조"
        
        amount = float(str(amount).replace(",", ""))

        if amount >= 10000000: # 1억 이상
            amount_in_100m = amount / 100000000
            return f"{amount_in_100m:.1f}억"
        elif amount >= 10000:# 100만원 미만
            amount_in_10k = amount / 10000
            return f"{round(amount_in_10k,1):.1f}만원"
        else: # 1만원 미만
            return f"{amount}원"
        
    except Exception as e:
          return f"오류 발생: {str(e)}"

# 상세페이지 금액을 원화로 포맷팅
def format_won(amount):
    try:
        # 쉼표를 포함한 문자열을 숫자로 변환 후, 다시 천 단위로 쉼표 추가
        amount = int(amount.replace(",", ""))  # 쉼표 제거 후 숫자로 변환
        return f"{amount:,}원"  # 천 단위로 쉼표 추가 후 원화 표시
    except ValueError:
        return "공고서 참조"
    
# 공동수급 정리
def format_joint_contract(value):
    if value and value.strip():
        return f"허용 [{value.strip()}]"
    return "공고서 참조"
    
st.title("📝 실시간 입찰 공고 및 낙찰 결과")

# 쿼리 파라미터로 현재 페이지 구분
page = st.session_state.get("page", "home")
   
tab1, = st.tabs(["📢 실시간 입찰 공고"])
# ------------------------
# 📢 실시간 입찰 공고 탭
# ------------------------
if page == 'home':    
    st_autorefresh(interval=60 * 1000, key='refresh_home_page') # 60초마다 새로고침
    with tab1:
        st.subheader("📢 현재 진행 중인 입찰 목록")

        # 2. DataFrame 컬럼명 변경
        # df_live.rename(columns=simple_info, inplace=True)

        df_live["입찰공고번호_차수"] = df_live["입찰공고번호"].astype(str) + "-" + df_live["입찰공고차수"].astype(str)
        df_live["금액"] = df_live.apply(lambda x:x["추정가격"] if x["업무구분명"] == "공사" 
                                      else x["배정예산금액"], axis=1)
        # 👉 날짜 형식 변환
        df_live["입찰공고일시"] = pd.to_datetime((df_live["입찰공고일자"]+df_live["입찰공고시각"]), format="%Y-%m-%d%H:%M")
        df_live["입찰마감일시"] = pd.to_datetime((df_live["입찰마감일자"]+df_live["입찰마감시각"]), format="%Y-%m-%d%H:%M")

        df_live["입찰공고일자"] = pd.to_datetime(df_live["입찰공고일자"], format="%Y-%m-%d")
        df_live["입찰마감일자"] = pd.to_datetime(df_live["입찰마감일자"], format="%Y-%m-%d")

        # 시간 형식 변환
        df_live["입찰공고시각"] = pd.to_datetime(df_live["입찰공고시각"], format="%H:%M")
        df_live["입찰마감시각"] = pd.to_datetime(df_live["입찰마감시각"], format="%H:%M")


        # 🔍 필터 UI
        search_keyword = st.text_input("🔎 공고명 또는 공고기관 검색")

        unique_categories = ["공사", "용역", "물품", "외자"]

        selected_cls = st.multiselect("📁 분류 선택", 
                                    options= unique_categories, 
                                    default = [])

        col2, col3, col4 = st.columns(3)        
            
        with col2:
            start_date = st.date_input("📅 게시일 기준 시작일", value=df_live["입찰공고일자"].min().date())
        with col3:
            end_date = st.date_input("📅 게시일 기준 종료일", value=df_live["입찰공고일자"].max().date())
        with col4:
            sort_col = st.selectbox("정렬기준",options=["실시간","게시일","마감일","금액"])
            if sort_col == "실시간" :
                sort_order = "내림차순"
                st.empty()
            else :
                sort_order = st.radio("정렬 방향", options=["오름차순", "내림차순"], horizontal=True,
                                  label_visibility="collapsed")
            
            
        # 🔎 필터링 적용
        filtered = df_live.copy()

        # 1. 분류 필터
        if selected_cls:
            filtered = df_live[df_live["업무구분명"].isin(selected_cls)]
        else:
            filtered = df_live.copy()

        # 2. 검색어 필터
        if search_keyword:
            filtered = filtered[
                filtered["입찰공고명"].str.contains(search_keyword, case=False, na=False, regex=False) |
                filtered["공고기관명"].str.contains(search_keyword, case=False, na=False, regex=False) |
                filtered["입찰공고번호_차수"].str.contains(search_keyword, case=False, na=False, regex=False)
            ]

        # 3. 게시일 범위 필터
        filtered = filtered[
            (filtered["입찰공고일자"].dt.date >= start_date) &
            (filtered["입찰공고일자"].dt.date <= end_date)
        ]

        # 5. 정렬 순서 수정
        ascending = True if sort_order == "오름차순" else False

        # 6. 정렬 적용
        if sort_col == "실시간" :
            filtered = filtered.sort_values(by=["입찰공고일자", "입찰공고시각"], ascending=False)
        else:
            ascending = True if sort_order == "오름차순" else False

        if sort_col == "게시일":
            filtered = filtered.sort_values(by=["입찰공고일자", "입찰공고시각"], ascending=ascending)
        elif sort_col == "마감일":
            filtered = filtered.sort_values(by="입찰마감일자", ascending=ascending)
        elif sort_col == "금액":
            filtered = filtered.sort_values(by="금액", ascending=ascending)

        # 결과 출력
        st.markdown(f"<div style='text-align: left; margin-bottom: 10px;'>검색 결과 {len(filtered)}건</div>", unsafe_allow_html=True)
      
        # 1. 페이지 크기 설정
        PAGE_SIZE = 10

        # 2. 데이터 분할 함수
        def paginate_dataframe(df, page_num, page_size):
            start_index = page_num * page_size
            end_index = (page_num + 1) * page_size
            return df.iloc[start_index:end_index]

        # 3. 현재 페이지 번호 초기화
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = 0


        # 4. 데이터 필터링 및 페이지 분할
        total_pages = (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE
        paginated_df = paginate_dataframe(filtered, st.session_state["current_page"], PAGE_SIZE)
        st.write("")
        st.write("")  
        # 테이블 헤더
        header_cols = st.columns([2, 1.5, 5, 3, 1, 1.5, 2, 2, 1.5])
        headers = ['공고번호',"구분",'공고명','공고기관','분류','금액','게시일','마감일','상세정보']

        for col, head in zip(header_cols, headers):
            col.markdown(f"**{head}**")

        st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True) # 헤더 아래 구분선

        # 행 렌더링 + 버튼 (페이지네이션된 데이터 사용)
        for i, row in paginated_df.iterrows():
            cols = st.columns([2, 1.5, 5, 3, 1, 1.5, 2, 2, 1.5])
            cols[0].write(row["입찰공고번호_차수"])
            cols[1].write(row["입찰공고상태명"])
            # cols[2].write(row["입찰공고명"])
            bid_title_link = f"[{row['입찰공고명']}]({row['입찰공고URL']})"
            cols[2].markdown(bid_title_link)
            cols[3].write(row["공고기관명"])
            cols[4].write(row["업무구분명"])
            cols[5].write(convert_to_won_format(row["금액"]))
            cols[6].markdown(f"<div style='text-align:center'>{row['입찰공고일자'].strftime('%Y-%m-%d')}<br>{row['입찰공고시각'].strftime('%H:%M')}</div>",
            unsafe_allow_html=True)
            if pd.isna(row["입찰마감일시"]):
                cols[7].write("공고서 참조")
            else:
                cols[7].markdown(f"<div style='text-align:center'>{row['입찰마감일자'].strftime('%Y-%m-%d')}<br>{row['입찰마감시각'].strftime('%H:%M')}</div>",
                unsafe_allow_html=True)
            if cols[8].button("보기", key=f"live_detail_{i}"):
                st.session_state["page"] = "detail"
                st.session_state["selected_live_bid"] = row.to_dict()
                st.rerun()

            # 각 행 아래에 구분선 추가
            st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)


        # 6. "이전" 및 "다음" 버튼
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

        # 7. 페이지 번호 표시    
        st.markdown(f"<div style='text-align: center;'> {st.session_state['current_page'] + 1}</div>", unsafe_allow_html=True)
        
elif page == "detail":
    # ⬅️ 뒤로가기 버튼 추가
    if st.button("⬅️ 목록으로 돌아가기"):
        st.session_state["page"] = "home"
        st.rerun()

    # 선택된 공고 정보가 있는지 확인
    if "selected_live_bid" in st.session_state:
        row = st.session_state["selected_live_bid"]
        
        # --- 상단 핵심 정보 섹션 (강조) ---
        마감일시 = row.get('입찰마감일시')
        # 마감시간 = row.get('입찰마감시각')
        마감일시_표시 = 마감일시.strftime("%Y년 %m월 %d일 %H시 %M분") if pd.notna(마감일시) else "공고서 참조"
        # 마감시간_표시 = 마감시간.strftime("%H:%M") if pd.notna(마감시간) else "공고서 참조"

        # 날짜 및 시간 처리
        게시일 = row.get('입찰공고일자')
        게시일_표시 = 게시일.strftime("%Y년 %m월 %d일") if pd.notna(게시일) else "정보 없음"

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
                    {row.get('입찰공고명', '공고명 없음')}
                </h2>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 1.2em; font-weight: 600; color: #333;">
                        📊 구분: {row.get('입찰공고상태명', '정보 없음')}
                    </span>
                    <span style="font-size: 1.2em; font-weight: 600; color: #333;">
                        🏢 수요기관: {row.get('수요기관명', '정보 없음')}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 1.2em; font-weight: 600; color: #333;">
                        📅 게시일: {게시일_표시}
                    </span>                   
                </div>
                    <span style="font-size: 1.2em; font-weight: 600; color: #333;">
                            ⏳ 공고마감일: {마감일시_표시}
                    </span>
                    <div style="font-size: 1.5em; font-weight: bold; color: #007bff; text-align: right;">
                        💰 금액: {format_won(row.get('금액'))}
                    </div>
            </div>
            """, unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([1,1,1])       
        
        # 업종제한 카드 형태 표시
        with col1:
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
                        {format_joint_contract(row['공동수급방식명'])}</span>
                    </div>
                    <div>
                        <span style="font-size: 16px; font-weight: bold; color: #333;">📍 지역제한</span><br>
                        <span style="font-size: 18px; font-weight: 500; color: #000;">
                            {row['참가가능지역명'] if row['지역제한여부'] == 'Y' and pd.notna(row['참가가능지역명']) else '없음'}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )          
            
        with col2:
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
                    <!-- 내용이 길면 스크롤되도록 -->
                    <p style="font-size: 18px; font-weight: bold; overflow-y: auto; max-height: 90px;">
                        {"<br>".join([f"{i+1}. {item.strip()}" for i,
                                    item in enumerate(str(row['투찰가능업종명']).split(',')) if str(item).strip()]) 
                                    if row['투찰가능업종명'] and str(row['투찰가능업종명']).strip() != "" else '공문서참조'}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

         # col3은 현재 비어 있으므로, 비워두거나 다른 내용 추가 가능
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
                        더 필요한 정보가 있다면 여기에 추가할 수 있습니다.<br>
                        예: 관련 법규, 특이사항 등
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # # 📌 MongoDB에서 GPT 요약 가져오기
        # summary_text, created_at = get_summary_from_mongodb_v2(row['공고번호'])

        # # 요약 생성 시간 표시
        # created_info = ""
        # if created_at:
        #     created_info = f" (생성일: {created_at})"

        # st.markdown(
        #     f"""
        #     <div style="
        #         background-color: #f0f8ff; 
        #         border-left: 5px solid #4682b4; 
        #         padding: 15px;
        #         margin-top: 10px;
        #         border-radius: 10px;
        #         box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        #     ">
        #     <div>
        #         <span style="font-size: 16px; font-weight: bold; color: #333;">AI 상세요약{created_info}</span><br>   
        #         <hr style="border: 1px solid #e5e5e5; margin-top: 10px; margin-bottom: 10px;">         
        #     </div>
        #         <p style="font-size: 16px; font-weight: 500;">{summary_text}</p>
        #     </div>
        #     """, unsafe_allow_html=True
        # )
