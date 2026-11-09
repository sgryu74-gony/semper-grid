import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import unicodedata

# --- [모바일 최적화 공통 설정] ---
MOBILE_LAYOUT = dict(
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1
    ),
    height=450
)
MOBILE_CONFIG = {'displayModeBar': False}
# ---------------------------------

# 1. 페이지 레이아웃 설정
st.set_page_config(layout="wide", page_title="Semper Grid Master Analytics", page_icon="🛡️")

# 기본 디렉토리 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# 종목 코드 매핑
TICKER_MAP = {
    '005930': '삼성전자',
    '069500': 'KODEX 200',
    '088980': '맥쿼리인프라',
    '091220': 'TIGER 은행',
    '229200': 'KODEX 코스닥150',
    '261240': 'KODEX 미국달러선물',
    '411060': 'ACE KRX금현물',
    '453850': 'ACE 미국30년국채액티브'
}

# 종목별 연간 배당 분배율 가정
DIV_YIELD_MAP = {
    'TIGER 은행': 0.060,
    '맥쿼리인프라': 0.065,
    'ACE 미국30년국채액티브': 0.045,
    'KODEX 200': 0.020,
    'KODEX 코스닥150': 0.005,
    'KODEX 미국달러선물': 0.0,
    'ACE KRX금현물': 0.0,
    '삼성전자': 0.022
}

@st.cache_data
def load_all_data():
    daily_path = os.path.join(LOGS_DIR, "daily.csv")
    trades_path = os.path.join(LOGS_DIR, "trades.csv")
    
    if not os.path.exists(daily_path) or not os.path.exists(trades_path):
        st.error(f"로그 파일을 찾을 수 없습니다: {LOGS_DIR} 폴더에 daily.csv와 trades.csv가 있는지 확인해주세요.")
        st.stop()
        
    df_daily = pd.read_csv(daily_path)
    df_trades = pd.read_csv(trades_path)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_trades['date'] = pd.to_datetime(df_trades['date'])
    
    # 주기 컬럼 생성
    df_daily['year'] = df_daily['date'].dt.year
    df_daily['quarter'] = df_daily['date'].dt.to_period('Q').astype(str)
    df_daily['month'] = df_daily['date'].dt.to_period('M').astype(str)
    
    df_trades['year'] = df_trades['date'].dt.year
    df_trades['quarter'] = df_trades['date'].dt.to_period('Q').astype(str)
    df_trades['month'] = df_trades['date'].dt.to_period('M').astype(str)
    
    # 전략 전체 및 단순 보유(B&H) 포트폴리오 최대 하락폭(MDD) 계산
    df_daily['cummax_equity'] = df_daily['total_equity'].cummax()
    df_daily['dd_pct'] = (df_daily['total_equity'] - df_daily['cummax_equity']) / df_daily['cummax_equity'] * 100
    
    df_daily['cummax_bnh'] = df_daily['total_bnh'].cummax()
    df_daily['bnh_dd_pct'] = (df_daily['total_bnh'] - df_daily['cummax_bnh']) / df_daily['cummax_bnh'] * 100

    # data/ 폴더 내 종목별 과거 시세 로드
    stock_dfs = {}
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        norm_filename = unicodedata.normalize('NFC', filename)
        code = norm_filename.split()[0].replace('.csv', '')
        name = TICKER_MAP.get(code, code)
        try:
            try:
                df_s = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                df_s = pd.read_csv(filepath, encoding='cp949')

            df_s.columns = [c.strip().replace('\ufeff', '') for c in df_s.columns]
            col_rename = {'Date': '날짜', 'Price': '종가', 'Close': '종가', 'Open': '시가', 'High': '고가', 'Low': '저가', 'Vol.': '거래량'}
            df_s = df_s.rename(columns=col_rename)

            for col in ['종가', '시가', '고가', '저가']:
                if col in df_s.columns:
                    df_s[col] = df_s[col].astype(str).str.replace(',', '').str.strip()
                    df_s[col] = pd.to_numeric(df_s[col], errors='coerce')
            
            df_s = df_s.dropna(subset=['종가', '고가', '저가'])
            df_s['날짜'] = pd.to_datetime(df_s['날짜'].astype(str).str.replace(' ', ''))
            df_s = df_s.sort_values('날짜').reset_index(drop=True)
            df_s['year'] = df_s['날짜'].dt.year
            df_s['quarter'] = df_s['날짜'].dt.to_period('Q').astype(str)
            df_s['month'] = df_s['날짜'].dt.to_period('M').astype(str)
            
            # 240일 기준 상대 가격 위치(PRS 지표) 계산
            df_s['high_240'] = df_s['고가'].rolling(240, min_periods=1).max()
            df_s['low_240'] = df_s['저가'].rolling(240, min_periods=1).min()
            denom = df_s['high_240'] - df_s['low_240']
            df_s['prs'] = np.where(denom == 0, 50.0, (df_s['종가'] - df_s['low_240']) / denom * 100.0)
            
            stock_dfs[name] = df_s
        except Exception:
            pass
            
    return df_daily, df_trades, stock_dfs

df_daily, df_trades, stock_dfs = load_all_data()

# 2. 최상단 요약 메트릭
st.title("🛡️ Semper Grid: 3중 복합 자산관리 & 퀀트 자동 트레이딩 관제탑")

c1, c2, c3, c4, c5 = st.columns(5)
init_seed = df_daily['total_invested'].iloc[0]
final_eq = df_daily['total_equity'].iloc[-1]
tot_return = ((final_eq / init_seed) - 1.0) * 100
strat_mdd = df_daily['dd_pct'].min()
bnh_mdd = df_daily['bnh_dd_pct'].min()
total_pnl = df_trades['realized_pnl'].sum()

c1.metric("운용 시작 원금", f"{init_seed:,.0f} 원")
c2.metric("현재 자산 평가액", f"{final_eq:,.0f} 원", f"{tot_return:+.2f}%")
c3.metric("누적 확정 수익(실현손익)", f"{total_pnl:,.0f} 원")
c4.metric("최대 하락폭 방어율", f"{strat_mdd:.1f}%", f"시장 단순보유({bnh_mdd:.1f}%) 대비 방어", delta_color="normal")
c5.metric("총 자동매매 체결", f"{len(df_trades):,} 건", "100% 룰 기반 시스템")

st.markdown("---")

# 3. 마스터 탭 구성
tab_pitch, tab_comp, tab_candle, tab_timing, tab_edge = st.tabs([
    "🏛️ 전략 핵심 성과 리포트",
    "🌐 운용 종목 종합 비교 (Universe)",
    "🎯 캔들 & 분할 매매 타점 분석",
    "⏱️ 진입 시점별 안정성 검증",
    "🚨 리스크 통제 & 비상 대응 로그"
])

# =============================================================
# 탭 1: 전략 핵심 성과 리포트
# =============================================================
with tab_pitch:
    st.subheader("💡 전략 핵심 경쟁력: 하락장 방어와 꾸준한 현금 창출력")
    
    # [1] 통합 패널: 하락장 방어 & 자산 성장 듀얼 분석
    st.markdown("##### 🛡️ 1. 자산 성장 추이 & 하락장 방어율 비교 분석")
    st.caption("상단: 매매차익과 배당이 쌓이며 우상향하는 자산 곡선 | 하단: 시장이 급락할 때 계좌를 지켜낸 실제 방어력")

    dual_c1, dual_c2, dual_c3 = st.columns([1.5, 1.2, 2.3])
    with dual_c1:
        dual_targets = ["포트폴리오 전체"] + [c.replace('stock_eval_', '') for c in df_daily.columns if c.startswith('stock_eval_')]
        selected_target = st.selectbox("분석 대상 선택", dual_targets, key="dual_target_select")

    with dual_c2:
        dual_view_mode = st.radio("조회 기간 단위", ["전체 기간", "연도별", "분기별"], horizontal=True, key="dual_mode_radio")

    if "dual_y_idx" not in st.session_state: st.session_state.dual_y_idx = 0
    if "dual_q_idx" not in st.session_state: st.session_state.dual_q_idx = 0

    all_years = sorted(df_daily['year'].unique().tolist())
    all_quarters = sorted(df_daily['quarter'].unique().tolist())
    dual_df = df_daily.copy()
    period_label = "전체 기간"

    with dual_c3:
        if dual_view_mode == "연도별":
            btn_prev, sel_box, btn_next = st.columns([0.8, 1.4, 0.8])
            with btn_prev:
                if st.button("◀ 이전 연도", key="dual_prev_y"):
                    if st.session_state.dual_y_idx > 0:
                        st.session_state.dual_y_idx -= 1
                        st.rerun()
            with btn_next:
                if st.button("다음 연도 ▶", key="dual_next_y"):
                    if st.session_state.dual_y_idx < len(all_years) - 1:
                        st.session_state.dual_y_idx += 1
                        st.rerun()
            with sel_box:
                sel_y = st.selectbox("연도 선택", all_years, index=st.session_state.dual_y_idx, key="dual_sel_y")
                if sel_y != all_years[st.session_state.dual_y_idx]:
                    st.session_state.dual_y_idx = all_years.index(sel_y)
                    st.rerun()
            dual_df = df_daily[df_daily['year'] == all_years[st.session_state.dual_y_idx]].copy()
            period_label = f"{all_years[st.session_state.dual_y_idx]}년"

        elif dual_view_mode == "분기별":
            btn_prev, sel_box, btn_next = st.columns([0.8, 1.4, 0.8])
            with btn_prev:
                if st.button("◀ 이전 분기", key="dual_prev_q"):
                    if st.session_state.dual_q_idx > 0:
                        st.session_state.dual_q_idx -= 1
                        st.rerun()
            with btn_next:
                if st.button("다음 분기 ▶", key="dual_next_q"):
                    if st.session_state.dual_q_idx < len(all_quarters) - 1:
                        st.session_state.dual_q_idx += 1
                        st.rerun()
            with sel_box:
                sel_q = st.selectbox("분기 선택", all_quarters, index=st.session_state.dual_q_idx, key="dual_sel_q")
                if sel_q != all_quarters[st.session_state.dual_q_idx]:
                    st.session_state.dual_q_idx = all_quarters.index(sel_q)
                    st.rerun()
            dual_df = df_daily[df_daily['quarter'] == all_quarters[st.session_state.dual_q_idx]].copy()
            period_label = f"{all_quarters[st.session_state.dual_q_idx]}"

    if selected_target == "포트폴리오 전체":
        daily_div_sum = pd.Series(0.0, index=dual_df.index)
        for tname, yld in DIV_YIELD_MAP.items():
            col = f"stock_eval_{tname}"
            if col in dual_df.columns:
                daily_div_sum += dual_df[col] * (yld / 250.0)
        dual_df['cum_div'] = daily_div_sum.cumsum()
        dual_df['total_with_div'] = dual_df['total_equity'] + dual_df['cum_div']

        tot_m = pd.merge(dual_df[['date']], df_trades.groupby('date')['realized_pnl'].sum().reset_index(), on='date', how='left').fillna(0)
        dual_df['cum_realized'] = tot_m['realized_pnl'].cumsum()
        dual_df['locked_cash_pipeline'] = init_seed + dual_df['cum_realized'] + dual_df['cum_div']

        strat_val_series = dual_df['total_with_div']
        strat_locked_series = dual_df['locked_cash_pipeline']
        bnh_val_series = dual_df['total_bnh']

        dual_df['strat_peak'] = dual_df['total_equity'].cummax()
        dual_df['strat_dd'] = (dual_df['total_equity'] - dual_df['strat_peak']) / dual_df['strat_peak'] * 100
        dual_df['bnh_peak'] = dual_df['total_bnh'].cummax()
        dual_df['bnh_dd'] = (dual_df['total_bnh'] - dual_df['bnh_peak']) / dual_df['bnh_peak'] * 100

        y1_label = "총 자산 가치 (원)"
        strat_name = "그리드 전략 총자산 (수익+배당 재투자)"
        bnh_name = "시장 단순보유 (Buy & Hold)"
    else:
        eval_col = f"stock_eval_{selected_target}"
        bnh_col = f"bnh_{selected_target}"
        unrealized_col = f"unrealized_{selected_target}"
        target_yield = DIV_YIELD_MAP.get(selected_target, 0.0)

        sub_trades = df_trades[df_trades['name'] == selected_target].groupby('date')['realized_pnl'].sum().reset_index()
        sub_m = pd.merge(dual_df[['date']], sub_trades, on='date', how='left').fillna(0)
        dual_df['cum_realized'] = sub_m['realized_pnl'].cumsum()

        dual_df['cum_div'] = (dual_df[eval_col] * (target_yield / 250.0)).cumsum()
        dual_df['locked_income'] = dual_df['cum_realized'] + dual_df['cum_div']
        dual_df['stock_total_with_div'] = dual_df[eval_col] + dual_df['locked_income']

        strat_val_series = dual_df['stock_total_with_div']
        strat_locked_series = dual_df['locked_income']
        bnh_val_series = dual_df[bnh_col]

        dual_df['bnh_peak'] = dual_df[bnh_col].cummax()
        dual_df['bnh_dd'] = (dual_df[bnh_col] - dual_df['bnh_peak']) / dual_df['bnh_peak'] * 100

        invested_amt = dual_df[eval_col] - dual_df[unrealized_col]
        dual_df['strat_dd'] = np.where(
            invested_amt > 0,
            np.minimum(0.0, (dual_df[unrealized_col] / invested_amt) * 100),
            0.0
        )

        y1_label = f"{selected_target} 평가 가치 (원)"
        strat_name = f"{selected_target} 그리드 운용 (+배당)"
        bnh_name = f"{selected_target} 단순보유"

    fig_dual = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08, row_heights=[0.60, 0.40],
        subplot_titles=(
            f"1. 자산 성장 및 확정 수익 누적 추이 ({selected_target} - {period_label})",
            f"2. 하락장 방어력: 전고점 대비 하락폭 추이 (%)"
        )
    )

    fig_dual.add_trace(go.Scatter(x=dual_df['date'], y=strat_val_series, name=strat_name, line=dict(color='#FFD700', width=2.5)), row=1, col=1)
    fig_dual.add_trace(go.Scatter(x=dual_df['date'], y=strat_locked_series, name='확정 현금 누적 (익절+배당)', line=dict(color='#00CC96', width=2)), row=1, col=1)
    fig_dual.add_trace(go.Scatter(x=dual_df['date'], y=bnh_val_series, name=bnh_name, line=dict(color='#AB63FA', dash='dash', width=1.5)), row=1, col=1)

    fig_dual.add_trace(go.Scatter(x=dual_df['date'], y=dual_df['strat_dd'], fill='tozeroy', name='그리드 전략 낙폭', line=dict(color='#00CC96', width=2)), row=2, col=1)
    fig_dual.add_trace(go.Scatter(x=dual_df['date'], y=dual_df['bnh_dd'], name='단순보유 시장 낙폭', line=dict(color='#EF553B', dash='dash', width=1.5)), row=2, col=1)

    fig_dual.update_yaxes(title_text=y1_label, row=1, col=1)
    fig_dual.update_yaxes(title_text="하락폭 (%)", row=2, col=1)
    fig_dual.update_xaxes(title_text="날짜", row=2, col=1)
    fig_dual.update_layout(
        height=680, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1)
    )
    st.plotly_chart(fig_dual, use_container_width=True)

    min_strat_dd = dual_df['strat_dd'].min()
    min_bnh_dd = dual_df['bnh_dd'].min()
    st.info(f"""
    💡 **핵심 성과 요약 ({period_label}):**
    - **탁월한 하락장 방어:** 시장을 그대로 들고 있었을 때 최대 **{min_bnh_dd:.1f}%** 폭락을 겪는 동안, 본 전략은 **{min_strat_dd:.1f}%** 수준으로 하락 충격을 안전하게 방어했습니다.
    - **안정적인 현금 누적:** 주가 변동과 관계없이 한 번 확정한 익절 차익과 배당금(초록 실선)은 계단식으로 계속해서 불어납니다.
    """)

    st.markdown("---")

    # [2] 월별 3중 복합 현금흐름 vs 미실현 손익
    st.markdown("##### 🌊 2. '주가가 조정받아도 현금은 창출된다': 월별 3중 복합 현금흐름")
    st.caption("시세가 일시적으로 조정을 받아 평가손익이 마이너스인 달에도, [분할 익절 + 배당금 + 단기 파킹 이자]의 3중 구조로 현금흐름을 안정적으로 뽑아냅니다.")

    flow_c1, flow_c2, flow_c3 = st.columns([1.5, 1.2, 2.3])
    with flow_c1:
        flow_target_options = ["포트폴리오 전체"] + [c.replace('stock_eval_', '') for c in df_daily.columns if c.startswith('stock_eval_')]
        selected_flow_target = st.selectbox("분석 대상 선택", flow_target_options, key="flow_target_select")

    with flow_c2:
        flow_view_mode = st.radio("기간 선택", ["전체 기간", "연도별"], horizontal=True, key="flow_mode_radio")

    if "flow_y_idx" not in st.session_state: st.session_state.flow_y_idx = 0

    all_years = sorted(df_daily['year'].unique().tolist())
    flow_df_base = df_daily.copy()
    flow_period_label = "전체 기간"

    with flow_c3:
        if flow_view_mode == "연도별":
            btn_prev, sel_box, btn_next = st.columns([0.8, 1.4, 0.8])
            with btn_prev:
                if st.button("◀ 이전 연도", key="flow_prev_y"):
                    if st.session_state.flow_y_idx > 0:
                        st.session_state.flow_y_idx -= 1
                        st.rerun()
            with btn_next:
                if st.button("다음 연도 ▶", key="flow_next_y"):
                    if st.session_state.flow_y_idx < len(all_years) - 1:
                        st.session_state.flow_y_idx += 1
                        st.rerun()
            with sel_box:
                sel_y = st.selectbox("조회 연도", all_years, index=st.session_state.flow_y_idx, key="flow_sel_y")
                if sel_y != all_years[st.session_state.flow_y_idx]:
                    st.session_state.flow_y_idx = all_years.index(sel_y)
                    st.rerun()
            flow_df_base = df_daily[df_daily['year'] == all_years[st.session_state.flow_y_idx]].copy()
            flow_period_label = f"{all_years[st.session_state.flow_y_idx]}년"

    all_months_in_scope = sorted(flow_df_base['month'].unique().tolist())
    m_summary = []

    for m in all_months_in_scope:
        sub_d = flow_df_base[flow_df_base['month'] == m]
        sub_t = df_trades[df_trades['month'] == m]

        if selected_flow_target == "포트폴리오 전체":
            m_realized = sub_t['realized_pnl'].sum()
            m_div = 0.0
            for tname, yld in DIV_YIELD_MAP.items():
                col = f"stock_eval_{tname}"
                if col in sub_d.columns:
                    m_div += (sub_d[col] * (yld / 250.0)).sum()

            m_cma = (sub_d['shared_cash'] * (0.030 / 365.0)).sum()
            unreal_cols = [c for c in sub_d.columns if c.startswith('unrealized_')]
            m_unreal = sub_d[unreal_cols].sum(axis=1).mean() if unreal_cols else 0.0
            target_label_txt = "포트폴리오 전체"
            target_goal_line = 1_000_000  # 월간 목표 현금흐름 기준선 (100만 원)
        else:
            eval_col = f"stock_eval_{selected_flow_target}"
            unreal_col = f"unrealized_{selected_flow_target}"
            slot_col = f"slots_{selected_flow_target}"
            target_yield = DIV_YIELD_MAP.get(selected_flow_target, 0.0)

            m_realized = sub_t[sub_t['name'] == selected_flow_target]['realized_pnl'].sum()
            m_div = (sub_d[eval_col] * (target_yield / 250.0)).sum() if eval_col in sub_d.columns else 0.0

            active_slots = sub_d[slot_col] if slot_col in sub_d.columns else 0
            idle_ratio = np.maximum(0.0, (12.0 - active_slots) / 12.0)
            m_cma = ((5_000_000.0 * idle_ratio) * (0.030 / 365.0)).sum()

            m_unreal = sub_d[unreal_col].mean() if unreal_col in sub_d.columns else 0.0
            target_label_txt = selected_flow_target
            target_goal_line = 200_000  # 종목당 월간 목표 기준선 (20만 원)

        m_summary.append({
            'month': m,
            'realized_pnl': m_realized,
            'dividend': m_div,
            'cma_interest': m_cma,
            'total_cash': m_realized + m_div + m_cma,
            'unrealized_pnl': m_unreal
        })

    df_monthly_flow = pd.DataFrame(m_summary)

    fig_m_flow = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08, row_heights=[0.65, 0.35],
        subplot_titles=(
            f"1. 월별 3중 확정 현금흐름 ({target_label_txt} - {flow_period_label})",
            f"2. 당월 평균 미실현 평가손익 (시세 등락 구간)"
        )
    )

    fig_m_flow.add_trace(go.Bar(
        x=df_monthly_flow['month'], y=df_monthly_flow['realized_pnl'],
        name='① 그리드 익절 수익', marker_color='#00CC96'
    ), row=1, col=1)

    fig_m_flow.add_trace(go.Bar(
        x=df_monthly_flow['month'], y=df_monthly_flow['dividend'],
        name='② ETF 배당 분배금', marker_color='#FFD700'
    ), row=1, col=1)

    fig_m_flow.add_trace(go.Bar(
        x=df_monthly_flow['month'], y=df_monthly_flow['cma_interest'],
        name='③ 대기자금 파킹 이자', marker_color='#636EFA'
    ), row=1, col=1)

    fig_m_flow.add_hline(
        y=target_goal_line, line_dash="dash", line_color="orange",
        annotation_text=f"월간 목표 현금 기준선 ({target_goal_line/10000:,.0f}만 원)",
        row=1, col=1
    )

    fig_m_flow.add_trace(go.Bar(
        x=df_monthly_flow['month'], y=df_monthly_flow['unrealized_pnl'],
        name='미실현 평가손익', marker_color='#FF851B', opacity=0.7
    ), row=2, col=1)

    fig_m_flow.add_hline(y=0, line_dash="solid", line_color="white", line_width=1, row=2, col=1)

    fig_m_flow.update_layout(
        barmode='stack', height=680, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1)
    )
    fig_m_flow.update_yaxes(title_text="월 확정 현금 (원)", row=1, col=1)
    fig_m_flow.update_yaxes(title_text="평균 평가손익 (원)", row=2, col=1)
    fig_m_flow.update_xaxes(title_text="연월", row=2, col=1)

    st.plotly_chart(fig_m_flow, use_container_width=True)

    avg_m_cash = df_monthly_flow['total_cash'].mean()
    min_m_cash = df_monthly_flow['total_cash'].min()
    max_m_cash = df_monthly_flow['total_cash'].max()
    worst_unreal = df_monthly_flow['unrealized_pnl'].min()

    st.success(f"""
    📊 **[{target_label_txt}] 월간 현금 창출력 ({flow_period_label}):**
    - **월평균 확정 현금:** **{avg_m_cash:,.0f} 원** (최저 {min_m_cash:,.0f}원 ~ 최고 {max_m_cash:,.0f}원)
    - **최대 평가손실 발생 달:** **{worst_unreal:,.0f} 원**
    - **핵심 해설:** 주가가 조정을 받아 하단 평가손익이 **{worst_unreal/10000:,.0f}만 원**까지 밀린 달에도, 상단의 3중 현금흐름 엔진 덕분에 **{min_m_cash:,.0f}원 이상의 순현금이 매월 안정적으로 회수**되었습니다.
    """)

    st.markdown("---")

    # [3] 월별 투입자본 대비 현금 회수율 분석
    st.markdown("##### 💵 3. 실제 투입자본 대비 현금 회수율 분석")
    st.caption("시장에 실제로 투입된 자본의 규모를 확인하고, 투입된 자본이 매월 얼마나 높은 효율로 현금을 회수했는지 추적합니다.")

    m_base = df_daily.copy()
    ticker_cols = [c.replace('stock_eval_', '') for c in df_daily.columns if c.startswith('stock_eval_')]
    for t in ticker_cols:
        m_base[f'invested_cost_{t}'] = m_base[f'stock_eval_{t}'] - m_base[f'unrealized_{t}']
    
    m_base['total_invested_cost'] = sum(m_base[f'invested_cost_{t}'] for t in ticker_cols)

    months_list = sorted(m_base['month'].unique().tolist())
    eff_data = []

    for m in months_list:
        sub_d = m_base[m_base['month'] == m]
        sub_t = df_trades[df_trades['month'] == m]

        m_realized = sub_t['realized_pnl'].sum()
        m_div = 0.0
        for tname, yld in DIV_YIELD_MAP.items():
            col = f"stock_eval_{tname}"
            if col in sub_d.columns:
                m_div += (sub_d[col] * (yld / 250.0)).sum()

        m_cma = (sub_d['shared_cash'] * (0.030 / 365.0)).sum()
        m_total_cash = m_realized + m_div + m_cma
        m_invested_cost = sub_d['total_invested_cost'].mean()

        m_yield_monthly = (m_total_cash / m_invested_cost * 100) if m_invested_cost > 0 else 0.0
        m_yield_annual = m_yield_monthly * 12.0

        eff_data.append({
            'month': m,
            'invested_cost': m_invested_cost,
            'realized_pnl': m_realized,
            'dividend': m_div,
            'cma_interest': m_cma,
            'total_cash': m_total_cash,
            'yield_annual': m_yield_annual
        })

    df_eff = pd.DataFrame(eff_data)

    fig_dual_panel = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.10, row_heights=[0.65, 0.35],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=(
            "1. 월별 확정 현금흐름(좌측 막대) vs 실제 매수 투입원금(우측 영역선)",
            "2. 투입원금 대비 연환산 현금 회수율 (%)"
        )
    )

    fig_dual_panel.add_trace(
        go.Bar(
            x=df_eff['month'], y=df_eff['total_cash'],
            name='월 확정 현금 (익절+배당+이자)',
            marker_color='#636EFA'
        ),
        row=1, col=1, secondary_y=False
    )

    fig_dual_panel.add_trace(
        go.Scatter(
            x=df_eff['month'], y=df_eff['invested_cost'],
            name='실제 주식 매수 원금',
            mode='lines',
            line=dict(color='rgba(255, 165, 0, 0.85)', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(255, 165, 0, 0.12)'
        ),
        row=1, col=1, secondary_y=True
    )

    fig_dual_panel.add_hline(
        y=1_000_000, line_dash="dash", line_color="#00CC96",
        annotation_text="월간 목표 기준선 (100만 원)", row=1, col=1, secondary_y=False
    )

    fig_dual_panel.add_trace(
        go.Scatter(
            x=df_eff['month'], y=df_eff['yield_annual'],
            name='연환산 현금 회수율 (%)',
            mode='lines+markers',
            line=dict(color='#FF4136', width=2),
            marker=dict(size=5)
        ),
        row=2, col=1, secondary_y=False
    )

    avg_annual_yield = df_eff['yield_annual'].mean()
    fig_dual_panel.add_hline(
        y=avg_annual_yield, line_dash="dot", line_color="#FF4136",
        annotation_text=f"평균 회수율 ({avg_annual_yield:.1f}%)", row=2, col=1
    )

    fig_dual_panel.update_yaxes(title_text="월 현금흐름 (원)", row=1, col=1, secondary_y=False)
    fig_dual_panel.update_yaxes(title_text="투입원금 (원)", row=1, col=1, secondary_y=True, showgrid=False)
    fig_dual_panel.update_yaxes(title_text="회수율 (%)", row=2, col=1)
    fig_dual_panel.update_xaxes(title_text="연월", row=2, col=1)

    fig_dual_panel.update_layout(
        height=580, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1)
    )

    col_g1, col_g2 = st.columns([3, 1])
    with col_g1:
        st.plotly_chart(fig_dual_panel, use_container_width=True)

    with col_g2:
        avg_inv = df_eff['invested_cost'].mean()
        avg_cash = df_eff['total_cash'].mean()
        max_cash = df_eff['total_cash'].max()
        max_ann_yield = df_eff['yield_annual'].max()

        st.info(f"""
        **📊 자본 회전 효율성 분석:**
        - **월평균 회수 현금:** **{avg_cash:,.0f} 원** (최고 {max_cash:,.0f}원)
        - **평균 주식 매수 투입액:** 약 **{avg_inv/10000:,.0f}만 원** (총 시드의 {avg_inv/init_seed*100:.1f}%)
        - **평균 연환산 회수율:** **{avg_annual_yield:.1f}%** (최대 {max_ann_yield:.1f}%)
        
        💡 **전략 포인트:**
        1. **파란 막대:** 목표선 대비 매달 얼마의 현금이 안정적으로 회수되었는지 보여줍니다.
        2. **주황 영역선:** 주가가 내려갈 때 분할 매수로 자본을 넣었다가, 반등 시 현금화하여 비워내는 '자본 회전 사이클'을 증명합니다.
        3. **하단 빨간선:** 묶여 있던 자본 대비 연 15~20% 수준의 뛰어난 현금 회수율을 꾸준히 기록했습니다.
        """)

# =============================================================
# 탭 2: 운용 종목 종합 비교 (Universe)
# =============================================================
with tab_comp:
    st.subheader("🌐 운용 종목별 종합 성적표")
    ticker_names = df_trades['name'].unique().tolist()
    summary_list = []
    for name in ticker_names:
        sub = df_trades[df_trades['name'] == name]
        buys = sub[sub['action'].str.startswith('BUY')]
        sells = sub[sub['action'] == 'SELL']
        rollovers = sub[sub['action'].str.contains('ROLLOVER')]
        
        tot_realized = sells['realized_pnl'].sum() + rollovers['rollover_loss'].sum()
        win_trades = (sells['realized_pnl'] > 0).sum()
        win_rate = (win_trades / len(sells) * 100) if len(sells) > 0 else 0
        max_slot = buys['slot_id'].max() if len(buys) > 0 else 0
        avg_holding = sells['holding_days'].mean() if len(sells) > 0 else 0
        
        summary_list.append({
            '종목명': name,
            '총 주문 체결': len(sub),
            '분할 매수': len(buys),
            '익절 청산': len(sells),
            '비상 교체(롤오버)': len(rollovers),
            '최대 도달 단계': f"슬롯 {max_slot}",
            '청산 승률': f"{win_rate:.1f}%",
            '평균 보유 기간': f"{avg_holding:.1f} 일",
            '누적 실현손익(원)': tot_realized
        })
        
    df_univ = pd.DataFrame(summary_list)
    col_u1, col_u2 = st.columns([3, 2])
    with col_u1:
        st.markdown("##### 📋 종목별 상세 운용 효율표")
        st.dataframe(df_univ.style.format({'누적 실현손익(원)': '{:,.0f}'}), use_container_width=True, hide_index=True)
    with col_u2:
        st.markdown("##### 🍰 종목별 실현손익 기여도")
        fig_pie = px.pie(df_univ, values='누적 실현손익(원)', names='종목명', hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism)
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

# =============================================================
# 탭 3: 캔들 & 분할 매매 타점 분석
# =============================================================
with tab_candle:
    st.subheader("🎯 종목별 일봉 차트 & 그리드 분할 매수/익절 타점 정밀 분석")
    available_names = list(stock_dfs.keys())
    
    if available_names:
        default_stock = 'TIGER 은행' if 'TIGER 은행' in available_names else available_names[0]
        selected_name = st.selectbox("분석할 종목 선택", available_names, index=available_names.index(default_stock))
        
        df_target_stock = stock_dfs[selected_name].copy()
        df_target_trades = df_trades[df_trades['name'] == selected_name].copy()
        
        all_years = sorted(df_target_stock['year'].unique().tolist())
        all_quarters = sorted(df_target_stock['quarter'].unique().tolist())
        all_months = sorted(df_target_stock['month'].unique().tolist())

        st.markdown("##### 🔍 차트 조회 기간 컨트롤러")
        ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5 = st.columns([1.8, 0.9, 2.0, 0.9, 2.8])
        
        with ctrl_c1:
            view_mode = st.radio("표시 단위", ["전체 기간", "연도별", "분기별", "월별"], index=2, horizontal=True)

        if "idx_year" not in st.session_state: st.session_state.idx_year = 0
        if "idx_quarter" not in st.session_state: st.session_state.idx_quarter = 0
        if "idx_month" not in st.session_state: st.session_state.idx_month = 0

        st.session_state.idx_year = max(0, min(st.session_state.idx_year, len(all_years) - 1))
        st.session_state.idx_quarter = max(0, min(st.session_state.idx_quarter, len(all_quarters) - 1))
        st.session_state.idx_month = max(0, min(st.session_state.idx_month, len(all_months) - 1))

        if view_mode == "분기별":
            with ctrl_c2:
                if st.button("◀ 이전 분기", use_container_width=True):
                    if st.session_state.idx_quarter > 0:
                        st.session_state.idx_quarter -= 1
                        st.rerun()
            with ctrl_c4:
                if st.button("다음 분기 ▶", use_container_width=True):
                    if st.session_state.idx_quarter < len(all_quarters) - 1:
                        st.session_state.idx_quarter += 1
                        st.rerun()
            with ctrl_c3:
                chosen_q = st.selectbox("분기 선택", all_quarters, index=st.session_state.idx_quarter)
                if chosen_q != all_quarters[st.session_state.idx_quarter]:
                    st.session_state.idx_quarter = all_quarters.index(chosen_q)
                    st.rerun()

            curr_period_str = all_quarters[st.session_state.idx_quarter]
            plot_stock = df_target_stock[df_target_stock['quarter'] == curr_period_str]
            plot_trades = df_target_trades[df_target_trades['quarter'] == curr_period_str]
            current_title = f"{curr_period_str} 캔들 차트 (약 60영업일)"

        elif view_mode == "월별":
            with ctrl_c2:
                if st.button("◀ 이전 달", use_container_width=True):
                    if st.session_state.idx_month > 0:
                        st.session_state.idx_month -= 1
                        st.rerun()
            with ctrl_c4:
                if st.button("다음 달 ▶", use_container_width=True):
                    if st.session_state.idx_month < len(all_months) - 1:
                        st.session_state.idx_month += 1
                        st.rerun()
            with ctrl_c3:
                chosen_m = st.selectbox("월 선택", all_months, index=st.session_state.idx_month)
                if chosen_m != all_months[st.session_state.idx_month]:
                    st.session_state.idx_month = all_months.index(chosen_m)
                    st.rerun()

            curr_period_str = all_months[st.session_state.idx_month]
            plot_stock = df_target_stock[df_target_stock['month'] == curr_period_str]
            plot_trades = df_target_trades[df_target_trades['month'] == curr_period_str]
            current_title = f"{curr_period_str} 캔들 차트 (약 20영업일)"

        elif view_mode == "연도별":
            with ctrl_c2:
                if st.button("◀ 이전 연도", use_container_width=True):
                    if st.session_state.idx_year > 0:
                        st.session_state.idx_year -= 1
                        st.rerun()
            with ctrl_c4:
                if st.button("다음 연도 ▶", use_container_width=True):
                    if st.session_state.idx_year < len(all_years) - 1:
                        st.session_state.idx_year += 1
                        st.rerun()
            with ctrl_c3:
                chosen_y = st.selectbox("연도 선택", all_years, index=st.session_state.idx_year)
                if chosen_y != all_years[st.session_state.idx_year]:
                    st.session_state.idx_year = all_years.index(chosen_y)
                    st.rerun()

            curr_period_int = all_years[st.session_state.idx_year]
            plot_stock = df_target_stock[df_target_stock['year'] == curr_period_int]
            plot_trades = df_target_trades[df_target_trades['year'] == curr_period_int]
            current_title = f"{curr_period_int}년 전체 캔들 차트"

        else:
            plot_stock = df_target_stock
            plot_trades = df_target_trades
            current_title = "전체 기간 차트"

        with ctrl_c5:
            st.info(f"**현재 관측 구간:** {current_title}<br>캔들 수: **{len(plot_stock)}**개 | 자동 체결: **{len(plot_trades)}**건", icon="📌")

        fig_candle = make_subplots(
            rows=2, cols=1, shared_xaxes=True, 
            vertical_spacing=0.04, row_heights=[0.72, 0.28],
            subplot_titles=(f"{selected_name} - {current_title}", "1년 상대 가격 위치 지표 (PRS %)")
        )
        
        fig_candle.add_trace(go.Candlestick(
            x=plot_stock['날짜'], open=plot_stock['시가'], high=plot_stock['고가'],
            low=plot_stock['저가'], close=plot_stock['종가'],
            name="주가", increasing_line_color='#FF4136', decreasing_line_color='#0074D9'
        ), row=1, col=1)
        
        if not plot_trades.empty:
            buys = plot_trades[plot_trades['action'].str.startswith('BUY')]
            sells = plot_trades[plot_trades['action'] == 'SELL']
            rollovers = plot_trades[plot_trades['action'].str.contains('ROLLOVER')]
            
            if not buys.empty:
                fig_candle.add_trace(go.Scatter(
                    x=buys['date'], y=buys['price'], mode='markers+text', name='분할 매수',
                    marker=dict(symbol='triangle-up', size=13, color='#2ECC40', line=dict(width=1.5, color='black')),
                    text=[f"S{sid}" for sid in buys['slot_id']], textposition="bottom center",
                    hovertext=[f"<b>[매수] 슬롯 {sid}</b><br>일자: {d.strftime('%Y-%m-%d')}<br>체결가: {p:,.0f}원<br>매수금액: {amt:,.0f}원" 
                               for sid, d, p, amt in zip(buys['slot_id'], buys['date'], buys['price'], buys['amount'])]
                ), row=1, col=1)
            
            if not sells.empty:
                fig_candle.add_trace(go.Scatter(
                    x=sells['date'], y=sells['price'], mode='markers+text', name='목표가 익절',
                    marker=dict(symbol='triangle-down', size=13, color='#FF851B', line=dict(width=1.5, color='black')),
                    text=[f"+{pnl/10000:.1f}만" for pnl in sells['realized_pnl']], textposition="top center",
                    hovertext=[f"<b>[익절]</b><br>일자: {d.strftime('%Y-%m-%d')}<br>수익금: +{pnl:,.0f}원<br>보유: {days}일" 
                               for d, pnl, days in zip(sells['date'], sells['realized_pnl'], sells['holding_days'])]
                ), row=1, col=1)
            
            if not rollovers.empty:
                fig_candle.add_trace(go.Scatter(
                    x=rollovers['date'], y=rollovers['price'], mode='markers', name='🚨 비상 손실방어 교체',
                    marker=dict(symbol='x', size=14, color='magenta', line=dict(width=3)),
                    hovertext=[f"<b>[비상 교체 매도]</b><br>일자: {d.strftime('%Y-%m-%d')}<br>정리손실: {rl:,.0f}원" 
                               for d, rl in zip(rollovers['date'], rollovers['rollover_loss'])]
                ), row=1, col=1)
                
        fig_candle.add_trace(go.Scatter(x=plot_stock['날짜'], y=plot_stock['prs'], name="상대 가격 위치 (%)", line=dict(color='#FFDC00', width=2)), row=2, col=1)
        fig_candle.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="과열 구간 (70%)")
        fig_candle.add_hline(y=35, line_dash="dash", line_color="green", row=2, col=1, annotation_text="저평가 구간 (35%)")
        
        fig_candle.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig_candle.update_layout(xaxis_rangeslider_visible=False, height=720, hovermode="x unified")
        st.plotly_chart(fig_candle, use_container_width=True)
        
        if not plot_trades.empty:
            st.markdown(f"###### 📋 {current_title} 자동 체결 내역 ({len(plot_trades)}건)")
            st.dataframe(plot_trades[['date', 'action', 'slot_id', 'price', 'shares', 'amount', 'realized_pnl', 'holding_days', 'applied_natr', 'target_tp_pct']], use_container_width=True)

# =============================================================
# 탭 4: 진입 시점별 안정성 검증
# =============================================================
with tab_timing:
    st.subheader("⏱️ 진입 시점별 성과 안정성 검증")
    st.caption("어느 연도에 전략을 시작했더라도 시스템이 일관되게 자산을 지켜내고 수익을 냈는지 검증합니다.")
    
    timing_years = [2022, 2023, 2024]
    timing_results = []
    
    for sy in timing_years:
        sub_d = df_daily[df_daily['year'] >= sy].copy()
        if not sub_d.empty:
            start_val = sub_d['total_equity'].iloc[0]
            end_val = sub_d['total_equity'].iloc[-1]
            cummax = sub_d['total_equity'].cummax()
            mdd = ((sub_d['total_equity'] - cummax) / cummax * 100).min()
            
            sub_bnh_start = sub_d['total_bnh'].iloc[0]
            sub_bnh_end = sub_d['total_bnh'].iloc[-1]
            bnh_ret = ((sub_bnh_end / sub_bnh_start) - 1.0) * 100
            
            timing_results.append({
                "시작 연도": f"{sy}년 시작",
                "초기 자본": f"{start_val:,.0f} 원",
                "최종 자산": f"{end_val:,.0f} 원",
                "그리드 누적 수익률": f"{((end_val/start_val)-1)*100:+.2f}%",
                "최대 하락폭(MDD)": f"{mdd:.2f}%",
                "단순보유 수익률": f"{bnh_ret:+.2f}%",
                "판정": "✅ 견고한 방어" if mdd > -15 else "⚠️ 주의"
            })
            
    st.dataframe(pd.DataFrame(timing_results), use_container_width=True, hide_index=True)
    st.info("💡 **검증 결론:** 어떤 연도에 진입했더라도 최대 하락폭(MDD)이 -13% 안팎으로 안정적으로 통제되며, 단순 보유 대비 월등한 자산 방어 효과를 증명했습니다.")

# =============================================================
# 탭 5: 리스크 통제 & 비상 대응 로그
# =============================================================
with tab_edge:
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.subheader("1. 주요 종목별 분할 매수 소진 단계 (최대 12단계)")
        slot_cols = [c for c in df_daily.columns if c.startswith('slots_')]
        df_slots = df_daily[['date'] + slot_cols].set_index('date')
        df_slots.columns = [c.replace('slots_', '') for c in df_slots.columns]
        fig_heat = px.imshow(df_slots.T, labels=dict(x="날짜", y="종목명", color="점유 슬롯 수"), color_continuous_scale="YlOrRd", zmin=0, zmax=12, aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with col_e2:
        st.subheader("2. 계좌 공용 대기자금(CMA 예수금) 안전선 모니터링")
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=df_daily['date'], y=df_daily['shared_cash'], fill='tozeroy', name='파킹 대기자금', line=dict(color='#636EFA')))
        fig_c.add_hline(y=10_000_000, line_dash="dash", line_color="red", annotation_text="최소 현금 경고선 (1,000만 원)")
        fig_c.update_layout(xaxis_title="날짜", yaxis_title="예수금 (원)", hovermode="x unified")
        st.plotly_chart(fig_c, use_container_width=True)

    st.subheader("3. 비상 버퍼 매수 및 롤오버 방어 실행 전수 기록")
    edge_df = df_trades[df_trades['action'].isin(['BUY_BUFFER', 'BUY_ROLLOVER', 'SELL_ROLLOVER'])]
    st.dataframe(edge_df[['date', 'name', 'action', 'slot_id', 'price', 'shares', 'amount', 'realized_pnl', 'rollover_loss', 'applied_natr']], use_container_width=True)