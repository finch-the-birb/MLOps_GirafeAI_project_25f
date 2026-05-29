from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from omegaconf import OmegaConf

from stock_forecaster.data.dataset import load_processed_frame
from stock_forecaster.data.inference_window import (
    ForwardReturnOutcome,
    build_dynamic_inference_sample,
)

API_URL = "http://127.0.0.1:8001/api/v1/predict"
DATA_PATH = Path("data/processed/fnspid_subset_thr03.parquet")
LABEL_THRESHOLD_PCT = 0.3
SEQ_LEN = 30

DATA_CFG = {
    "processed_file": str(DATA_PATH),
    "seq_len": SEQ_LEN,
    "feature_columns": ["open", "high", "low", "close", "volume", "change_pct"],
    "max_news_per_window": 32,
    "max_news_chars": 512,
    "max_news_chars_per_day": 256,
    "label_threshold_pct": LABEL_THRESHOLD_PCT,
}


@st.cache_data(show_spinner=False)
def load_market_data() -> tuple[pd.DataFrame, list[str], dict[str, list[date]]]:
    """Tickers with 2023 rows and their trading dates (calendar subset)."""
    frame = load_processed_frame(DATA_PATH)
    frame["date"] = pd.to_datetime(frame["date"])
    frame_2023 = frame[frame["date"].dt.year == 2023].copy()
    tickers = sorted(frame_2023["ticker"].unique().tolist())
    trading_dates: dict[str, list[date]] = {}
    for ticker in tickers:
        dates = frame_2023.loc[frame_2023["ticker"] == ticker, "date"].dt.date.unique().tolist()
        trading_dates[ticker] = sorted(dates)
    return frame, tickers, trading_dates


def build_price_chart(
    series: pd.DataFrame,
    *,
    ticker: str,
    target_dt: date,
    window_start: date | None,
    window_end: date | None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["close"],
            mode="lines",
            name="Close",
            hovertemplate="Дата: %{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
        )
    )

    target_row = series[series["date"].dt.date == target_dt]
    if not target_row.empty:
        close_val = float(target_row.iloc[0]["close"])
        fig.add_trace(
            go.Scatter(
                x=[target_row.iloc[0]["date"]],
                y=[close_val],
                mode="markers+text",
                name="Дата таргета",
                marker={"size": 14, "color": "crimson", "symbol": "diamond"},
                text=[f"{close_val:.2f}"],
                textposition="top center",
            )
        )

    if window_start is not None and window_end is not None:
        window_series = series[
            (series["date"].dt.date >= window_start) & (series["date"].dt.date <= window_end)
        ]
        if not window_series.empty:
            fig.add_vrect(
                x0=window_series["date"].iloc[0],
                x1=window_series["date"].iloc[-1],
                fillcolor="rgba(99, 110, 250, 0.12)",
                line_width=0,
                annotation_text=f"Окно модели ({SEQ_LEN} торг. дней)",
                annotation_position="top left",
            )

    fig.update_layout(
        title=f"{ticker} — close (окно может включать хвост 2022)",
        xaxis_title="Дата",
        yaxis_title="Close",
        hovermode="x unified",
        dragmode="zoom",
        height=520,
    )
    return fig


def _outcome_from_api(payload: dict) -> ForwardReturnOutcome | None:
    if payload.get("actual_forward_return_pct") is None:
        return None
    threshold = float(payload.get("label_threshold_pct", LABEL_THRESHOLD_PCT))
    forward = float(payload["actual_forward_return_pct"])
    meets = forward > threshold
    return ForwardReturnOutcome(
        target_date=str(payload.get("target_date", "")),
        close_on_target=0.0,
        close_next=None,
        forward_return_pct=forward,
        threshold_pct=threshold,
        meets_threshold=meets,
        direction=str(payload.get("actual_direction") or ("UP" if meets else "DOWN")),
    )


def main() -> None:
    st.set_page_config(page_title="Stock Forecaster", layout="wide")
    st.title("Multimodal Stock Forecaster")
    st.caption(
        "Hybrid **gated_fusion**, thr **0.3%**. Любой **торговый** день 2023; "
        f"окно {SEQ_LEN} дней строится на лету (при необходимости с хвостом 2022)."
    )

    try:
        frame_all, tickers, trading_dates = load_market_data()
    except FileNotFoundError:
        st.error(f"Не найден датасет: `{DATA_PATH}`.")
        return

    if not tickers:
        st.error("Нет тикеров с данными за 2023 год.")
        return

    with st.sidebar:
        st.header("Запрос к модели")
        ticker = st.selectbox("Тикер", tickers, index=0)
        ticker_dates = set(trading_dates.get(ticker, []))

        target_dt = st.date_input(
            "Дата таргета (2023)",
            value=date(2023, 1, 3),
            min_value=date(2023, 1, 1),
            max_value=date(2023, 12, 31),
            format="YYYY-MM-DD",
            help="Календарь 2023. Если день не торговый — будет подсказка.",
        )
        if target_dt not in ticker_dates:
            st.warning(
                f"{target_dt.isoformat()} — не торговый день для {ticker} "
                "(выходной/праздник или нет строки в parquet)."
            )

        predict_clicked = st.button("Получить прогноз", type="primary", use_container_width=True)

    window_start: date | None = None
    window_end: date | None = None
    local_outcome: ForwardReturnOutcome | None = None
    if target_dt in ticker_dates:
        try:
            dynamic = build_dynamic_inference_sample(
                ticker,
                target_dt.isoformat(),
                OmegaConf.create(DATA_CFG),
            )
            window_start = date.fromisoformat(dynamic.window_start_date)
            window_end = date.fromisoformat(dynamic.window_end_date)
            local_outcome = dynamic.outcome
        except ValueError as exc:
            st.sidebar.error(str(exc))

    series = frame_all[frame_all["ticker"] == ticker].sort_values("date")
    chart_from = (
        date(2022, 11, 1) if window_start and window_start.year == 2022 else date(2023, 1, 1)
    )
    series_chart = series[series["date"].dt.date >= chart_from]

    col_chat, col_chart = st.columns([1, 2])

    with col_chat:
        st.subheader("Прогноз модели")
        if predict_clicked:
            if target_dt not in ticker_dates:
                st.error("Выберите торговый день для этого тикера.")
            else:
                with st.spinner("Inference API..."):
                    try:
                        resp = httpx.post(
                            API_URL,
                            json={"ticker": ticker, "target_date": target_dt.isoformat()},
                            timeout=120.0,
                        )
                        resp.raise_for_status()
                        payload = resp.json()
                    except httpx.HTTPError as exc:
                        st.error(f"Ошибка API: {exc}")
                    else:
                        direction = payload.get("prediction", "?")
                        prob = float(payload.get("probability", 0.0))
                        news_count = int(payload.get("news_analyzed", 0))
                        emoji = "📈" if direction == "UP" else "📉"
                        st.markdown(
                            f"**{emoji} {direction}** (вероятность UP: **{prob:.1%}**)  \n"
                            f"Новостей в окне: **{news_count}**"
                        )
                        if payload.get("window_start_date"):
                            st.caption(
                                f"Окно: `{payload['window_start_date']}` → "
                                f"`{payload['window_end_date']}` → таргет `{target_dt}`"
                            )
                        api_outcome = _outcome_from_api(
                            {**payload, "target_date": target_dt.isoformat()}
                        )
                        if api_outcome:
                            st.markdown("**Факт (next-day return)**")
                            st.markdown(api_outcome.summary_markdown)
        else:
            st.info("Выберите тикер и дату, затем нажмите «Получить прогноз».")

        if local_outcome and not predict_clicked:
            st.markdown("**Факт (next-day return)**")
            st.markdown(local_outcome.summary_markdown)

    with col_chart:
        if series_chart.empty:
            st.warning("Нет данных для графика.")
        else:
            fig = build_price_chart(
                series_chart,
                ticker=ticker,
                target_dt=target_dt,
                window_start=window_start,
                window_end=window_end,
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
