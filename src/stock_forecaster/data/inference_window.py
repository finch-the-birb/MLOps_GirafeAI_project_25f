"""Point-in-time inference windows built on the fly from processed parquet."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from stock_forecaster.data.dataset import SampleRecord, load_processed_frame

_PLACEHOLDER_DAY = "No news available for day."
_PLACEHOLDER_WINDOW = "No news available for window."


@dataclass(frozen=True, slots=True)
class ForwardReturnOutcome:
    """Actual next-day return relative to the label threshold."""

    target_date: str
    close_on_target: float
    close_next: float | None
    forward_return_pct: float | None
    threshold_pct: float
    meets_threshold: bool | None
    direction: str | None

    @property
    def summary_markdown(self) -> str:
        if self.forward_return_pct is None or self.meets_threshold is None:
            return "Нет следующего торгового дня — фактическое движение не посчитать."
        op = ">" if self.meets_threshold else "≤"
        direction = self.direction or ("UP" if self.meets_threshold else "DOWN")
        return (
            f"Фактическое движение (close → next close): **{self.forward_return_pct:+.3f}%**  \n"
            f"Условие метки: `{self.forward_return_pct:+.3f}% {op} {self.threshold_pct:.1f}%`  \n"
            f"→ **{direction}**"
        )


@dataclass(frozen=True, slots=True)
class DynamicInferenceSample:
    record: SampleRecord
    news_count: int
    outcome: ForwardReturnOutcome
    window_start_date: str
    window_end_date: str


def validate_target_date_2023(target_date: str) -> date:
    """Parse and ensure ``target_date`` is a valid calendar date in 2023."""
    try:
        parsed = datetime.strptime(target_date.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        msg = "target_date must use YYYY-MM-DD format"
        raise ValueError(msg) from exc
    if parsed.year != 2023:
        msg = "target_date must fall within calendar year 2023"
        raise ValueError(msg)
    return parsed


def _count_news(daily_news: tuple[str, ...], news_text: str) -> int:
    if daily_news:
        return sum(1 for text in daily_news if text.strip() and text.strip() != _PLACEHOLDER_DAY)
    return sum(
        1 for line in news_text.splitlines() if line.strip() and line.strip() != _PLACEHOLDER_WINDOW
    )


def _compute_outcome(
    ticker_frame: pd.DataFrame,
    target_idx: int,
    *,
    threshold_pct: float,
) -> ForwardReturnOutcome:
    target_row = ticker_frame.iloc[target_idx]
    target_date = str(target_row["date"].date())
    close_t = float(target_row["close"])
    close_next: float | None = None
    forward_return_pct: float | None = None
    meets_threshold: bool | None = None
    direction: str | None = None

    if target_idx + 1 < len(ticker_frame):
        next_row = ticker_frame.iloc[target_idx + 1]
        close_next = float(next_row["close"])
        forward_return_pct = (close_next - close_t) / close_t * 100.0
        meets_threshold = forward_return_pct > threshold_pct
        direction = "UP" if meets_threshold else "DOWN"

    return ForwardReturnOutcome(
        target_date=target_date,
        close_on_target=close_t,
        close_next=close_next,
        forward_return_pct=forward_return_pct,
        threshold_pct=threshold_pct,
        meets_threshold=meets_threshold,
        direction=direction,
    )


def _build_sample_at_index(
    ticker: str,
    ticker_frame: pd.DataFrame,
    target_idx: int,
    data_cfg: DictConfig,
) -> DynamicInferenceSample | None:
    """Build one sample when ``target_idx`` points to a valid target trading day."""
    seq_len = int(data_cfg.seq_len)
    feature_columns = list(data_cfg.feature_columns)
    max_news_per_window = int(data_cfg.max_news_per_window)
    max_news_chars = int(data_cfg.max_news_chars)
    max_news_chars_per_day = int(data_cfg.get("max_news_chars_per_day", 256))
    threshold_pct = float(data_cfg.get("label_threshold_pct", 0.5))

    if target_idx + 1 >= len(ticker_frame):
        return None
    end_idx = target_idx - 1
    if end_idx < seq_len - 1:
        return None

    window = ticker_frame.iloc[end_idx - seq_len + 1 : end_idx + 1]
    end_date = window["date"].max()
    news_mask = window["date"] <= end_date
    news_lines = window.loc[news_mask, "news"].dropna().astype(str).tolist()
    news_lines = [line for line in news_lines if line.strip()][:max_news_per_window]
    news_text = "\n".join(news_lines) if news_lines else _PLACEHOLDER_WINDOW
    if len(news_text) > max_news_chars:
        news_text = news_text[:max_news_chars]

    daily_news: list[str] = []
    for _, row in window.iterrows():
        day_text = str(row.get("news", "")).strip() or _PLACEHOLDER_DAY
        if len(day_text) > max_news_chars_per_day:
            day_text = day_text[:max_news_chars_per_day]
        daily_news.append(day_text)

    target_row = ticker_frame.iloc[target_idx]
    label = int(target_row["target_label"]) if pd.notna(target_row.get("target_label")) else 0
    features = window[feature_columns].to_numpy(dtype=np.float32)
    record = SampleRecord(
        ticker=ticker.upper(),
        target_date=str(target_row["date"].date()),
        time_series=features,
        news_text=news_text,
        daily_news=tuple(daily_news),
        label=label,
    )
    outcome = _compute_outcome(ticker_frame, target_idx, threshold_pct=threshold_pct)
    return DynamicInferenceSample(
        record=record,
        news_count=_count_news(record.daily_news, record.news_text),
        outcome=outcome,
        window_start_date=str(window["date"].iloc[0].date()),
        window_end_date=str(window["date"].iloc[-1].date()),
    )


def iter_dense_test_samples(data_cfg: DictConfig) -> list[DynamicInferenceSample]:
    """
    All test-year samples with stride=1 semantics (inference-style, 2022 tail allowed).

    One entry per (ticker, trading day) in ``test_years`` where a full lookback window
    and next-day label exist.
    """
    test_years = {int(y) for y in data_cfg.test_years}
    frame = load_processed_frame(Path(data_cfg.processed_file))
    frame["date"] = pd.to_datetime(frame["date"])

    samples: list[DynamicInferenceSample] = []
    for ticker, group in frame.groupby("ticker"):
        ticker_frame = group.sort_values("date").reset_index(drop=True)
        for target_idx in range(len(ticker_frame)):
            if ticker_frame.iloc[target_idx]["date"].year not in test_years:
                continue
            built = _build_sample_at_index(str(ticker), ticker_frame, target_idx, data_cfg)
            if built is not None:
                samples.append(built)
    return samples


def build_dynamic_inference_sample(
    ticker: str,
    target_date: str,
    data_cfg: DictConfig,
) -> DynamicInferenceSample:
    """
    Build a single inference window aligned with training semantics.

    ``target_date`` is a 2023 trading day. The model input window contains
    ``seq_len`` rows ending on the previous trading day; history may include 2022.
    """
    parsed_target = validate_target_date_2023(target_date)
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        msg = "ticker must not be empty"
        raise ValueError(msg)

    frame = load_processed_frame(Path(data_cfg.processed_file))
    frame["date"] = pd.to_datetime(frame["date"])
    ticker_frame = (
        frame[frame["ticker"].str.upper() == normalized_ticker]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if ticker_frame.empty:
        msg = f"Ticker {normalized_ticker} not found in processed dataset"
        raise ValueError(msg)

    target_positions = ticker_frame.index[ticker_frame["date"].dt.date == parsed_target].tolist()
    if not target_positions:
        msg = (
            f"No trading row for {normalized_ticker} on {parsed_target.isoformat()}. "
            "Pick a date present in the dataset (weekends/holidays are excluded)."
        )
        raise ValueError(msg)
    target_idx = int(target_positions[0])

    seq_len = int(data_cfg.seq_len)
    end_idx = target_idx - 1
    if end_idx < seq_len - 1:
        msg = (
            f"Not enough trading history before {parsed_target.isoformat()} "
            f"(need {seq_len} prior rows; extend parquet with 2022 tail)."
        )
        raise ValueError(msg)

    built = _build_sample_at_index(normalized_ticker, ticker_frame, target_idx, data_cfg)
    if built is None:
        msg = f"Cannot build window for {normalized_ticker} on {parsed_target.isoformat()}"
        raise ValueError(msg)
    return built
