"""Point-in-time aligned sliding-window dataset for multimodal samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRecord:
    ticker: str
    target_date: str
    time_series: np.ndarray
    news_text: str
    daily_news: tuple[str, ...]
    label: int


class FnspidWindowDataset(Dataset):
    """
    Sliding-window dataset with strict point-in-time news alignment.

    News for window ending at T includes only articles with date <= T.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        seq_len: int,
        feature_columns: list[str],
        years: list[int] | None = None,
        label_column: str = "target_label",
        max_news_per_window: int = 32,
        max_news_chars: int = 512,
        fusion_mode: str = "late",
        max_news_chars_per_day: int = 256,
        window_stride: int = 1,
    ) -> None:
        self.seq_len = seq_len
        self.window_stride = max(1, int(window_stride))
        self.feature_columns = feature_columns
        self.max_news_per_window = max_news_per_window
        self.max_news_chars = max_news_chars
        self.fusion_mode = fusion_mode
        self.max_news_chars_per_day = max_news_chars_per_day
        self.samples: list[SampleRecord] = []

        frame = data_frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        if years is not None:
            frame = frame[frame["date"].dt.year.isin(years)]

        for ticker, ticker_frame in frame.groupby("ticker"):
            sorted_frame = ticker_frame.sort_values("date").reset_index(drop=True)
            self._build_ticker_samples(sorted_frame, ticker, label_column)

    def _build_ticker_samples(
        self,
        group: pd.DataFrame,
        ticker: str,
        label_column: str,
    ) -> None:
        first_end = self.seq_len - 1
        for end_idx in range(first_end, len(group) - 1, self.window_stride):
            window = group.iloc[end_idx - self.seq_len + 1 : end_idx + 1]
            target_row = group.iloc[end_idx + 1]
            end_date = window["date"].max()

            news_mask = window["date"] <= end_date
            news_lines = window.loc[news_mask, "news"].dropna().astype(str).tolist()
            news_lines = [line for line in news_lines if line.strip()][: self.max_news_per_window]
            news_text = "\n".join(news_lines) if news_lines else "No news available for window."
            if len(news_text) > self.max_news_chars:
                news_text = news_text[: self.max_news_chars]

            daily_news: list[str] = []
            for _, row in window.iterrows():
                day_text = str(row.get("news", "")).strip()
                if not day_text:
                    day_text = "No news available for day."
                if len(day_text) > self.max_news_chars_per_day:
                    day_text = day_text[: self.max_news_chars_per_day]
                daily_news.append(day_text)

            features = window[self.feature_columns].to_numpy(dtype=np.float32)
            label = int(target_row[label_column])
            self.samples.append(
                SampleRecord(
                    ticker=ticker,
                    target_date=str(target_row["date"].date()),
                    time_series=features,
                    news_text=news_text,
                    daily_news=tuple(daily_news),
                    label=label,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        item: dict[str, object] = {
            "time_series": torch.from_numpy(sample.time_series),
            "news_text": sample.news_text,
            "daily_news": list(sample.daily_news),
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "ticker": sample.ticker,
            "target_date": sample.target_date,
        }
        return item


def load_processed_frame(processed_file: Path) -> pd.DataFrame:
    """Load parquet/csv processed dataset."""
    processed_file = Path(processed_file)
    if processed_file.suffix == ".parquet":
        return pd.read_parquet(processed_file)
    return pd.read_csv(processed_file, parse_dates=["date"])
