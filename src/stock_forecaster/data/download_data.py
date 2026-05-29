"""Download or pull FNSPID subset via DVC API with targeted HuggingFace file downloads."""

from __future__ import annotations

import logging
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

try:
    from dvc.api import DVCFileSystem
except ImportError:
    DVCFileSystem = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

HF_REPO_ID = "Zihan1004/FNSPID"
NEWS_FILENAME = "Stock_news/nasdaq_exteral_data.csv"
PRICE_ZIP_FILENAME = "Stock_price/full_history.zip"

# Top-50 liquid US tickers (subset for local training).
DEFAULT_TOP_TICKERS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "GOOG",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "BRK-B",
    "UNH",
    "JNJ",
    "V",
    "XOM",
    "JPM",
    "WMT",
    "MA",
    "PG",
    "LLY",
    "HD",
    "CVX",
    "MRK",
    "ABBV",
    "KO",
    "PEP",
    "COST",
    "AVGO",
    "MCD",
    "CSCO",
    "TMO",
    "ACN",
    "ABT",
    "DHR",
    "BAC",
    "DIS",
    "WFC",
    "CRM",
    "LIN",
    "AMD",
    "PM",
    "TXN",
    "INTC",
    "ORCL",
    "IBM",
    "QCOM",
    "GE",
    "CAT",
    "INTU",
    "AMAT",
    "MU",
    "GS",
    # Extended to 80 liquid names (indices 50-79).
    "UNP",
    "RTX",
    "HON",
    "LOW",
    "UPS",
    "MS",
    "SPGI",
    "BA",
    "DE",
    "BLK",
    "ELV",
    "CI",
    "PLD",
    "ADP",
    "GILD",
    "AMGN",
    "SYK",
    "MDLZ",
    "TJX",
    "C",
    "VZ",
    "CMCSA",
    "NKE",
    "MO",
    "SO",
    "DUK",
    "BMY",
    "SCHW",
    "PGR",
    "LMT",
)

TICKER_ALIASES: dict[str, str] = {
    "BRK.B": "BRK-B",
    "BRK/B": "BRK-B",
    "GOOG": "GOOGL",
}


def _normalize_ticker_symbol(symbol: str) -> str:
    cleaned = str(symbol).strip().upper()
    return TICKER_ALIASES.get(cleaned, cleaned)


def pull_with_dvc(target_path: Path) -> bool:
    """Attempt to pull a DVC-tracked artifact using the Python API."""
    if DVCFileSystem is None:
        logger.warning("dvc is not installed; skipping DVC pull.")
        return False

    try:
        fs = DVCFileSystem()
        remote_path = str(target_path).replace("\\", "/")
        if not fs.exists(remote_path):
            logger.info("DVC path not found: %s", remote_path)
            return False
        fs.get(remote_path, str(target_path), recursive=False)
        logger.info("Pulled %s via DVC.", target_path)
        return True
    except Exception as exc:
        logger.warning("DVC pull failed: %s", exc)
        return False


def _download_hf_file(filename: str, cache_dir: Path) -> Path:
    """Download a single HF dataset file with resume (one artifact, fewer HTTP calls)."""
    logger.info("Downloading %s from HuggingFace...", filename)
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(cache_dir),
        resume_download=True,
    )
    resolved = Path(local_path)
    logger.info("Downloaded %s (%d bytes)", resolved, resolved.stat().st_size)
    return resolved


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: str(col).strip().lower().replace(" ", "_") for col in frame.columns}
    return frame.rename(columns=renamed)


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _load_news_for_tickers(
    news_path: Path,
    tickers: set[str],
    max_rows_per_ticker: int | None = None,
    chunk_size: int = 50_000,
    full_scan: bool = True,
    stale_chunk_limit: int = 15,
    min_tickers_ready_ratio: float = 0.9,
) -> dict[tuple[str, pd.Timestamp], list[str]]:
    """Chunk-read news CSV; ``full_scan=True`` reads the whole file (no early quota stop)."""
    ticker_counts: dict[str, int] = defaultdict(int)
    news_by_day: dict[tuple[str, pd.Timestamp], list[str]] = defaultdict(list)
    chunks_read = 0
    stale_chunks = 0
    min_ready_tickers = max(1, int(len(tickers) * min_tickers_ready_ratio))

    for raw_chunk in pd.read_csv(news_path, chunksize=chunk_size, low_memory=False):
        chunks_read += 1
        chunk_frame = _normalize_columns(raw_chunk)
        ticker_col = _pick_column(chunk_frame, ("stock_symbol", "symbol", "ticker"))
        date_col = _pick_column(chunk_frame, ("date", "published_date", "datetime"))
        text_col = _pick_column(
            chunk_frame,
            ("article_title", "title", "headline", "text", "news"),
        )
        if ticker_col is None or date_col is None:
            msg = f"News CSV missing ticker/date columns. Found: {list(chunk_frame.columns)}"
            raise ValueError(msg)

        chunk_frame["_ticker"] = chunk_frame[ticker_col].map(_normalize_ticker_symbol)
        filtered = chunk_frame[chunk_frame["_ticker"].isin(tickers)].copy()
        if text_col:
            filtered = filtered[filtered[text_col].astype(str).str.strip().ne("")]
        # FNSPID news timestamps are UTC; prices use naive dates — strip tz for merge keys.
        filtered["_day"] = (
            pd.to_datetime(filtered[date_col], errors="coerce", utc=True)
            .dt.tz_localize(None)
            .dt.normalize()
        )
        filtered = filtered.dropna(subset=["_day"])

        rows_added = 0
        batch_frames: list[pd.DataFrame] = []
        for ticker, ticker_frame in filtered.groupby("_ticker", sort=False):
            if max_rows_per_ticker is not None:
                quota = max_rows_per_ticker - ticker_counts[ticker]
                if quota <= 0:
                    continue
                batch_frames.append(ticker_frame.head(quota))
            else:
                batch_frames.append(ticker_frame)

        if batch_frames:
            batch = pd.concat(batch_frames, ignore_index=True)
            text_values = batch[text_col].astype(str) if text_col else pd.Series([""] * len(batch))
            for ticker, day, news_text in zip(
                batch["_ticker"],
                batch["_day"],
                text_values,
                strict=True,
            ):
                news_by_day[(ticker, day)].append(news_text)
                ticker_counts[ticker] += 1
                rows_added += 1

        stale_chunks = stale_chunks + 1 if rows_added == 0 else 0
        if chunks_read % 10 == 0:
            logger.info(
                "News chunks=%d tickers=%d news_rows=%d",
                chunks_read,
                len(ticker_counts),
                sum(ticker_counts.values()),
            )

        if not full_scan and max_rows_per_ticker is not None:
            ready_tickers = sum(
                1 for ticker in tickers if ticker_counts.get(ticker, 0) >= max_rows_per_ticker
            )
            if ready_tickers >= min_ready_tickers:
                logger.info(
                    "Stopping news scan: %d/%d tickers reached row quota at chunk %d",
                    ready_tickers,
                    len(tickers),
                    chunks_read,
                )
                break

            if stale_chunks >= stale_chunk_limit:
                logger.info(
                    "Stopping news scan: no new rows for %d chunks (ready=%d/%d)",
                    stale_chunk_limit,
                    ready_tickers,
                    len(tickers),
                )
                break

    logger.info(
        "News scan finished: chunks=%d tickers_with_news=%d total_headlines=%d",
        chunks_read,
        len(ticker_counts),
        sum(ticker_counts.values()),
    )
    return news_by_day


def _extract_price_frames(
    zip_path: Path,
    tickers: set[str],
    extract_dir: Path,
) -> pd.DataFrame:
    """Read per-ticker CSVs from full_history.zip via archive.open."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    ticker_lookup = {ticker.replace(".", "-").upper(): ticker for ticker in tickers}
    ticker_lookup.update({ticker.upper(): ticker for ticker in tickers})
    frames: list[pd.DataFrame] = []
    extracted = 0

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if "__MACOSX" in member or not member.lower().endswith(".csv"):
                continue
            stem = Path(member).stem.upper().replace(".", "-")
            ticker = ticker_lookup.get(stem)
            if ticker is None:
                continue
            with archive.open(member) as zip_file:
                price_frame = _normalize_columns(pd.read_csv(zip_file))
            date_col = _pick_column(price_frame, ("date", "datetime"))
            if date_col is None:
                continue
            price_frame["ticker"] = ticker
            price_frame["date"] = pd.to_datetime(price_frame[date_col]).dt.normalize()
            for col, out_name in (
                ("open", "open"),
                ("high", "high"),
                ("low", "low"),
                ("close", "close"),
                ("volume", "volume"),
            ):
                if col in price_frame.columns:
                    price_frame[out_name] = pd.to_numeric(price_frame[col], errors="coerce")
            keep = ["ticker", "date", "open", "high", "low", "close", "volume"]
            frames.append(price_frame[[name for name in keep if name in price_frame.columns]])
            extracted += 1

    logger.info("Loaded price history for %d tickers from zip", extracted)
    if not frames:
        msg = "No price CSVs extracted for requested tickers. Check zip layout or ticker list."
        raise ValueError(msg)
    return pd.concat(frames, ignore_index=True)


def _merge_news_and_prices(
    prices: pd.DataFrame,
    news_by_day: dict[tuple[str, pd.Timestamp], list[str]],
) -> pd.DataFrame:
    def _news_for_row(ticker: str, day: pd.Timestamp) -> str:
        day_key = pd.Timestamp(day)
        if day_key.tzinfo is not None:
            day_key = day_key.tz_localize(None)
        day_key = day_key.normalize()
        headlines = news_by_day.get((ticker, day_key), [])
        return "\n".join(headlines) if headlines else "No news available for window."

    prices = prices.copy()
    prices["news"] = [
        _news_for_row(ticker, day)
        for ticker, day in zip(prices["ticker"], prices["date"], strict=True)
    ]
    return prices


def _add_change_pct_and_labels(
    frame: pd.DataFrame,
    label_threshold_pct: float = 0.5,
) -> pd.DataFrame:
    """Add daily change_pct and next-day binary target_label per ticker."""
    if "close" not in frame.columns:
        msg = "Price frame must include a close column for labels."
        raise ValueError(msg)

    sorted_frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    sorted_frame["change_pct"] = (
        sorted_frame.groupby("ticker", sort=False)["close"].pct_change() * 100.0
    )
    next_close = sorted_frame.groupby("ticker", sort=False)["close"].shift(-1)
    forward_return_pct = (next_close - sorted_frame["close"]) / sorted_frame["close"] * 100.0
    sorted_frame["target_label"] = (forward_return_pct > label_threshold_pct).astype("Int64")
    sorted_frame = sorted_frame.dropna(subset=["change_pct"]).copy()
    sorted_frame["target_label"] = sorted_frame["target_label"].fillna(0).astype(int)
    return sorted_frame


def download_fnspid_subset(
    output_path: Path,
    top_tickers: int = 80,
    max_rows_per_ticker: int | None = None,
    label_threshold_pct: float = 0.5,
    full_news_scan: bool = True,
) -> Path:
    """
    Build a manageable FNSPID subset via targeted HF downloads.

    ``max_rows_per_ticker=None`` keeps all headlines found in the news CSV for each ticker.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = output_path.parent / "raw" / "hf"
    extract_dir = output_path.parent / "raw" / "prices"

    available = DEFAULT_TOP_TICKERS[:top_tickers]
    if len(available) < top_tickers:
        logger.warning(
            "Requested %d tickers but only %d defined; using all defined names.",
            top_tickers,
            len(available),
        )
    tickers = set(available)
    logger.info(
        "Building subset for %d tickers (full_news_scan=%s, max_rows_per_ticker=%s)",
        len(tickers),
        full_news_scan,
        max_rows_per_ticker,
    )

    news_path = _download_hf_file(NEWS_FILENAME, raw_dir)
    news_by_day = _load_news_for_tickers(
        news_path,
        tickers,
        max_rows_per_ticker=max_rows_per_ticker,
        full_scan=full_news_scan,
    )

    zip_path = _download_hf_file(PRICE_ZIP_FILENAME, raw_dir)
    prices = _extract_price_frames(zip_path, tickers, extract_dir)
    frame = _merge_news_and_prices(prices, news_by_day)
    frame = _add_change_pct_and_labels(frame, label_threshold_pct=label_threshold_pct)
    placeholder = "No news available for window."
    with_news = int((frame["news"] != placeholder).sum())
    logger.info(
        "Rows=%d tickers=%d rows_with_news=%d (%.1f%%)",
        len(frame),
        frame["ticker"].nunique(),
        with_news,
        100.0 * with_news / max(len(frame), 1),
    )
    frame.to_parquet(output_path, index=False)
    logger.info("Saved %d rows to %s", len(frame), output_path)
    return output_path


def download_data(
    processed_file: Path,
    top_tickers: int = 80,
    force_rebuild: bool = False,
    label_threshold_pct: float = 0.5,
    full_news_scan: bool = True,
    max_rows_per_ticker: int | None = None,
) -> Path:
    """Pull via DVC when possible; otherwise download from HuggingFace."""
    if processed_file.exists() and not force_rebuild:
        logger.info("Processed data already exists: %s", processed_file)
        return processed_file

    if not force_rebuild and pull_with_dvc(processed_file):
        return processed_file

    return download_fnspid_subset(
        processed_file,
        top_tickers=top_tickers,
        max_rows_per_ticker=max_rows_per_ticker,
        label_threshold_pct=label_threshold_pct,
        full_news_scan=full_news_scan,
    )


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build FNSPID subset parquet.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/fnspid_subset_thr03.parquet"),
    )
    parser.add_argument("--top-tickers", type=int, default=80)
    parser.add_argument(
        "--max-rows-per-ticker",
        type=int,
        default=None,
        help="Cap news rows per ticker (default: no cap, full scan).",
    )
    parser.add_argument("--label-threshold-pct", type=float, default=0.3)
    parser.add_argument(
        "--no-full-news-scan",
        action="store_true",
        help="Use legacy early-stop news loading (not recommended).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if parquet already exists.",
    )
    args = parser.parse_args()
    download_data(
        args.output,
        top_tickers=args.top_tickers,
        force_rebuild=args.force,
        label_threshold_pct=args.label_threshold_pct,
        full_news_scan=not args.no_full_news_scan,
        max_rows_per_ticker=args.max_rows_per_ticker,
    )


if __name__ == "__main__":
    main()
