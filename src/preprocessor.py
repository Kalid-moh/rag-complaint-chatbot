import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
import zipfile
from pathlib import Path
import warnings

from src.settings import (
    RAW_CSV, ZIP_URL, ZIP_PATH, FILTERED_CSV, WORD_COUNT_PLOT,
    RAW_DIR, PROCESSED_DIR, NOTEBOOKS_DIR,
    RELEVANT_CFPB_PRODUCTS, PRODUCT_MAPPING
)

warnings.filterwarnings("ignore", category=FutureWarning)


class CFPBDataProcessor:
    def __init__(self) -> None:
        self.filtered_df: pd.DataFrame | None = None
        self.total_complaints: int = 0
        self.total_with_narrative: int = 0
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for d in [RAW_DIR, PROCESSED_DIR, NOTEBOOKS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def download_dataset(self):
        if RAW_CSV.exists():
            print(f"Raw dataset already exists: {RAW_CSV}")
            return self

        print("Downloading latest CFPB complaints dataset...")
        try:
            response = requests.get(ZIP_URL, stream=True, timeout=60)
            response.raise_for_status()

            with open(ZIP_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            with zipfile.ZipFile(ZIP_PATH, "r") as z:
                z.extractall(RAW_DIR)

            print(f"Dataset downloaded and extracted to {RAW_CSV}")
            return self
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}")

    def load_data(self):
                if self.filtered_df is not None:
                     print("Filtered data already loaded.")
                     return self

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found. Run download_dataset() first.")

    print("Starting memory-safe chunked loading...")

    required_cols = [
        'Product',
        'Consumer complaint narrative',
        'Complaint ID',
        'Date received',
        'Issue'
    ]

    chunksize = 100_000
    filtered_chunks = []

    self.total_complaints = 0
    self.total_with_narrative = 0

    # ✅ Indent the try inside the method
    try:
        for i, chunk in enumerate(pd.read_csv(
            RAW_CSV,
            usecols=required_cols,
            chunksize=chunksize,
            dtype=str,
            engine='python',        # safer parser
            on_bad_lines='skip',    # skip malformed lines
            low_memory=True          # reduces memory footprint
        )):
            self.total_complaints += len(chunk)

            # Check which rows have narratives
            narrative = chunk['Consumer complaint narrative']
            has_narrative = narrative.notna() & (narrative.str.strip() != '')
            self.total_with_narrative += has_narrative.sum()

            # Filter only relevant products and rows with narrative
            mask = chunk['Product'].isin(RELEVANT_CFPB_PRODUCTS) & has_narrative
            filtered = chunk[mask].copy()

            if not filtered.empty:
                filtered_chunks.append(filtered)

            print(f"Chunk {i+1} processed | Filtered so far: {sum(len(c) for c in filtered_chunks):,}")

        # Combine all filtered chunks at the end
        self.filtered_df = pd.concat(filtered_chunks, ignore_index=True) if filtered_chunks else pd.DataFrame()
        print(f"\nLoaded & filtered {len(self.filtered_df):,} relevant complaints")
        return self

    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")



    def perform_eda(self):
        if self.filtered_df is None or self.filtered_df.empty:
            raise ValueError("Run load_data() first.")

        print("\n=== EDA ===")
        print(f"Total filtered complaints: {len(self.filtered_df):,}")

        print("\nProduct distribution:")
        print(self.filtered_df['Product'].value_counts())

        self.filtered_df['word_count'] = self.filtered_df['Consumer complaint narrative'].str.split().str.len()

        print("\nWord count stats:")
        print(self.filtered_df['word_count'].describe())

        plt.figure(figsize=(12, 6))
        sns.histplot(self.filtered_df['word_count'], bins=100, kde=True, color="teal")
        plt.title("Narrative Word Count Distribution")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.xlim(0, 1000)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(WORD_COUNT_PLOT)
        plt.close()
        print(f"Plot saved: {WORD_COUNT_PLOT}")

        return self

    def filter_and_clean(self):
        if self.filtered_df is None or self.filtered_df.empty:
            raise ValueError("Run load_data() first.")

        print("\nMapping categories...")
        self.filtered_df['product_category'] = self.filtered_df['Product'].map(PRODUCT_MAPPING)

        print("\nFinal distribution:")
        print(self.filtered_df['product_category'].value_counts())

        def clean_text(text):
            if pd.isna(text) or not str(text).strip():
                return ""
            text = str(text).lower()
            text = re.sub(r'x{4,}', " ", text)
            text = re.sub(r'[^a-z0-9\s]', " ", text)
            text = re.sub(r"\bi am writing to file a complaint\b", " ", text, flags=re.I)
            text = re.sub(r"\bdear cfpb\b", " ", text, flags=re.I)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        print("Cleaning narratives...")
        self.filtered_df['clean_narrative'] = self.filtered_df['Consumer complaint narrative'].apply(clean_text)

        return self

    def save(self):
        if self.filtered_df is None or self.filtered_df.empty:
            raise ValueError("No data to save.")

        self.filtered_df.to_csv(FILTERED_CSV, index=False)
        print(f"\nSaved to {FILTERED_CSV}")
        print(f"Final records: {len(self.filtered_df):,}")

        return self

    def run_full_pipeline(self):
        self.download_dataset().load_data().perform_eda().filter_and_clean().save()
        print("\nTask 1 completed!")


if __name__ == "__main__":
    processor = CFPBDataProcessor()
    processor.run_full_pipeline()