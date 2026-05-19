Kaggle Data

Kaggle 2008–2024, daily headlines, perfectly covers your full test window:sp500_headlines_2008_2024.csv
URL:  kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024

 hkapoor/indian-financial-news-articles-20032020 (Kaggle)
2003–2020, full-text Indian financial news (Economic Times). Covers most of your test window: nifty_news_2003_2020.csv

# 1. Download both datasets from Kaggle, unzip, place CSVs in:
#    Rename them to:
#      sp500_headlines_2008_2024.csv
#      nifty_news_2003_2020.csv

Alternative Data from Higggingface Json(Not currently used)
https://huggingface.co/datasets/Brianferrell787/financial-news-multisource

#  Combine + normalize (auto-detects both known filenames)
python -m data.normalize_corpus
python -m data.normalize_corpus --input raw_news.csv --output articles.csv
python -m data.normalize_corpus --input news.jsonl --output articles.csv
# 1. Manually download dataset from Kaggle, unzip, find the CSV
# 2. Normalize it
python -m data.normalize_corpus --input "C:\path\to\downloaded.csv" --output articles.csv
# 3. Run sentiment pipeline (now with years of real data)
python -m data.fetch_sentiment --input articles.csv --output data\cache\
# 4. Run main
python main.py --market NIFTY50