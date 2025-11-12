#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steam Reviews Analysis
Requires:
    pip install pandas matplotlib seaborn textblob wordcloud tqdm
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from tqdm import tqdm

# ============ 설정 ============
INPUT_FILE = "steam_reviews.jsonl"   # 크롤링 결과 파일명
OUTPUT_CSV = "steam_reviews_analyzed.csv"

# ------------------------------
def load_reviews(file_path):
    """JSONL 형식의 리뷰 파일을 DataFrame으로 불러옵니다."""
    reviews = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading reviews"):
            try:
                data = json.loads(line)
                reviews.append(data)
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(reviews)
    print(f"✅ 총 {len(df):,}개의 리뷰 로드 완료")
    return df

def basic_stats(df):
    """기초 통계"""
    print("\n[기본 정보]")
    print(df[["_app_id", "voted_up", "timestamp_created"]].describe(include="all"))

    total = len(df)
    pos = df["voted_up"].sum()
    neg = total - pos
    print(f"👍 추천 리뷰: {pos:,} ({pos/total:.1%})")
    print(f"👎 비추천 리뷰: {neg:,} ({neg/total:.1%})")

def sentiment_analysis(df):
    """TextBlob을 이용한 간단한 감성 점수 계산"""
    sentiments = []
    for text in tqdm(df["review"].fillna(""), desc="Sentiment analysis"):
        blob = TextBlob(text)
        sentiments.append(blob.sentiment.polarity)
    df["sentiment"] = sentiments
    return df

def plot_sentiment_distribution(df):
    """감성 분포 시각화"""
    plt.figure(figsize=(8,4))
    sns.histplot(df["sentiment"], bins=30, kde=True, color="skyblue")
    plt.title("Sentiment Polarity Distribution")
    plt.xlabel("Polarity (-1 = Negative, +1 = Positive)")
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png")
    plt.show()

def plot_daily_reviews(df):
    """날짜별 리뷰 수 시각화"""
    df["date"] = pd.to_datetime(df["timestamp_created"], unit="s")
    daily = df.groupby(df["date"].dt.date).size()
    plt.figure(figsize=(10,4))
    daily.plot(kind="line", color="green")
    plt.title("Reviews Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig("reviews_over_time.png")
    plt.show()

def generate_wordcloud(df):
    """워드클라우드 생성 (영문 기준, 한국어는 konlpy 형태소 분석기 사용 가능)"""
    text = " ".join(df["review"].fillna("").tolist())
    wc = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        max_words=200,
        colormap="viridis"
    ).generate(text)
    plt.figure(figsize=(10,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Steam Reviews")
    plt.tight_layout()
    plt.savefig("wordcloud.png")
    plt.show()

def save_to_csv(df):
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"💾 분석 결과가 {OUTPUT_CSV} 에 저장되었습니다.")

def main():
    df = load_reviews(INPUT_FILE)

    # 기본 통계
    basic_stats(df)

    # 감성 분석
    df = sentiment_analysis(df)

    # 시각화
    plot_sentiment_distribution(df)
    plot_daily_reviews(df)
    generate_wordcloud(df)

    # CSV 저장
    save_to_csv(df)

if __name__ == "__main__":
    main()
