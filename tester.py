import torch
import numpy as np
import pandas as pd
import json
import os
from FinanceSentimentAnalyzer import predict_headline_sentiment
from datetime import datetime

def write_sentiment_files(news_file_path, output_folder):
    news_file = open(news_file_path,)
    file = json.load(news_file)
    fields = ['formatted_date', 'articles', 'sentiment']

    for ticker in file:
        try:
            ts = int(ticker['news'][0]['datetime'])
            
            date = datetime.fromtimestamp(ts)

            all_data = []
            name = ticker['head'].replace('/', '-')
            total_sentiment = 0
            articles = 0
            for news in ticker['news']:
                try:
                    check_date = datetime.fromtimestamp(int(news['datetime']))

                    if check_date.strftime('%Y-%m-%d') == date.strftime('%Y-%m-%d'):
                        total_sentiment += round(predict_headline_sentiment(news['headline']))*2-1
                        articles += 1
                    else:
                        all_data.append([date.strftime('%Y-%m-%d'), articles, total_sentiment])
                        date = check_date
                        total_sentiment = round(predict_headline_sentiment(news['headline']))*2-1
                        articles = 1
                except:
                    continue
            
            all_data.append([date.strftime('%Y-%m-%d'), articles, total_sentiment])
            all_data.reverse()
            df = pd.DataFrame(all_data, columns=fields)
            all_data = []
            df.to_csv(f'{output_folder}\\{name}.csv', index=False)
        except:
            continue


def prepare_data(stocks_folder, sentiments_folder):
    # Initialize an empty list to store the combined data
    data = []

    # Get the list of files in both folders
    stock_files = os.listdir(stocks_folder)
    sentiment_files = os.listdir(sentiments_folder)

    # Find the common files between the two folders
    common_files = set(stock_files).intersection(set(sentiment_files))

    for file_name in common_files:
        # Load the stock data
        stock_file_path = os.path.join(stocks_folder, file_name)
        stock_data = pd.read_csv(stock_file_path)
        stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
        stock_data['formatted_date'] = stock_data['Date'].dt.strftime('%Y-%m-%d')
        stock_data.drop('Date', axis=1, inplace=True)

        # Load the sentiment data
        sentiment_file_path = os.path.join(sentiments_folder, file_name)
        sentiment_data = pd.read_csv(sentiment_file_path)

        # Combine the stock and sentiment data (matching by date if necessary)
        combined_df = stock_data.merge(sentiment_data, on='formatted_date', how='left')

        # Append the combined dataframe to the list
        data.append(combined_df)

    # Concatenate all dataframes in the list to create a single dataframe
    final_df = pd.concat(data, ignore_index=True)
    
    return final_df

dir_path = os.path.dirname(os.path.realpath(__file__))

news_file = f'{dir_path}\\raw_data\\company_news.json'
sentiment_foler = f'{dir_path}\\raw_data\\sentiment_data'

# write_sentiment_files(news_file, sentiment_foler)

df = prepare_data(f'{dir_path}\\raw_data\\stocks', sentiment_foler)

print(df)