def preprocessing(df_tmp):
    df = df_tmp.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    
    df = df[
        (df['Host_Popularity_percentage'] >= 0) & (df['Host_Popularity_percentage'] <= 100) &
        (df['Guest_Popularity_percentage'] >= 0) & (df['Guest_Popularity_percentage'] <= 100) &
        (df['Number_of_Ads'].isin([0, 1, 2, 3]))
    ]
    
    df['Episode_Number'] = df['Episode_Title'].str.extract(r'Episode (\d+)', expand=False).fillna(0)
    df['Episode_Number'] = pd.to_numeric(df['Episode_Number'], errors='coerce').astype(int)

    df['Guest_Popularity_missing'] = df['Guest_Popularity_percentage'].isna().astype(int)
    df['Guest_Popularity_percentage'].fillna(0, inplace=True)

    df["Number_of_Ads"].fillna(1, inplace=True)
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    df['Episode_Title'] = df['Episode_Title'].astype('category')
    df['Episode_Length_minutes'] = pd.to_numeric(df['Episode_Length_minutes'], errors='coerce')
    df['Genre'] = df['Genre'].astype('category')
    df['Host_Popularity_percentage'] = pd.to_numeric(df['Host_Popularity_percentage'], errors='coerce')
    df['Publication_Day'] = df['Publication_Day'].astype('category')
    df['Publication_Time'] = df['Publication_Time'].astype('category')
    df['Guest_Popularity_percentage'] = pd.to_numeric(df['Guest_Popularity_percentage'], errors='coerce')
    df['Number_of_Ads'] = df['Number_of_Ads'].astype(int)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('category')


    day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                                'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['Publication_Day_Numeric'] = df['Publication_Day'].map(day_order).fillna(0)

    time_order = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    df['Publication_Time_Numeric'] = df['Publication_Time'].map(time_order).fillna(0)

    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['Episode_Sentiment_Numeric'] = df['Episode_Sentiment'].map(sentiment_map).fillna(0)

    genre_map = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 
                        'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    df['Genre_Numeric'] = df['Genre'].map(genre_map).fillna(0)

    df['Ads_per_minute'] = df['Number_of_Ads'] / (1 + df['Episode_Length_minutes'])
    df['Has_Ads'] = (df['Number_of_Ads'] > 0).astype(int)

    df['Is_Weekend'] = df['Publication_Day_Numeric'].isin([5, 6]).astype(int)

    df['Episode_Length_lv_1'] = ((df['Episode_Length_minutes'] > 30) & (df['Episode_Length_minutes'] <= 60)).astype(int)
    df['Episode_Length_lv_2'] = ((df['Episode_Length_minutes'] > 60) & (df['Episode_Length_minutes'] <= 90)).astype(int)
    df['Episode_Length_lv_3'] = (df['Episode_Length_minutes'] > 90).astype(int)

    df['Host_Popularity_lv_1'] = ((df['Host_Popularity_percentage'] > 70) & 
                                    (df['Host_Popularity_percentage'] <= 90)).astype(int)
    df['Host_Popularity_lv_2'] = (df['Host_Popularity_percentage'] > 90).astype(int)

    df['Guest_Popularity_lv_1'] = ((df['Guest_Popularity_percentage'] > 70) & 
                                        (df['Guest_Popularity_percentage'] <= 90)).astype(int)
    df['Guest_Popularity_lv_1'] = (df['Guest_Popularity_percentage'] > 90).astype(int)

    df["Guest_Host_Combined_Popularity"] = (df["Guest_Popularity_percentage"] + df["Host_Popularity_percentage"]) / 2

    df = df.drop(columns=["Publication_Day","Publication_Time","Episode_Sentiment","Genre", "Episode_Title"])
    return df

