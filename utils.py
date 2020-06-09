import pandas as pd
import xml.etree.ElementTree as et

columns = ['id', 'summary', 'rating', 'text', 'category']

def rating_to_sentiment(x):
    return 'positive' if x > 3 else 'negative'


def rating_to_label(x):
    return 1 if x > 3 else 0

def xml_to_dataframe(xml_file): 
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    rows = []
    id=1
    for node in xroot: 
        res = []
        res.append(id)
        id+=1
        for el in columns[1:]: 
            if node is not None and node.find(el) is not None:
                res.append(node.find(el).text.strip())
            else: 
                res.append(None)
        rows.append({columns[i]: res[i] 
                     for i, _ in enumerate(columns)})
    
    out_df = pd.DataFrame(rows, columns=columns)
    out_df['rating'] = out_df.rating.astype(float)
    out_df['sentiment'] = out_df['rating'].apply(rating_to_sentiment)
    out_df['language'] = xml_file.split('/')[3]
    return out_df


file_paths = [r'./data/amazon-dataset/english/books/train.review',
         r'./data/amazon-dataset/english/books/test.review',
         r'./data/amazon-dataset/french/books/train.review',
         r'./data/amazon-dataset/french/books/test.review',
         r'./data/amazon-dataset/german/books/train.review',
         r'./data/amazon-dataset/german/books/test.review']

def get_data():
    df = None
    for file_path in file_paths:
        if df is None:
            df = xml_to_dataframe(file_path)
        else:
            df_new = xml_to_dataframe(file_path)
            df = df.append(df_new)

    return df


def preprocess_labels(df):
    return df['rating'].apply(rating_to_label)
