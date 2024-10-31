from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('data/book_summaries_test.csv')
train, test = train_test_split(df, test_size=0.1, random_state=42)
print(len(test))

test.to_csv('data_small/book_summaries_test.csv', index=False)