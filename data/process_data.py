import sys
import re
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load data into dataframe from csv files and merge them.

    Args:
    messages_filepath: str. The path of the messages csv file
    categories_filepath: str. The path of the categories csv file

    Returns:
    df: dataframe
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """Clean category data and expand them into independent columns.

    Args:
    df: Dataframe consisting of category column

    Returns:
    df: Cleaned dataframe with independent category columns
    """
    category_vals = df.categories
    new_vals = []
    # First clean the values and append to series.
    for vals in category_vals:
        vals = re.sub("[a-zA-Z-_]", "", vals)
        new_vals.append(vals)
    # Create new dataframe with category values
    new_vals = pd.Series(new_vals)
    new_vals = new_vals.str.split(';', expand=True)
    # Extract all category column names. Clean strings by removing values
    category_col = df.categories[0].replace('-1', '').replace('-0', '').split(';')
    # Rename the category column names to match that of original category column values
    new_vals.columns = category_col
    # replace non binary data into binary
    new_vals.related.replace('2', '1', inplace=True)
    df = df.drop(columns='categories')
    # Join the two dataframes together
    df = df.join(new_vals, how='inner')
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """Save dataframe in database format

    Args:
    df: Dataframe to be saved.
    database_filename: str. Filename of database.
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_relief', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')


    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as ' 
              'well as the filepath of the database to save the cleaned data ' 
              'disaster_messages.csv disaster_categories.csv ' 
              'DisasterResponse.db')


if __name__ == '__main__':
    main()