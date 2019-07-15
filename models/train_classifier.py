from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
import sys
import nltk
import pickle
import pandas as pd
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """Load data into dataframe from database.

    Args:
    database_filepath: str. The path of the database file.

    Returns:
    X: feature data values
    y: class variables
    category_names: list of feature coloumns
    """

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    table_name = 'disaster_relief'
    df = pd.read_sql_table(table_name, engine)
    category_names = df.columns.tolist()
    # select only messeges recieved directly for greater accuracy
    df = df[df['genre'] == 'direct']
    X = df.message.values
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    return X, y, category_names


def tokenize(text):
    """Clean and tokenize data recieved.

    Args:
    text: str. Corpus of raw data.

    Returns:
    clean_tokens: list of tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    # Clean tokens and add to list
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """Create machine learning model using count vectorizer and tf-idf transfomer. Use Gridseach to
    find the best parameters.

    Returns:
    model: GridSearchCV
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [20, 50, 100]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Clean and tokenize data recieved.

    Args:
    text: str. Corpus of raw data.

    Returns:s
    clean_tokens: list of tokens
    """
    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print('---------------------------------------------------------------------------')
        print("Category:", category_names[i])
        print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print("Accuracy: ", accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i]))
    print("Best Parameters:", model.best_params_)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()