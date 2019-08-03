# Disaster Response Pipeline Project

### A Machine Learning Message classifier trained on messages received during disasters to classify them according to their respective category(e.g. health, medicine, relief, etc.) 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Important Files:
 1. `data/process_data.py`: ETL Pipeline used for data preperation 
 2. `models/train_classifier.py`: Machine Learning Pipeline used to build, train and test model
 3. `app/run.py`: Python file to instantiate web application
 4. `app/templates/*.html`: HTML files for web application