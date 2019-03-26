import requests
import pandas as pd
from sqlalchemy import create_engine
import constants
import datetime
import getPredictionData
#this file contains all the operations called upon by the API

#connects to the database
db = constants.DATABASES['production']
engine_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
    user =      db['USER'],
    password =  db['PASSWORD'],
    host =      db['HOST'],
    port =      db['PORT'],
    database =  db['NAME']
)

#function called by flask to get all the historic data 
def getAllHistorical(curr_Pair):
    engine = create_engine(engine_string)
    data = pd.read_sql_table(curr_Pair, engine) #TODO change table name to a var
    df = pd.DataFrame
    df = data[['date']].copy()
    df['date'] = pd.to_datetime(df['date'],unit='s')
    df['open'] =  data[['bidopen', 'askopen']].mean(axis=1)
    df['close'] = data[['bidclose', 'askclose']].mean(axis=1)
    df['high'] = data[['bidhigh', 'askhigh']].mean(axis=1)
    df['low'] = data[['bidlow', 'asklow']].mean(axis=1)
    
    return df.to_json(orient='records')

#function called by flask to get the prediction for the next hour.
def getCurrData(curr_Pair):
    return getPredictionData.prepareData(curr_Pair)

def getNowDateForCurr(curr_Pair):
    engine = create_engine(engine_string)
    data = pd.read_sql_table(curr_Pair, engine) #TODO change table name to a var
    df = pd.DataFrame
    df = data[['date']].copy()
    df['date'] = pd.to_datetime(df['date'],unit='s')
    df = df[-1:]
    return df.to_json(orient='records')