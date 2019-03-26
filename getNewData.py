import datetime
from sqlalchemy import create_engine
import time
import constants
import pandas as pd
import fxcmpy
import tokens
import io

con = fxcmpy.fxcmpy(access_token=tokens.FXCM_API_KEY, log_level='error', server='demo')
db = constants.DATABASES['production']
now = datetime.datetime.now()

#split
for key, value in constants.TRADED_PAIRS.items():
    engine_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
            user =      db['USER'],
            password =  db['PASSWORD'],
            host =      db['HOST'],
            port =      db['PORT'],
            database =  db['NAME']
        )
    engine = create_engine(engine_string)
    data = pd.read_sql_table(value, engine) #TODO change table name to a var
    df = pd.DataFrame
    df = data[['date']].date
    lastDate = df.iloc[-1]
    
    start = lastDate
    
    stop = now
    
    newData = con.get_candles(key, period='H1', start=start, stop=stop)
    newData = newData.iloc[1:]
    newEngine = create_engine(engine_string)
    
    newData.to_sql(value, newEngine, if_exists='append',index=True, index_label='date') #truncates the table, can also use append: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
    conn = newEngine.raw_connection()
    curnew = conn.cursor()
    output = io.StringIO()
    output.seek(0)
    contents = output.getvalue()
    print(contents)
    curnew.copy_from(output, value, null="") # null values become ''
    conn.commit() 

conn.close()
con.close()