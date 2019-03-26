import requests
import fxcmpy
import pandas
from sqlalchemy import create_engine
import io
import tokens
import constants

#This file is responsible for pulling new data into the database
print("Connecting to API..")
con = fxcmpy.fxcmpy(access_token=tokens.FXCM_API_KEY, log_level='error', server='demo')
print("Connection Status: " + con.connection_status)
data = pandas.DataFrame()
db = constants.DATABASES['production']
print("Writing data to tables...")
for key, value in constants.TRADED_PAIRS.items():
    data = con.get_candles(key, period='H1', number=1000)

    engine_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
        user =      db['USER'],
        password =  db['PASSWORD'],
        host =      db['HOST'],
        port =      db['PORT'],
        database =  db['NAME']
    )

    engine = create_engine(engine_string)
    data.head(0).to_sql(value, engine, if_exists='replace',index=True) #truncates the table, can also use append: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
    conn = engine.raw_connection()
    curnew = conn.cursor()
    output = io.StringIO()
    data.to_csv(output, sep='\t', header=False, index=True)
    output.seek(0)
    contents = output.getvalue()
    curnew.copy_from(output, value, null="") # null values become ''
    conn.commit() 
    print("Write Complete for: " + key)

conn.close() #engine connect
con.close() #API FOREX