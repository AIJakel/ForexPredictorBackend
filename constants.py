#this file contains a list of constants
TRADED_PAIRS = {
    "USD/JPY":'usd_jpy',
    "EUR/USD":'eur_usd',
    "GBP/USD":'gbp_usd',
    "AUD/USD":'aud_usd',
    "USD/CAD":'usd_cad',
    "USD/CHF":'usd_chf',
    "NZD/USD":'nzd_usd'
}
#connect to DB local
DATABASES = {
    'local':{
        'NAME': 'postgres',
        'USER': 'postgres',
        'PASSWORD':'password',
        'HOST':'localhost',
        'PORT':5432
    },
    'production':{
        'NAME': 'dfrc7rmuva7l3r',
        'USER': 'pinuludhlgaezr',
        'PASSWORD':'1d42f62958829a1aa743d39ad84b4c95f54d16c6d11014c2e971b231b5dfa18d',
        'HOST':'ec2-174-129-10-235.compute-1.amazonaws.com',
        'PORT':5432
    }
}