import fitbit
import gather_keys_oauth2 as Oauth2
import pandas as pd
import datetime
CLIENT_ID ='22DF24'
CLIENT_SECRET = '7848281e9151008de32698f7dd304c68'
# Note: This ID belong's to the fitbit under 'UBC_MIST' (Not Hooman's)

server = Oauth2.OAuth2Server(CLIENT_ID, CLIENT_SECRET)
server.browser_authorize()
ACCESS_TOKEN = str(server.fitbit.client.session.token['access_token'])
REFRESH_TOKEN = str(server.fitbit.client.session.token['refresh_token'])
auth2_client = fitbit.Fitbit(CLIENT_ID, CLIENT_SECRET, oauth2=True, access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)
#access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)

yesterday = str((datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d"))   # To avoid updating dates everyday
yesterday2 = str((datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
today = str(datetime.datetime.now().strftime("%Y%m%d"))


fit_statsHR = auth2_client.intraday_time_series('activities/heart', base_date=yesterday2, detail_level='1sec') #collects data


#put it in a readable format using Panadas
time_list = []
val_list = []
for i in fit_statsHR['activities-heart-intraday']['dataset']:
	val_list.append(i['value'])
	time_list.append(i['time'])
heartdf = pd.DataFrame({'Heart Rate':val_list,'Time':time_list})

#tries to save the data locally
heartdf.to_csv('Data/Sample_Data'+ \
yesterday+'.csv', \
columns=['Time','Heart Rate'], header=True, \
index = False)


#More data to be extracted such as sleep log data
fit_statsSl = auth2_client.sleep(date='today')
stime_list = []
sval_list = []
for i in fit_statsSl['sleep'][0]['minuteData']:
	stime_list.append(i['dateTime'])
	sval_list.append(i['value'])
sleepdf = pd.DataFrame({'State':sval_list,
'Time':stime_list})
sleepdf['Interpreted'] = sleepdf['State'].map({'2':'Awake','3':'Very Awake','1':'Asleep'})
sleepdf.to_csv('/Users/shsu/Downloads/python-fitbit-master/Sleep/sleep' + \
today+'.csv', \
columns = ['Time','State','Interpreted'],header=True,
index = False)

#also Sleep Summary
fit_statsSum = auth2_client.sleep(date='today')['sleep'][0]
ssummarydf = pd.DataFrame({'Date':fit_statsSum['dateOfSleep'],
'MainSleep':fit_statsSum['isMainSleep'],
'Efficiency':fit_statsSum['efficiency'],
'Duration':fit_statsSum['duration'],
'Minutes Asleep':fit_statsSum['minutesAsleep'],
'Minutes Awake':fit_statsSum['minutesAwake'],
'Awakenings':fit_statsSum['awakeCount'],
'Restless Count':fit_statsSum['restlessCount'],
'Restless Duration':fit_statsSum['restlessDuration'],
'Time in Bed':fit_statsSum['timeInBed']
} ,index=[0])