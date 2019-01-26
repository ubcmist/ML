import fitbit
import gather_keys_oauth2 as Oauth2
import pandas as pd
import datetime
import os.path as osp
import os

# ************ UBC MIST Account *********
# CLIENT_ID ='22DF24'
# CLIENT_SECRET = '7848281e9151008de32698f7dd304c68'

# ************ Hooman's Account *********
CLIENT_ID ='22D68G'
CLIENT_SECRET = '32e28a7e72842298fd5d97ce123104ca'

"""for obtaining Access-token and Refresh-token"""
server = Oauth2.OAuth2Server(CLIENT_ID, CLIENT_SECRET)
server.browser_authorize()
ACCESS_TOKEN = str(server.fitbit.client.session.token['access_token'])
REFRESH_TOKEN = str(server.fitbit.client.session.token['refresh_token'])
"""Authorization"""
auth2_client = fitbit.Fitbit(CLIENT_ID, CLIENT_SECRET, oauth2=True, access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)

yesterday = str((datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d"))   # To avoid updating dates everyday
yesterday2 = str((datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
today = str(datetime.datetime.now().strftime("%Y%m%d"))

# ****************************************************************
# ************* get heart rate data / for Yesterday ***************
# ****************************************************************
heart_rate_data_csv_address = 'Data/Heart/Heart_Rate_Data/'+ yesterday +'.csv'
if not osp.exists(osp.dirname(heart_rate_data_csv_address)):
	os.makedirs(osp.dirname(heart_rate_data_csv_address))
if not osp.isfile(heart_rate_data_csv_address):
	fit_statsHR = auth2_client.intraday_time_series('activities/heart', base_date=yesterday2, detail_level='1sec') #collects data

	#put it in a readable format using Panadas
	time_list = []
	val_list = []
	for i in fit_statsHR['activities-heart-intraday']['dataset']:
		val_list.append(i['value'])
		time_list.append(i['time'])
	heartdf = pd.DataFrame({'Heart Rate':val_list,'Time':time_list})

	# saving the data locally
	heartdf.to_csv(heart_rate_data_csv_address, columns=['Time','Heart Rate'], header=True, index = False)


# ****************************************************************
# ************* Heart Rate Summary / for Today ***************
# ****************************************************************
heart_rate_summary_csv_address = 'Data/Heart/Heard_Summary/' + today + '.csv'
if not osp.exists(osp.dirname(heart_rate_summary_csv_address)):
	os.makedirs(osp.dirname(heart_rate_summary_csv_address))
if not osp.isfile(heart_rate_summary_csv_address):
	fitbit_stats = auth2_client.intraday_time_series('activities/heart', base_date='today', detail_level='1sec')
	stats = fitbit_stats

	# get heart summary
	# TODO get the summary of calories burned in each heart rate zone and total. include it
	hsummarydf = pd.DataFrame({'Date': stats["activities-heart"][0]['dateTime'],
							   'HR max': stats["activities-heart"][0]['value']['heartRateZones'][0]['max'],
							   'HR min': stats["activities-heart"][0]['value']['heartRateZones'][0]['min']}, index=[0])


	hsummarydf.to_csv(heart_rate_summary_csv_address, header=False, index=False, mode='a')

# ****************************************************************
# ************* Sleep Data / for Today (last night) ***************
# ****************************************************************
sleep_data_csv_address = 'Data/Sleep/Sleep_Data/' + today + '.csv'
if not osp.exists(osp.dirname(sleep_data_csv_address)):
	os.makedirs(osp.dirname(sleep_data_csv_address))
if not osp.isfile(sleep_data_csv_address):
	# TODO Fix this part. can't get the sleep data and gets zero in return
	fit_statsSl = auth2_client.sleep(date='today')
	stime_list = []
	sval_list = []
	# for i in fit_statsSl['sleep'][0]['minuteData']:
	# 	stime_list.append(i['dateTime'])
	# 	sval_list.append(i['value'])
	# sleepdf = pd.DataFrame({'State':sval_list, 'Time':stime_list})
	# sleepdf['Interpreted'] = sleepdf['State'].map({'2':'Awake','3':'Very Awake','1':'Asleep'})
	#
	# sleepdf.to_csv(sleep_data_csv_address,
	# 						columns = ['Time','State','Interpreted'], header=True , index = False)

# ****************************************************************
# ************* Sleep Summary / for Today (last night) ***************
# ****************************************************************
sleep_summary_csv_address = 'Data/Sleep/Sleep_Summary/' + today + '.csv'
if not osp.exists(osp.dirname(sleep_summary_csv_address)):
	os.makedirs(osp.dirname(sleep_summary_csv_address))
# if not osp.isfile(sleep_summary_csv_address):
	# TODO Fix the sleep data collection from the previous TODO and fix this one after
	# fit_statsSum = fit_statsSl['sleep'][0]
	# ssummarydf = pd.DataFrame({'Date':fit_statsSum['dateOfSleep'],
	# 							'MainSleep':fit_statsSum['isMainSleep'],
	# 							'Efficiency':fit_statsSum['efficiency'],
	# 							'Duration':fit_statsSum['duration'],
	# 							'Minutes Asleep':fit_statsSum['minutesAsleep'],
	# 							'Minutes Awake':fit_statsSum['minutesAwake'],
	# 							'Awakenings':fit_statsSum['awakeCount'],
	# 							'Restless Count':fit_statsSum['restlessCount'],
	# 							'Restless Duration':fit_statsSum['restlessDuration'],
	# 							'Time in Bed':fit_statsSum['timeInBed']}
	# 							,index=[0])
	#
	#
	# ssummarydf.to_csv(sleep_summary_csv_address)
	# # ssummarydf.to_csv('Data/Sleep/sleepsummary.csv', header=False, index=False, mode='a')
