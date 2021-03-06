import argparse
import fitbit
import gather_keys_oauth2 as Oauth2
import pandas as pd
import datetime
import os.path as osp
import os

# some date variables
yesterday = str((datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d"))   # To avoid updating dates everyday
today = str(datetime.datetime.now().strftime("%Y%m%d"))

#region: parsing arguements passed via CMD line
parser = argparse.ArgumentParser(
    description='Test code to learn how scripts with args work',
    epilog="For more information call me baby")

parser.add_argument('-a', '--address', type=str, default="defaultAddress", help='location to save .csv file')
parser.add_argument('-d', '--date', type=str, default=yesterday, help='date of required data')
parser.add_argument('-t', '--type', type=str, default="heart", help='heart or sleep data')
parser.add_argument('-u', '--username', type=str, default="hooman", help='Username associated with the DevFitbit SDK account'
                                                                       'and fitbit tracker in use.')
args = vars(parser.parse_args())
#endregion: parsing arguements passed via CMD line

# region: Parse date argument. can accept range of dates separated by -, and single dates separated by comma
date_string_raw = args['date']
dates_comma_separated_list = date_string_raw.split(",")# separate dates by comma to get single dates.
# Range of dates separated with "-" can still exist in elements of the resulting list

dates_list_final = []
for dates in dates_comma_separated_list:
    if '-' in dates:
    # finding elements which are range of dates
        date_start, date_end = dates.split("-")
        date_list_int = list(range(int(date_start), int(date_end) + 1))
        for single_date in date_list_int: # adding all the dates in the date range to the final list
            dates_list_final += [str(single_date)]
    else:
    # element in the list is just a single day
        dates_list_final += [dates]
# endregion: Parse date and range of dates

# region: Sanity check for date! can't obtain data older than 21 days
# TODO check to see if this is necessary!
# if int(today)-int(args['date']) > 21:
#     raise("Error: Requested data belongs to date({}).".format(args['date']) +
#           " Cannot download data older than 21 days. Please try again.")
# endregion: Sanity check for date! can't obtain data older than 21 days

# region: API Credentials & FitbitSpecific FolderName
# ************ UBC MIST Account *********
if args['username'] == "mist":
    CLIENT_ID ='22DF24'
    CLIENT_SECRET = '7848281e9151008de32698f7dd304c68'
    fitbit_specific_folder_name = 'Fitbit_m'
# ************ Hooman's Account *********
else:
    CLIENT_ID = '22D68G'
    CLIENT_SECRET = '32e28a7e72842298fd5d97ce123104ca'
    fitbit_specific_folder_name = 'Fitbit_h'
# endregion: API Credentials

# region: obtaining Access-token and Refresh-token
"""for obtaining Access-token and Refresh-token"""
server = Oauth2.OAuth2Server(CLIENT_ID, CLIENT_SECRET)
server.browser_authorize()
ACCESS_TOKEN = str(server.fitbit.client.session.token['access_token'])
REFRESH_TOKEN = str(server.fitbit.client.session.token['refresh_token'])
"""Authorization"""
auth2_client = fitbit.Fitbit(CLIENT_ID, CLIENT_SECRET, oauth2=True, access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)
# endregion: obtaining Access-token and Refresh-token

#region: Creating .csv file for requested heart rate data
if args['type'] == 'heart':
    # region: getting the base address and creating the folder
    if args['address'] == 'defaultAddress':
        base_address = 'Data/Heart/Heart_Rate_Data/'
    else:
        base_address = args['address']
    base_address = osp.join(base_address, fitbit_specific_folder_name)
    # creating the folder if it doesn't exist
    if not osp.exists(base_address):
        os.makedirs(base_address)
    #endregion

    for date in dates_list_final:
        heart_rate_data_csv_address = osp.join(base_address, date + '_x.csv')
        if not osp.isfile(heart_rate_data_csv_address): # only download the data if the file doesn't exist
            required_date = date[0:4]+'-'+date[4:6]+'-'+date[6:8] # required format : "%Y-%m-%d"
            fit_statsHR = auth2_client.intraday_time_series('activities/heart', base_date=required_date, detail_level='1sec') #collects data

            #region: Put data in a readable format using Panadas. make a dataframe
            time_list = []
            val_list = []
            for i in fit_statsHR['activities-heart-intraday']['dataset']:
                val_list.append(i['value'])
                time_list.append(i['time'])
            heartdf = pd.DataFrame({'Time':time_list, 'Heart Rate':val_list})
            #endregion: Put data in a readable format using Panadas. make a dataframe
            # saving the data locally
            heartdf.to_csv(heart_rate_data_csv_address, columns=['Time','Heart Rate'], header=True, index = False)

            #region: modify timestamp to include full date
            def date_time_parser(x):
                return pd.datetime.strptime(date[:4]+'-'+date[4:6]+'-'+ date[6:]+ '-' +x, '%Y-%m-%d-%H:%M:%S')
            heart_series = pd.read_csv(heart_rate_data_csv_address, header=0, parse_dates=[0], index_col=0,
                                                    squeeze=True,  date_parser=date_time_parser)
            heart_series.to_csv(heart_rate_data_csv_address, header =True, date_format='%Y-%m-%d-%H:%M:%S')
            #endregion: modify timestamp to include full date

#creating .csv file for requested sleep rate data        
elif args['type'] == 'sleep':
    raise("ERROR: sleep data extraction not yet implemented.")
    sleep_data_csv_address = 'Data/Sleep/Sleep_Data/' + args['date'] + '.csv'
    if not osp.exists(osp.dirname(sleep_data_csv_address)):
        os.makedirs(osp.dirname(sleep_data_csv_address))
    if not osp.isfile(sleep_data_csv_address):
        # TODO Fix this part. can't get the sleep data and gets zero in return
        fit_statsSl = auth2_client.sleep(date=args['date'])
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

#endregion: Creating .csv file for requested heart rate data