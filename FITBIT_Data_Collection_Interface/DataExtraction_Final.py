import argparse
import fitbit
import gather_keys_oauth2 as Oauth2
import pandas as pd
import datetime
import os.path as osp
import os

# ************ Hooman's Account *********
CLIENT_ID ='22D68G'
CLIENT_SECRET = '32e28a7e72842298fd5d97ce123104ca'

# ************ UBC MIST Account *********
# CLIENT_ID ='22DF24'
# CLIENT_SECRET = '7848281e9151008de32698f7dd304c68'

#parsing arguements passed via CMD line
parser = argparse.ArgumentParser(
    description='Test code to learn how scripts with args work',
    epilog="For more information call me baby")

parser.add_argument('-d', '--date', type=str, default="20190101", help='date of required data')
parser.add_argument('-t', '--type', type=str, default="heart", help='heart or sleep data')
parser.add_argument('-a', '--address', type=str, default="/", help='location to save .csv file')
args = vars(parser.parse_args())

#creating .csv file for requested heart rate data 
if args['type'] == 'heart':
    heart_rate_data_csv_address = args['address']+ args['date'] +'.csv'
    if not osp.exists(osp.dirname(heart_rate_data_csv_address)):
        os.makedirs(osp.dirname(heart_rate_data_csv_address))
    if not osp.isfile(heart_rate_data_csv_address):
        required_date = args['date'][0:4]+'-'+args['date'][4:6]+'-'+args['date'][6:8];
        fit_statsHR = auth2_client.intraday_time_series('activities/heart', base_date=required_date, detail_level='1sec') #collects data

        #put it in a readable format using Panadas
        time_list = []
        val_list = []
        for i in fit_statsHR['activities-heart-intraday']['dataset']:
            val_list.append(i['value'])
            time_list.append(i['time'])
        heartdf = pd.DataFrame({'Heart Rate':val_list,'Time':time_list})

        # saving the data locally
        heartdf.to_csv(heart_rate_data_csv_address, columns=['Time','Heart Rate'], header=True, index = False)


#creating .csv file for requested sleep rate data        
elif args['type'] == 'sleep':
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
