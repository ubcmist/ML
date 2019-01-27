# Data Extraction uisng Fitbit Developer API

This folder contains the source code for fitbit api and scripts to use them. We developed the scripts for easier access to fitbit data. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites


```
python 3.5
Pandas
fitbit api library
and of course, a fitbit tracking device
```

### Installing

Install the required libraries for fitbit api by following the instructions in:
https://towardsdatascience.com/collect-your-own-fitbit-data-with-python-ff145fa10873.

However, you can use the steps bellow if you would like to use anaconda:

1. Clone our ML repository in your computer. The required libraries are in  the cloned repository.
2. Open anaconda prompt and activate the python 3.5 environment if not used by default.
3. Cd to the folder ML\FITBIT_Data_Collection_Interface\
4. Following step 2 of the link above, install the required libraries which are in this folder using the code below:
```
pip install -r requirements/base.txt’ .
```
Note:
If you get to error that cherrypy is not installed, install it using
```
conda install -c anaconda cherrypy 
```

### Running the script

This script uses the arguments from the terminal to configure the data extraction method.

The arguments are:
```
'-a', '--address', type=str, default="defaultAddress", help='location to save .csv file'
'-d', '--date', type=str, default=yesterday, help='date of required data in format of %Y%M%D'
'-t', '--type', type=str, default="heart", help='heart or sleep data'
'-u', '--username', type=str, default="mist", help='Username associated with the DevFitbit SDK account and fitbit tracker in use.'
```

So an example of running the code would be:
```
python DataExtraction_Final.py -a Data/Heart/Heart_Rate_Data -d 20190125 -t heart -u mist
```
or just 
```
python DataExtraction_Final.py -a Data/Heart/Heart_Rate_Data -d 20190125
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* thanks to orcasgit for their python-fitbit codes
