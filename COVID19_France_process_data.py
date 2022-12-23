# -----------------------------------------------------------------------------
# import libraries
import sys
import re
import numpy as np
import pandas as pd

import pygal
from pygal_maps_fr import maps
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# requires installing
# https://gtk-win.sourceforge.io/home/index.php/Main/Downloads

# -----------------------------------------------------------------------------
def load_data(filepath):
    '''
    DESCRIPTION
        load content of the file defined by filepath and convert it to
         dataframe
    INPUT
        filepath is the filepath of the file to load
    OUTPUT
        df is the dataframe of the file's content
    '''
    df = pd.read_csv(filepath.replace('\\','/'),
        dtype={"date": str, "dep": str, 'lib_dep': str, 'lib_reg': str})
    # Enforce the type of these data is mandatory at this stage to avoid 
    #  warning 'Low memory' message for further use of the dataframe.
    return df

# -----------------------------------------------------------------------------
def merge_data(df1, df2):
    '''
    DESCRIPTION
        Merge both dataframes messages and categories by their 'id' column
    INPUT
        df1 is the first dataframe that will be completed with the second
        df2 is the second dataframe to complete the first one
    OUTPUT
        df1 is the merge of both dataframes, sorted by the id values
    '''
    df = pd.merge(df1, df2, how='outer', on=['id'])
    df.sort_values(['id'])
    return df

# -----------------------------------------------------------------------------
def format_data(df):
    '''
    DESCRIPTION
        Enforce type of some data in the dataframe
    INPUT
        df is the dataframe to work on
    OUTPUT
        df is the dataframe modified
    '''
    # Convert data some data in the expected string type
    # Basically they are defined as object sometimes because there is several 
    #  types.
    df['reg'] = df['reg'].astype(str)
    df['dep'] = df['dep'].astype(str)  # department number '2A' and '2B'
    df['date'] = df['date'].astype(str)  # format is YYYY-MM-DD
    df['lib_reg'] = df['lib_reg'].astype(str)  # department's name
    df['lib_dep'] = df['lib_dep'].astype(str)  # region's name

    return df

# -----------------------------------------------------------------------------
def clean_data(df):
    '''
    DESCRIPTION
        clean dataframe from empty data .....
    INPUT
        df is the dataframe to clean
    OUTPUT
        df_clean is the cleaned dataframe
    '''

    # Clean the dataframe:

    # Remove useless or undefined columns
    df.drop(columns=['reg_rea', 'reg_incid_rea', 'cv_dose1', 'R', 'lib_dep',
                     'lib_reg'], inplace=True)

    # Remove all rows containing at least one nan (loose 5.9% of the dataset
    #  according to initial analysis of the dataset)
    df.dropna(axis='index', how='any', inplace=True)

    return df

# -----------------------------------------------------------------------------
def replace_invalid_data(df, label):
    '''
    DESCRIPTION
        Replace invalid data like nan or infinite by values
    INPUT
        df is the dataframe to work on
        label of the category (string) where invalid data will be replaced
    OUTPUT
        df is the dataframe modified
    '''
    # Replace nan values by 0
    df[label] = df[label].fillna(0)

    # get the maximum ot he values which is non-infinite
    #  to ensure providing a finite value as maximum when needed
    serie = list()
    for val in df[label]:
        if val != np.inf:
            serie.append(val)
    maxi = max(serie)
    # Replace infinite (inf) values by the maximum of the serie
    df[label].replace({np.inf: maxi, -np.inf: maxi}, inplace=True)  

    return df

# -----------------------------------------------------------------------------
def dummy_data(df, cat):
    '''
    DESCRIPTION
        add dummy data to df removing the source of dummy
    INPUT
        df is the initial dataframe to work on
        cat is the name of the category we want dummying
    OUTPUT
        df_dummy is the dataframe as copy of df but with dummy columns,
         removing the source of dummy
    '''
    df_dummy = df.copy(deep=True)

    # Create a new dataframe for preparing dummies from a copy of the 
    #  selected category 
    df_cat = pd.DataFrame([])
    df_cat[cat] = np.array(df[cat])

    # Convert category values to just numbers 0 or 1 (dummy values) of category
    df_dum = pd.get_dummies(df_cat[cat], dtype=int)

    # A priori, this is not the smartest solution (join, merge, concat)
    #  but it works with adding unexpected rows that requires additional
    #  understanding while merging both dummies and df. This solution works
    #  without unexpected behaviour.
    for col in df_dum.columns:
        df_dummy[col] = np.array(df_dum[col])

    # Drop the original category column from `df_dummy`
    df_dummy.drop(columns=cat, inplace=True)

    # Remove duplicates
    # - check if duplicates before removing them
    duplicates = df_dummy.duplicated().sum()
    if duplicates > 0:
        df_dummy.drop_duplicates(keep='first', inplace=True)

    # clean df_dummy from rows with full nan values
    # otherwise, at least the row is a nana row.
    df_dummy.dropna(axis='index', how='all', inplace=True)

    return df_dummy

# -----------------------------------------------------------------------------
def get_cumul_days():
    '''
    DESCRIPTION
        Return a list to ease count, month per month, passed day from the
         beginning of the year.
    INPUT
        nil
    OUTPUT
        cnt_day_at_start_month is a list that gives the nb of past days from
         beginning of the year for every month
    '''
    cnt_day_at_start_month = []

    day_by_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cnt = 0

    for i in day_by_month:
        cnt += i 
        cnt_day_at_start_month.append(cnt)

    return cnt_day_at_start_month

# -----------------------------------------------------------------------------
def get_cnt_day_at_start_month(m, calendar):
    '''
    DESCRIPTION
        get the number of past days from the beginning of year and at 
         beginning of the selected month.
    INPUT
        m is the selected month (str of int, int or float)
        calendar is a list that gives the nb of past days from beginning of the
                 year for every month
    OUTPUT
        cnt is the count of days (int) from beginning of the year at start of
         the selected month.
    '''
    cnt = calendar[int(m)-1]
    return cnt

# -----------------------------------------------------------------------------
def get_timestamp_array(dates_str):
    '''
    DESCRIPTION
        Convert an np.array of dates in string format
        to a timestamp in number of days.
    INPUT
        dates_str is a np.array of dates YYYY-MM-DD in string format
        calendar is a list that gives the nb of past days from beginning of
                 the year for every month
    OUTPUT
        timestamp is np.array with data converted in number of days (int)
    '''
    timestamp = list()

    # Create a kind of calendar that is a list that gives the nb of past days
    #  from beginning of the year for every month.
    calendar = get_cumul_days()

    # Run along the dates_str vector for converting all dates from string to
    #  number of days (int).
    for d in dates_str:
        try:
            y = float(d[:4])
            m = float(d[5:7])
            d = float(d[8:10])
            # convert the data in number of days
            cnt_days_for_months = float(get_cnt_day_at_start_month(m, calendar))
            ts = y * 365.24219 + cnt_days_for_months + d
        except:
            ts = 0.0
        ts_r = round(ts, 0)
        timestamp.append(int(ts_r))

    return np.array(timestamp)

# -----------------------------------------------------------------------------
def add_timestamp(df):
    '''
    DESCRIPTION
        Add a column 'timestamp' in the dataframe that provide the number of 
         seconds related to the 'date' values (string).
    INPUT
        df is the dataframe to complete        
    OUTPUT
        df is hte dataframe completed with the new column 'timestamp'
    '''

    df['timestamp'] = get_timestamp_array(np.array(df['date']))
    df = replace_invalid_data(df, 'timestamp')
    df.groupby(['timestamp'])
    return df

# -----------------------------------------------------------------------------
def compute_test(df):
    '''
    DESCRIPTION
        Create a column 'tx_test' with the result of 'tx_pos' / 'tx_incid'        
    INPUT
        df is the dataframe
    OUTPUT
        df is the dataframe modified
    '''
    # div function allows supporting missing value (in case we miss something) 
    #  with fill_value
    df['tx_test'] = df['tx_pos'].div(df['tx_incid'])   
    
    # Replace invalid data
    df = replace_invalid_data(df, 'tx_test')

    return df


# -----------------------------------------------------------------------------
def compute_degrad(df):
    '''
    DESCRIPTION
        Create columns 'tx_rea' and 'tx_dchosp' as described below
    INPUT
        df is the dataframe
    OUTPUT
        df is the dataframe modified
    '''
    # Compute Rate of people admitted in intensive care over people 
    #  hospitalized
    df['tx_rea'] = df['incid_rea'].div(df['incid_hosp'], fill_value=np.nan)
    # Replace invalid data
    df = replace_invalid_data(df, 'tx_rea')

    # Compute Rate of deceases at hospital over people hospitalized.
    df['tx_dchosp'] = df['incid_dchosp'].div(df['incid_hosp'], fill_value=np.nan)    
    # Replace invalid data
    df = replace_invalid_data(df, 'tx_dchosp')

    return df

# -----------------------------------------------------------------------------
def compute_IO(df):
    '''
    DESCRIPTION
        Create column 'tx_incid_IO' as the trend of hospitals occupancy
        = ('incid_hosp' - 'incid_rad' - 'incid_dchosp') / ('hosp' + 'rea')
    INPUT
        df is the dataframe
    OUTPUT
        df is the dataframe modified
    '''

    # According to the number of operations, I prefer working on numpy array
    #  rather than on dataframe

    # get values in numpy array
    incid_hosp = np.array(df['incid_hosp'])
    incid_rad = np.array(df['incid_rad'])
    incid_dchosp = np.array(df['incid_dchosp'])
    hosp = np.array(df['hosp'])
    rea = np.array(df['rea'])

    # Avoid division by zero by replacing 0 by 1 people to allow division 
    #  in the denominator, and thus be able to even compute an change.
    # At this level, no such difference between one and zero patient.
    hospitzd = np.add(hosp, rea)  # computation of the raw denominator
    ones = np.ones(np.shape(hospitzd), dtype=type(hospitzd))
    hospitzd = np.where(np.absolute(hospitzd) < 1, ones, hospitzd)

    tx_incid_IO = np.divide(np.subtract(incid_hosp,
                                        np.add(incid_rad,
                                               incid_dchosp)),
                            hospitzd)
    df['tx_incid_IO'] = tx_incid_IO
    # Replace invalid data
    df = replace_invalid_data(df, 'tx_incid_IO')

    return df

# -----------------------------------------------------------------------------
def compute_data(df):
    '''
    DESCRIPTION
        Compute data for helping answering questions ; new data are stored in
         the dataframe.
    INPUT
        df is the dataframe to work on
    OUTPUT
        df is the dataframe with original data and new computed data 
    '''

    # Question 1
    print(' - tested people')
    df = compute_test(df)

    # Question 2
    print(" - hospitalized people's health degradation")
    df = compute_degrad(df)

    # Question BONUS 1
    print(' - Input/Output patients flow at hospital')
    df = compute_IO(df)

    # Question BONUS 2
    print(' - Add timestamp')
    df = add_timestamp(df)

    return df

# -----------------------------------------------------------------------------
def get_area_names(df, aera_id_label:str, aera_name_label: str):
    '''
    DESCRIPTION
        build a dictionary of a type or area (department or region) allowing
         getting a name of this area from its id
    INPUT
        df is the globale dataframe
        aera_id_label is the id (str) of the area: 'dep' (department) or 'reg'
         (region)
        area_name_label is the name (str) of the area: 'lib_dep' (department)
         or 'lib_reg' (region)
    OUTPUT
        dict_areas is a dictionary: id is the key, name is the value
    '''

    dict_areas = dict()
    # get the list of area id and related names
    area_ids = np.array(df[aera_id_label])
    area_names = np.array(df[aera_name_label])
    area_id_uniq = []
    for i, area_id in enumerate(area_ids):
        if area_id not in area_id_uniq:
            area_id_uniq.append(area_id)
            dict_areas[area_id] = area_names[i]
        else:
            pass
    del df, aera_id_label, aera_name_label, area_ids, area_names, area_id_uniq
    return dict_areas

# -----------------------------------------------------------------------------
def get_dep_name(dict_lib_dep, dep_id):
    '''
    DESCRIPTION
        give the name of a department in accordance with its id
    INPUT
        dep_id is the id (str, int or float) of the department from which we
         want the name
    OUTPUT
        dep_name is the name (str) of the department related to the given id
    '''
    name = None

    if type(dep_id) != 'str':
        if type(dep_id) == float:
            dep_id = int(round(dep_id, 0))
        dep_id = str(dep_id)
        if len(dep_id) == 1:
            dep_id = '0' + dep_id
    try:
        name = dict_lib_dep[dep_id]
    except:
        name = 'unknown'
    return name

# -----------------------------------------------------------------------------
def get_values_at_last_date(df, category, deps):
    '''
    DESCRIPTION
        Provide values of the selected category at the last date available in
         the dataframe, for every department.
    INPUT
        df is the dataframe
        category is the name of the columns for which we want values at the 
         last date for every department
        deps is the list of unique department number
    OUPUT
        values is a dictionary with {dep: {'date': date, 'value': value of the
         category at the stored date}}
    '''

    values = dict()

    # format a dedicated list of data for the search
    lst = [np.array(df['dep']),
        np.array(df['date']),
        np.array(df['timestamp']),
        np.array(df[category])]

    # Run over every department
    k = 0
    m = True
    cnt = 0
    cond = False
    for dep in deps:
        time_ts = 0
        # run along the rows of the dataframe
        for i, j in enumerate(lst[0]):
            # When I meet the selected department
            #  and the timestamp is higher (ensure get finally the most
            #  recent timestamp/date and related values of the category),
            #  we stored date and value of the category in values, relatively
            #  to every department.
            if (j == dep) and (time_ts < lst[2][i]):
                id_max_date = i    

        # Use the last id_max_date stored to get values at the last date 
        #  for a department.
        values[dep] = {'date': lst[1][id_max_date], 'value': lst[3][id_max_date]}

    return values

# -----------------------------------------------------------------------------
def get_timeserie(df, category, dep, duration_week):
    '''
    DESCRIPTION
        get all values of a categories along the time for a selected department
    INPUT
        df is the dataframe;
        category is the label of columns (string) for which we want the time
                 series;
        dep is the number (string) of department for which we want the time
            series;
        duration_week is the maximum number of weeks (int) for which we want
                      the time series (as far as the data set covers this
                      period).
    OUTPUT
        A list of dates (string) as available in the data set; ascending time;
        A list of values for the selected category on the period fo the 
          selected department.
    '''
    # get dataframe the selecte category with time and departments
    df_ = df[['dep', 'date', 'timestamp', category]]

    # keep only rows with the selected dep
    df_ = df_[df_['dep'] == dep]

    # sort dataframe by descending timestamp (most recent at first)
    df_.sort_values(by=['timestamp'], ascending=False, inplace=True)
    
    # get last timestamp (most recent date)
    timestamp_max = df_['timestamp'].values[0]

    # compute start of the time window
    timestamp_min = timestamp_max - 7*duration_week  # unit here is the day
                                                     # 7 days per week

    # keep only dataframe on the required timestamp period
    df_ = df_[df_['timestamp'] > timestamp_min]
    
    # get expected values and related dates
    dates = df_['date'].values.tolist()
    values = df_[category].values.tolist()

    # reverse is for getting time flow from left to right
    dates_rev, values_rev = list(), list()
    for i in range(len(dates)):
        j = len(dates) - 1 - i
        dates_rev.append(dates[j])
        values_rev.append(values[j])

#    return list(reversed(dates)), list(reversed(values))
    return dates_rev, values_rev

# -----------------------------------------------------------------------------
def format_values_for_vizu(dico_values, nb_decimal):
    '''
    DESCRIPTION
        format values for a vizualization on a map
    INPUT
        dico_values is a dictionary such {dep: {'date': date, 'value': value
           of the category at the stored date}}
        nb_decimal is the number of decimal (int) expected on the result values
    OUTPUT
        series is a dictionary such as
         {date1: {depY: valueY, ...}, date2: {...}}
    '''

    series = dict()

    dates = list()
    for dep, j in dico_values.items():
        date = j['date']
        if date not in dates: # Create a sub-dictionary at every new date
            dates.append(date)
            series[date] = dict()
        series[date][dep] = round(j['value'], nb_decimal)

    return series

# -----------------------------------------------------------------------------
def visualization_test(df):
    '''
    DESCRIPTION
        Display visualization related to test people rate last 60 days
    INPUT
        df is the dataframe
    OUTPUT
        nil (display of plots)
    '''

    # Get unique values of 'dep'
    deps = pd.unique(df['dep'])

    # get the selected values
    tx_test = get_values_at_last_date(df, 'tx_test', deps)

    # Prepare and display visualization
    fr_chart_tx_test = maps.Departments(human_readable=True)
    title_1 = 'COVID-19 test rate for 100 000 inhabitants'
    title_2 = '\nin the last 60 days (per department)'
    fr_chart_tx_test.title = title_1 + title_2
    tx_test_vizu = format_values_for_vizu(tx_test, 3)
    for date, j in tx_test_vizu.items():
        fr_chart_tx_test.add(date, j)
    fr_chart_tx_test.render_in_browser()  # display the picture

# -----------------------------------------------------------------------------
def visualization_degrad(df):
    '''
    DESCRIPTION
        Display visualization related to degradation of patient's health
         at hospital
    INPUT
        df is the dataframe
    OUTPUT
        nil (display of plots)
    '''

    # Get unique values of 'dep'
    deps = pd.unique(df['dep'])

    # Get the selected values
    tx_rea = get_values_at_last_date(df, 'tx_rea', deps) 

    # Prepare and display visualization for 'intensive care'
    fr_chart_tx_rea = maps.Departments(human_readable=True)
    title_1 = 'Rate of new people in intensive care'
    title_2 = '\n(per department)'
    fr_chart_tx_rea.title = title_1 + title_2
    tx_rea_vizu = format_values_for_vizu(tx_rea, 3)
    for date, j in tx_rea_vizu.items():
        fr_chart_tx_rea.add(date, j)
    fr_chart_tx_rea.render_in_browser()  # display the picture
    
    # Get the selected values
    tx_dchosp = get_values_at_last_date(df, 'tx_dchosp', deps) 

    # Prepare and display visualization for 'decease'
    fr_chart_tx_dchosp = maps.Departments(human_readable=True)
    fr_chart_tx_dchosp.title = 'Rate of new deceases due to COVID-19\n(per department)'
    tx_dchosp_vizu = format_values_for_vizu(tx_dchosp, 3)
    for date, j in tx_dchosp_vizu.items():
        fr_chart_tx_dchosp.add(date, j)
    fr_chart_tx_dchosp.render_in_browser()  # display the picture

# -----------------------------------------------------------------------------
def visualization_IO(df):
    '''
    DESCRIPTION
        Display visualization related to Input/output of patients at hospital
    INPUT
        df is the dataframe
    OUTPUT
        nil (display of plots)
    '''

    # Get unique values of 'dep'
    deps = pd.unique(df['dep'])

    # Get the selected values
    tx_incid_IO = get_values_at_last_date(df, 'tx_incid_IO', deps)

    # Prepare and display visualization
    fr_chart_tx_IO = maps.Departments(human_readable=True)
    title_1 = 'Flow of admission/discharge of patients at hospital in the last 24h'
    title_2 = '\n(per department)'
    fr_chart_tx_IO.title = title_1 + title_2
    tx_IO_vizu = format_values_for_vizu(tx_incid_IO, 3)
    for date, j in tx_IO_vizu.items():
        fr_chart_tx_IO.add(date, j)
    fr_chart_tx_IO.render_in_browser()  # display the picture

# -----------------------------------------------------------------------------
def space_x_labels(lst, nb_x_labels=6):
    '''
    DESCRIPTION
        replace values by empty between vzlues of the given sampling rate
        The purpose consists in reduce the number of labels on the x axis
    INPUT
        lst is of list of values
    OUTPUT
        lst_out is of list with values (string) at the sampling rate,
         empty ... otherwise
    '''
    lst_out = list()
    
    # compute sampling according the number of requested x labels
    sampling_rate = int(round(len(lst) / nb_x_labels))

    # build a reference index grid
    ref_id_grid = list()
    maxi = 1 + int(round(len(lst) / sampling_rate, 0))
    for x in range(maxi):
        ref_id_grid.append(x * sampling_rate)

    for i, val in enumerate(lst):
        if i in ref_id_grid:
            value = val
        else:
            value = ''
        lst_out.append(value)

    return lst_out

# -----------------------------------------------------------------------------
def visualization_timeseries(df, dict_lib_dep, dep='31'):
    '''
    DESCRIPTION
        Display visualization related to the evolution of number of
         hospitalized people and people in intensive care.
    INPUT
        df is the dataframe
    OUTPUT
        nil (display of plots)
    '''

    # get the full name of the selected department
    dep_name = get_dep_name(dict_lib_dep, dep)

    # get values to plot: require for the last 30 weeks at least
    nb_weeks = 30
    dates_hosp, values_hosp = get_timeserie(df, 'hosp', dep, nb_weeks)
    dates_intensive, values_intensive = get_timeserie(df, 'rea', dep, nb_weeks)

    # Adjust x vector to reduce the number of x labels displayed with the X axis.
    dates_hosp = space_x_labels(dates_hosp)

    # Prepare and display visualization
    # based on https://www.pygal.org/en/stable/documentation/types/line.html
    date_chart = pygal.Line(x_label_rotation=35)
    title_1 = "Evolution in time of the hospital occupancy"
    title_2 = "in the department of " + dep_name    
    date_chart.title =  title_1 + '\n' + title_2
    date_chart.x_labels = map(str, dates_hosp)
    date_chart.add("hospitalized", values_hosp)
    date_chart.add("intensive care", values_intensive)
    date_chart.render_in_browser()  # display the picture

# -----------------------------------------------------------------------------
def visualization(df, dict_lib_dep):
    '''
    DESCRIPTION
        Processing of all requested visualizations
    INPUT
        df is the dataframe
    OUTPUt
        nil     (display of viusalization)
    '''
    
    # Question 1
    visualization_test(df)

    # Question 2
    visualization_degrad(df)

    # Question 3
    # No visualization related to the model, only comments

    # Question BONUS 1
    visualization_IO(df)

    # Question BONUS 2
    visualization_timeseries(df, dict_lib_dep)
 
# -----------------------------------------------------------------------------
def prepare_modelization(df):
    '''
    DESCRIPTION
        Compute data for answering questions ; new data are stored in the
         dataframe.
    INPUT
        df is the dataframe to work on
    OUTPUT
        df_mdl is the dataframe with original data and new computed data 
    '''

    # Copy df for linear model
    #  in order to keep a maxim of data for processing and analsyis 
    #  and adjust the dataframe for specific modelization
    df_mdl = df.copy(deep=True)

    # Since I goes on the linear model, 
    # Remove useless string data (covers case of dummies with 'dep')
    if 'dep' in df_mdl.columns:
         # without dummies
        df_mdl.drop(columns=['date', 'dep', 'reg', 'dchosp', 'rad', 'incid_rad'], inplace=True)
    else:
        # with dummies
        df_mdl.drop(columns=['date', 'reg', 'dchosp', 'rad', 'incid_rad'], inplace=True)

    # Clean data for modelization
    # by removing any row with a invalid value
    df_mdl.dropna(axis='index', how='any', inplace=True)

    return df_mdl

# -----------------------------------------------------------------------------
def get_X_y(df: object, response:str):
    '''
    DESCRIPTION
        Split df into exploratory X data and response y data
    INPUT
           df  is the dataframe
           response  is the target parameter
    OUTPUT
           X : A matrix holding all of the parameters you want to consider
                when predicting the response
           y : the corresponding response vector
    '''
    # output
    X, y = None, None

    # Get the Response var
    if response in df.columns:
        # Split into explanatory and response variables (1/2)
        #  Get response variable
        y = df[response]
        df = df.drop(columns=[response])  # Remove pred_name from df

    else:
        print(" - CAUTION: Unable to find the response in df")
        y = None

    # Get the Exploratory vars
    # Split into explanatory and response variables (2/2)
    #  Get the input variables i.e. at this level just a copy of df
    X = df.copy(deep=True)

    if X is None:
        print(' - No X found')
    if y is None:
        print(' - No y found')

    del df, response
    return X, y

# -----------------------------------------------------------------------------
def split_to_train_test_data(X: object, y: object, testrate=.3):
    '''
    DESCRIPTION
        Split into train and test X/y data
    INPUT
        X  is explanatory variables
        y  is target response variable
        testrate is proportion of the dataset to include in
                 the test split,between 0.0 and 1.0;
                 default value = 0.3
    OUTPUT
        Xytrain is the list of input and response data for training the model
                [Xtrain, ytrain]
        Xytest is the list of input and response data for test the model
                [Xtest, ytest]
    '''
    Xtrain, Xtest, ytrain, ytest = None, None, None, None

    if (X is not None) and (y is not None):
        try:
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                        test_size=testrate,
                                                        random_state=42)
        except:
            print(' - Unable to get train and test data')

    return [Xtrain, ytrain], [Xtest, ytest]


# -----------------------------------------------------------------------------
def get_model(Xytrain, Xytest):
    '''
    DESCRIPTION
        Returns a linear prediction model according to train data,
        and return r2 scores on train and test data.
    INPUT
        Xytrain is the list of input and response data for training the model
                [Xtrain, ytrain]
        Xytest is the list of input and response data for test the model
                [Xtest, ytest]
    OUTPUT
        model : fitted linear regression model object from sklearn
    '''
    model = None

    Xtrain, ytrain = Xytrain
    Xtest, ytest = Xytest

    if (Xtrain is not None) and (Xtest is not None) and \
        (ytrain is not None) and (ytest is not None):
        # Establish model

        # Linear model from scikit-learn:
        #  https://scikit-learn.org/stable/modules/linear_model.html#

        # Linear Regression
        model = LinearRegression()

        # fit the model
        model.fit(Xtrain, ytrain)

        if model is None:
            print(" - Model not found")

    return model

# -----------------------------------------------------------------------------
def evaluate_model(model, Xytrain, Xytest, Xy):
    '''
    DESCRIPTION
        Return the score of prediction for the model
    INPUT
        model is the fitted model to evaluate
        Xytrain is a list of the train data: [Xtrain, ytrain]
        Xytest is a list of the test data: [Xtest, ytest]
    OUTPUT
        score : score of prediction of the model on the entire dataset
                Display scores for train, test and full data set
    '''
    score = 0

    # sub-variables
    mdl_score_train, mdl_score_test, mdl_score = None, None, None

    Xtrain, ytrain = Xytrain
    Xtest, ytest = Xytest
    X, y = Xy

    if model is not None:
        # Evaluate this model

        # Get the model's score
        # Return the coefficient of determination of the prediction
        mdl_score_train = model.score(Xtrain, ytrain)
        print(" - Model score (train): ", mdl_score_train)
        mdl_score_test = model.score(Xtest, ytest)
        print(" - Model score (test) : ", mdl_score_test)
        mdl_score = model.score(X, y)
        print(" - Model score        : ", mdl_score)
        # They shoudl be respectively equal or very close to r2_scores
        #  computed here-above.

        score = mdl_score

    return score

# -----------------------------------------------------------------------------
def coef_weights(model, X) -> object:
    '''
    DESCRIPTION
        Return a dataframe with coefficients of weight on every parameter
         for this mode (real and absolute values) sorted in the descending order
         of the absolute values
    INPUT
        model   : model for which we are looking coefficients
        X       : input parameters of the model
    output
        coefs_df : dataframe with model's coefficients; that can be
                    used to understand the most influential coefficients
                    in a linear model by providing the coefficient
                    estimates along with the name of the variable
                    attached to the coefficient.
    '''

    coefs_df = pd.DataFrame()
    # Get name of every column in front  of its coefficients
    coefs_df['est_int'] = X.columns
    # get coefficients of the linear model
    coefs_df['coefs'] = model.coef_
    # get absolute value of these coefficients
    coefs_df['abs_coefs'] = np.abs(model.coef_)
    # Sort coefficient by descending order
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)

    print('\t', coefs_df)

    del model, X
    return coefs_df



# -----------------------------------------------------------------------------
def main():
    if len(sys.argv) == 2:

        hospital_data_filepath = sys.argv[1]

        print('Loading data...\n    HOSPITAL: {}'
              .format(hospital_data_filepath))
        df = load_data(hospital_data_filepath)

        print('Enforce type of some data...')
        df = format_data(df)

        print("get departments dictionary") # before cleaning the data
        dict_lib_dep = get_area_names(df, 'dep', 'lib_dep')

        print('Cleaning data...')
        df = clean_data(df)

        # Add dummy values of 'dep'
        # df = dummy_data(df, 'dep')
        # After analysis, this was finally cancelled because it requires
        #  much more resources for a gain that is really not significant.

        print('Prepare data for modelization...')
        df_mdl = prepare_modelization(df)

        print('Compute data...')
        df = compute_data(df)

        print('Visualization...')
        visualization(df, dict_lib_dep)


        # Modelization        
        response = 'pos_7j'  # define the target:
                             # COVID-19 positive test on a week
        
        print('Split data into X/y...')  # Split into Response y
                                    # and Exploratory variables X
        X, y = get_X_y(df_mdl, response)
        
        print('Split data to train/test...')
        Xytrain, Xytest = split_to_train_test_data(X, y)

        print('Get trained model...')
        model = get_model(Xytrain, Xytest)

        print('\nEvaluate the model / Metrics...')
        score = evaluate_model(model, Xytrain, Xytest, [X, y])
        
        print('\nEvaluate inputs of the model...')
        # Evaluate inputs of the model: Define impact of input parameter
        coef_df = coef_weights(model, X)

    else:
        print('While located in the related local folder,\n'
              'Please provide the filepath of the dataset file as first '\
              'argument. \n\nExample: python COVID19_France_process_data.py '\
              'COVID19_France_data.csv')

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()