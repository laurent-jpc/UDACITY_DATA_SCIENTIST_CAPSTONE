===============================================================================
PROJET/PURPOSE:

UDACITY - DATA SCIENCE project: Data Science Capstone


===============================================================================
TITLE:

Monitoring per department of the COVID-19 situation in France (16-Dec-2022)
Rebound of contamination and hospitalization in the last month.


===============================================================================
DESCRIPTION:

Business Understanding

- Question 1: What is the rate of the population per department which is tested
   for COVID-19? According to the global health policy, it could allow
   adjusting locally the communication for increasing test of people. 
   
- Question 2: What is the level of degradation of hospitalized COVID-19 patients
   per department in the last 24h? It could help to quickly identify where to
   reinforce resources for treating patients in specific department when needed
   and increase probability to save lifes with limited means.
   
- Question 3: What is the trend of admission/discharge at the hospitals? 
   It would give a trend of hospitals occupancy per department.
   
- Question 4: What is the evolution of the patients in intensive care and 
  dead in my department in the last days or weeks?   
   
- Question 5: How many people would be declared positive to the Covid-19 test
   in the upcoming week per department according to the present situation? 
   It would help to early manage logistics of patients and health resources
   and increase communication related to safety precaution.


===============================================================================
DATA UNDERSTANDING:
	
Data access:

Data are stored in the csv file named "COVID19_France_data.csv". It is provided
 (below the GITHUB uploading 30Mo-limit) with "," used as separator.
This csv data file is read and transpose in dataframe via the python script
 provided on GITHUB.

 
Data Description:

- Description of data - context:

	'date'    = date (object) when the information is given: YYYY-MM-DD;
	
	'dep'     = number (str or int)of the french department;	
	
	'reg'     = number (int) of the french region;
	
	'lib_dep' = name (object) of the french department;
	
	'lib_reg' = name (object) of the french region.

- Description of data - Hospital situation:

	'hosp'         = number (int) of patients currently hospitalized due to
					  COVID-19;
					  
	'incid_hosp'   = number (float) of new hospitalized patients in the last
					  24h;
					  
	'rea' is the   = number (int) of patients currently in resuscitation or
					  intensive care unit;
					  
	'incid_rea'    = number (float) of new patients who were admitted to the 
				     resuscitation unit in the last 24h;
					 
	'rad'          = cumulative number (int) of patients who where
					  hospitalized for COVID-19 but back to home due to 
					  improvement of their health;
					  
	'incid_rad'    = number (float) of the patients back to home in the last
					  24h;
					  
	'reg_rea'       = undefined (int);
	
	'reg_incid_rea' = undefined (float).
	
- Description of data - decease due to COVID-19:

	'dchosp'       = number (int) of decease at hospital;
	
	'incid_dchosp' = number (float) of new patients deceased at the hospital
					  in the last 24h.
  
- Description of data - tests:  
  
	'pos'      = number (float) of people declared positive (D-3 date of test);
	'pos_7j'   = number (float) of people declared positive on a week
				  (D-3 data of test);
	'cv_dose1' = undefined (float).

- Description of data - COVID-19 epidemic monitoring indicators:  
  
	'tx_pos'   = Positivity rate (float) is the number of people tested
			      positive (RT-PCR or antigenic assay) for the first time in the
			      last 60 days over the number of people tested (positive or
			      negative) on a given period, without being tested positive
			      in the last 60 days;
				  
	'tx_incid' = Incidence rate (float) is the number of people tested
			      positive (RT-PCR or antigenic assay) for the first time in
				  the last 60 days over the size of population; it is given
				  for 100 000 of inhabitants;
				  
	'TO' 	   = Occupancy rate (float) is the number of hostpitalized COVID-19
		          patients over the initial number of beds at hospital (before 
				  increase of this number).
				  
	'R' 	   = Virus replication rate (float) is the average number of people
				  that can be contaminated by a infected person.
			      R>1, epidemic is spreading. R<1, epidemic is declining.

		  
Analysis of the need to answer the questions:
		  
According to Question 1, I would propose to compute the number of people tested
 for COVID-19 (positive or negative) over the size of the population, expressed
 for 100 000 inhabitants. So, to get current rate 'nb_test / pop (100 000 hab)',
 I will need last 'tx_incid' and 'tx_pos' of every department.

According to Question 2, I would propose to monitor in the last 24h the
 degradation of health of hospitalized people and people in intensive care unit.
 Thus I propose to monitor the rate "nb of people admitted in intensive care
 over nb of people hospitalized" ('incid_rea' / 'incid_hosp' = 'tx_rea') and
 "nb of decease over nb of people admitted in intensive car" ('incid_dchosp' /
 'incid_hosp' = 'tx_dchsop') per department.

According to Question 3, I would propose to monitor the trend of hospitals
 occupancy per department. Thus I propose to monitor the rate of Input/Output 
 of patients at the hospital in the last 24h as adding new people hospitalized
 minus new people back to home and new  over the number of patients at the
 hospital: ('incid_hosp' - 'incid_rad' - 'incid_dchosp') / ('hosp' + 'rea')  

According to Question 4, I would propose to plot the evolution of the
 patients in intensive care and dead in my department along the time in the
 six last months?    
 
According to Question 5, I would propose to build a prediction model would give 
 the number of people would be declared positive to the Covid-19 test in the
 upcoming week per department according to the current sitation. Model will use
 'pos_7j' as target according to information information of the day of report.

 
Data processing:

Observations reported below have been extablished during preparation of the
 project while studying the data on Jupyter  with Python scripts. These scripts
 where not implemented in the python script that aims at answering the questions.

 
- There is only one csv data file; no need to merge several data sheets;


- There is no row or column that contains only nan values;


- Irrelevant and outliers per columns:

	> Names of the french departments and regions can be removed after creating
	  a dictionary to find the name from the number for further use.
	  
	> Undefined parameters 'reg_rea', 'reg_incid_rea' and 'cv_dose1' can be
       removed because useless without a description;
	   
	> Values of of 'dep' are given in string format from '01' to '32' then in
       int format from 32 to 95 plus 971 to 976. All values will be converted
	   in string format due to presence of values '2A' and '2B' rather than
	   in int format.
	
	> Globaly, what is called rate of something is not normalized by definition.
	  Actually, we note that rates like 'tw_pos' contains numerous values high
	  above 1 for this reason.
	
	> 'tx_pos' contains values higher than 1. It contains 5.9% of nan.
	
	> 'tx_incid' contains 5.9% of nan
	
	> 'R' contains 85% nan; we gonna remove this column since too many values
       are missing.
	   
	> 'incid_hosp' contains 0.1% of nan
	
	> 'incid_rea' contains 0.1% of nan
	
	> 'incid_rad' contains 0.1% of nan
	
	> 'incid_dchosp' contains 0.1% of nan 
	
	> 'reg_incid_rea' contains 0.1% of nan
	
	> 'pos' contains 5.9% of nan
	
	> 'pos_7j' contains 5.9% of nan
	
	> 'cv_dose1' contains 99.9% of nan; we gonna remove this column since too
       many values are missing.
	   
  Finally, following columns can be removed: 'reg_rea', 'reg_incid_rea',
   'cv_dose1' and 'R'. Parameters 'lib_dep' and 'lib_reg' can be remove only
   after creation of the dedicated dictionary.

   
- Irrelevant and outliers per row:

	> Only 0.1% of rows contains 50% or more of nan. We can remove these
	  almost-nan rows without loosing too much data.
	  
	> Globally, we see that on remaining columns, we would loose at maximum
	  5.9% of data if we remove all rows with at least a nan. I consider this
	  is acceptable. Actually, we really loose 5.9% ofd rows.

	  
- Enforce format to data in object format:
	> 'date' in str ; format is YYYY-MM-DD
	> 'dep' in str ; notably due to presence of department number '2A' and '2B'
	  and because the visualization function use string format of normalized
	  department number as given here.
	> 'lib_dep' in str (before creating the dictionary)
	> 'lib_reg' in str (before creating the dictionary)
  This should be done while reading the csv file because otherwise, we got an 
   warning Low memory message. Add "dtype={"date": str, "dep": str, 'lib_dep':
   str, 'lib_reg': str}" cancels the warning.
    > Although initially it seemed to be not needed to convert 'reg' to 'str',
	  build of the dictionary reveals it is; so it has to be done!
 
There is no dummy parameters.
At a time, it was considered to use 'dep' (departement -string-) values as
 dummies. Nerverthless, considering the huge additional quantity of data
 (+101 dummy columns in the dataframe against 19 before dummies) and the
 non-significant difference in scores of the LinearRegression model with
 and with dummies (between -0.4% to -0.5% ; see below the scores in chapter
 'Modelization'), I decided to not use these dummies.


===============================================================================
PREPARE DATA 

Data processing:

According to Question 1, I compute a rate 'nb_test / pop (100 000 hab)',
  dividing 'tx_pos' by 'tx_incid' (nb_positive/nb_test over
 nb_positive/nb_pop). Then I get a 'rate' which is the nb of people tested 
 for COVID-19 (positive or negative) over the size of the population, given
 for 100 000 inhabitants. I call the result column 'tx_test'.

 
According to Question 2, I compute both rates to monitor in the last 24h 
  the degradation of health of hospitalized people and people in intensive car
  unit as follows:

- Rate of people admitted in intensive care over people hospitalized:
   'incid_rea' / 'incid_hosp' = 'tx_rea'

- Rate of deceases at hospital over people admitted in intensive care:
   'incid_dchosp' / 'incid_rea' = 'tx_dchsop' 

For this both operations, it helps to replace nan by 0 and inf values by 
the maximum of the numeric values to provide a reasonable finite a high
value instead of infinite.
 
 
According to Question 3, I compute the Rate of Patient-Input/Output at
 hospital as
 ('incid_hosp' - 'incid_rad' - 'incid_dchosp') / ('hosp' + 'rea') 

When necessary, I replace the sum of 'hosp' + 'rea' (denominator) by 1 when
 it equals to zero to be able to show the change. Actually, there is not a
 big difference in this context between zero people.
 
For this both operations, it helps to replace nan by 0 and inf values by 
the maximum of the numeric values to provide a reasonable finite a high
value instead of infinite.


According to Question 4, there is no computation to perform. Nevertheless, 
there is a bit of preparation to capture the appropriate timeseries.
I propose to display data on the last 30 weeks, both hospitalizations and 
deceases in the same plot using the same axis to keep ratio between both.


According to Question 5, I built a prediction model to give 'pos_7j' as target
 according to other information, except the date because I would appreciate a
 situation only regarding the other figures with dependencies with the time.
 For the occasion I create a dataframe as a copy of the current dataframe.
 Thus I can apply additional processing to the dataframe dedicated to data
 modelization.

Basically, I choose a linear model since the dataframe contains mostly figures.
 It implies that data in string format must be used: categories 'date' and 'dep'
 are removed from the model-dataframe .
 
This model is fitted with 70% ( train) of the dataset and tested with 30%
 (test). Since the model is based on a linear regression, I decided to use
 model's score as metrics to validate the model (Best possible score is 1.0).
 
In addition, computation of model's coefficients indicate that 'reg',
 'dchosp', 'rad' and 'incid_rad' have very low influence (<0.5) on the model.
 I choose to remove them too from the model-dataframe.
 Coefficients of parameters are provided at the end of the run of the code.
 
The LinearRegression's score without 'dep' dummies:
 >  0.8815 (train); 0.8739 (test)
The LinearRegression's score with 'dep' dummies (+101 dummies):
 >  0.8857 (train); 0.8775 (test)
Use of dummies values provide slightly better result but not significant 
 according to the huge additional quantity of data.
The code for dummy is implemented but not used in the version delivered.
Other scikit's linear models give same results or worst, so I keep this one.
 So my model has a rather satisfactory score.


===============================================================================
VISUALIZATION:

Visualization required additional processing to get the appropriate values 
 and provide these values in the appropriate format according to the 
 visualization function.

Graphical visualizations are displayed on pages of the default web browser.
 
Concerning Question 1, 2 and 3, visualization is based on the display of
 results on a map of France, splited by department. Every department as a
 color according to the related value.

Concerning Question 4, visualization consists in a plot of two curves, values
 of two parameters along the time.
 
Concerning Question 5, visualization consists in display of score and 
 coefficients of the model.
 
 
Explain the results:


Question 1:

	The graph "COVID-19 test rate for 100 000 inhabitants in the last 60 days
     (per department)" shows a map of France with level of color consistent
     with the level of the rate per department at the date of 16-Dec-2022.
	This map shows that COVID-19 testing behaviour among people is rather 
	 homogeneous but low in mainland France while is 2 or 3 times more on 
	 islands in Guyane, Réunion and Mayotte.

	 
Question 2:

	The graph "Rate of new people in intensive care (per department)" shows 
	 a map of France with level of color consistent with the level of the rate
	 per department at the date of 16-Dec-2022.
	This map shows there is some departments with significant level mainly in
     region of Paris, the middle of France, in the South-west and the 
	 North-East, plus The Reunion.

	The graph "Rate of new deceases due to COVID-19e (per department)" shows 
	 a map of France with level of color consistent with the level of the rate
	 per department at the date of 16-Dec-2022.
	This map shows a France with a level of decease due to COVID-19 globaly
     low except in some departments of the South-west, North-East and in
	 Guyane with rates between 1.5 and 3 times higher than elsewhere.

Question 3:

	The graph "Flow of admission/discharge of patients at hospital in the last 
	 24h (per department)" shows a map of France with level of color consistent
	 with the level of the rate per department at the date of 16-Dec-2022.
	Although some departments see the number of patients at hospitals decreasing,
	the majority of departments slightly increased their number of patients these
	last 24h.

	
Question 4:

	The graph "Evolution in time of the hospital occupancy in the department of
     Haute-Garonne" shows a plot of two lines: one for the number of 
	 hospitalized people, a second for the number of people in intensive care.
	In my department of Haute-Garonne, the graph shows a continuous increase of
     hospitalization since beginning of November: the level was multiplied by
 	 1.5 in a month.
	In parallel, the number of deceases at the hospital remains increased in the 
	 same period with a level multiplied by 3 in a month, reaching at last 40
     deceases per day due to COVID-19. This number remains in the result of 
	 these last months.
   
   
Question 5: Modelization
   The model to predict COVID-19 positive cases per week gives rather good
   result
    with a global score of 0.88.
   Analysis of the contribution of every parameters shows that the result is
    mainly linked with the Occupancy rate (TO) which is actuall rather 
	consequence of the prediction.
   It shows a high level a correlation between the number of positive cases
    and the amount of work in the hospitals despite the vaccination campaign.

   
===============================================================================
FINDINGS

About 6% of data were removed from the initial data set due to missing values.
 It concerns mainly first set of data at the beginning of the data collection.
 Indeed, method, measure, collection and presentation of data change in time
 before getting the current format of data.
 Nevertheless, we benefit from a huge amount of valid data for the analysis.
 
Computation of some rates raised the problem of division by 0. While taking 
 about group of people and with the will of showing result for every 
 department, I replace 0 by 1 when necessary.
 
I also face infinite values; then I replaced with infinite values by maximum
 numerical values of the serie, in order to be reasonable realistic and show
 result.

Although testing several type of models, I kept the Linear Regression which
 give rather good results with an implementation rather simple. 
I wonder if some other type of models could be more suitable in this case.
 

===============================================================================
VERSION:

ID: 1.2.0

In comparison with previous version 1.1.0, this version 1.2.0 brings following
 changes:

 - Database was updated
 
 - The whole python code was review and modified. It is not implemented in a 
   .py file rather than on a Jupyter Python file.
   
 - The post was totaly rewritten and illustrated.


===============================================================================
INSTRUCTIONS:

- Create local folder as workspace for this programme;
- Load the csv data sheet file into this workspace;
- Load the python script  file into this workspace;
- Install the required librairies: pip install -r requirements.txt;
- Open a Command prompt or equivalent.
- Change the directory to be in the workspace: cd ... workspace
- then enter "python COVID19_France_process_data.py COVID19_France_data.csv"

			
===============================================================================			
PUBLIC RELEASE: UPDATE!

You can find the published results here:
https://medium.com/@laurent.jp.costa/rebound-of-covid-19-contamination-in-france-how-we-use-to-live-with-it-d180162048ba


===============================================================================
ENVIRONMENT:

Refer to the file requirements.txt

It may be necessary to install GTK for visualization.
 source code: https://gtk-win.sourceforge.io/home/index.php/Main/Downloads


===============================================================================
REPOSITORY’S FILES: UPDATE!

File “README.md”
File “COVID19_France_data.csv” – data sheet file under csv format providing
 hospital data related to COVID 19 in France.
File "requirements.txt" - librairies required for the proper execution of 
 the programme.



===============================================================================
DATA SOURCE:

Hospital data: 

- Name: Santé publique France
- Licence: Open licence version 2.0 / ID 60190d00a7273a8100dd4d38 / (https://www.etalab.gouv.fr/licence-ouverte-open-licence/)
- Link: https://www.data.gouv.fr/fr/datasets/synthese-des-indicateurs-de-suivi-de-lepidemie-covid-19/
- Update: 19-Dec-2022
- Data extraction date: 20-Dec-2022
- Request for reuse: Indicate the following link in my presentation of results: https://www.data.gouv.fr/fr/datasets/synthese-des-indicateurs-de-suivi-de-lepidemie-covid-19/


===============================================================================
LEGAL:

Reuse and exploitation of source data: Open licence (refer to DATA SOURCE).