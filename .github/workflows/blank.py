name: import data

on: [push]

jobs:
  build:

    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v1
    - name: Run a one-line script
      run: echo Hello, world!
    - name: Run a multi-line script
      run: |
        echo Add other actions to build,
        echo test, and deploy your project.
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#create dictionary: {'collumn name':[value]}
data = {'year ': [
2010 , 2011 , 2012 ,
2010 , 2011 , 2012 ,
2010 , 2011 , 2012],
'team ': ['FCBarcelona ', 'FCBarcelona ',
'FCBarcelona ', 'RMadrid ',
'RMadrid ', ' RMadrid ',
'ValenciaCF ', 'ValenciaCF ',
'ValenciaCF '],
'wins ': [30, 28, 32, 29, 32, 26, 21, 17, 19],
'draws ': [6, 7, 4, 5, 4, 7, 8, 10, 8],
'losses ': [2, 3, 2, 4, 2, 5, 9, 11, 11]}
football = pd.DataFrame(data , columns = [
'year ', 'team ', 'wins ', 'draws ', 'losses'])
#transform dictionary to dataframe: pd.DataFrame()
p=pd.DataFrame(data)
'''import data: pd.read_csv(orexcel, html ... etc.)('filename.csv', na_values="see what symbol for not available
", usecols"select what collumn data to retrieve")'''
edu=pd.read_csv('educ_uoe_fine01_1_Data.csv', na_values= ':', usecols=['GEO', 'UNIT', 'TIME','Value'])
#see first and last fime rows data: head() and 
edu_head=edu.head() 
edu_tail=edu.tail()
edu_describe=edu.describe()
