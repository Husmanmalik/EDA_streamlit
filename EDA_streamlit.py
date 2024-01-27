import streamlit as st 
import pandas as pd 
import numpy as np 
import seaborn as sns 
# from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# webapp title
st.markdown('''
# **Exploratory Data Analysis web application**
''')
# how to upload file from pc

with st.sidebar.header("Upload your Dataset(.csv)"):
    uploaded_file = st.sidebar.file_uploader("Upload your file",type=['csv']) # adding a sidebar on the rightside of page
    df=sns.load_dataset('titanic')
    st.sidebar.markdown('[Example csv file](df)')
# profiling report for pandas
if uploaded_file is not None: #agar upload file is none na ho to
    @st.cache
    def load_csv(): #function define kiya ha 
        csv=pd.read_csv(uploaded_file)
        return csv
    df=load_csv()
    pr=ProfileReport(df,explorative=True)
    # display the profile report in sidebar
    st.header('**Input Df**')
    st.write('---')
    st.header('**Profiling report with pandas**')
    st_profile_report()
else:
    st.info("Awaiting for CSV file,")# bhai kuch upload to kar do
    if st.button('Press to use example Data'):
      #example dataset
       @st.cache
       def load_data(): #shwoing that if data is upload then perform the same function on this random dataframe
           a=pd.DataFrame(np.random.rand(100,5),
                                 columns=['age','banana','codanic','codanic','Dr.aamar'])
           return a
       df=load_data()
       pr =ProfileReport(df,explorative=True)
       st.header('**Input Dataframe**')
       st.write('---')
       st.header('**Pandas Profiling Report**')
       st_profile_report()

    

