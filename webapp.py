import streamlit as st 
from streamlit_option_menu import option_menu 

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

models = {
            'Logisitc Regression': LogisticRegression(),
            'C-Support Vector Classification.': SVC(),
            'Random Forest Classifier': RandomForestClassifier(),
            'Ridge Classifier': RidgeClassifier(),
            'K Neighbors Classifier': KNeighborsClassifier(),
            'Decision Tree Classifier': DecisionTreeClassifier()}

drugs = ['Alcohol',
         'Amyl',
         'Amphet',
         'Benzos',
         'Caff',
         'Cannabis',
         'Coke',
         'Crack',
         'Ecstasy',
         'Heroin',
         'Ketamine',
         'Legalh',
         'LSD',
         'Meth',
         'Mushrooms',
         'Nicotine',
         'VSA'    ]

def create_data(column):
    drug_df = df_ML.copy()
    drug_df[column] = np.where(drug_df[column] >= 3, 1, 0)
    return drug_df

def preprocessing_inputs(df, column):
    df = df.copy()
    
    # Split df into X and y
    y = df[column]
    X = df.drop(column, axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = pd.DataFrame(scaler.transform(X_train), 
                           index=X_train.index, 
                           columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), 
                          index=X_test.index, 
                          columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test

def models_for_drug_consumer(data, binarizer):
    model_names =[]
    drug_names = []
    values = []
    l = []

    for drug in drugs:
        if binarizer == True:
            data = create_data(drug)
        X_train, X_test, y_train, y_test = preprocessing_inputs(data, drug)
        drug_names.append(drug)
        for name, model in models.items():
            model_names.append(name)
            model.fit(X_train, y_train)
            yhat = model.predict(X_test)
            acc = accuracy_score(y_test, yhat)
            values.append(acc) 

    split_values = [values[x:x+len(models)] for x in range(0, len(values), len(models))]
    model_names = model_names[:len(models)]
    df_results = pd.DataFrame(split_values, columns = model_names)
    df_results = df_results.set_axis(drug_names)
    return df_results

def displayHeatmap(df, transpose, color):

    if transpose == True:
        df = df.T

    df_l = df.values.tolist()

    fig = go.Figure(data=go.Heatmap(
        z=df_l,
        x=list(df.columns.values),
        y=list(df.index.values),
        colorscale=color))
    
    return fig

def Map(col,name):
    df_map = df_copy.groupby([col, name], as_index=False).count()
    df_map = df_map[[col, name, 'Gender']]


    df_map = df_map.pivot(index=col, columns=name, values="Gender")
    del df_map[0]
    df_map.rename(columns={1: name}, inplace=True)
    return df_map

df_analysis = pd.read_csv("df_analysis.csv",sep=";")
df_ML = pd.read_csv("ML.csv",sep=";")
ex = False

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ["Home","Data Analysis","Machine Learning"]
    )
    
    if selected == "Data Analysis":
        filter_options =  st.expander('Filter options', expanded=ex)
        with filter_options:
            st.header("Filtering")
            country = st.multiselect(
                'Select the country:', 
                options = df_analysis['Country'].unique(),
                default = df_analysis['Country'].unique()
            )
            

            age = st.multiselect(
                'Select age:', 
                options = df_analysis['Age'].unique(),
                default = df_analysis['Age'].unique()
            )

            gender = st.multiselect(
                'Select gender:', 
                options = df_analysis['Gender'].unique(),
                default = df_analysis['Gender'].unique()
            )

            education = st.multiselect(
                'Select education:', 
                options = df_analysis['Education'].unique(),
                default = df_analysis['Education'].unique()
            )

            ethnicity= st.multiselect(
                'Select ethnicity:', 
                options = df_analysis['Ethnicity'].unique(),
                default = df_analysis['Ethnicity'].unique()
            )

            df_filter = df_analysis.query(
                "Country == @country & Age == @age & Gender == @gender & Education == @education & Ethnicity == @ethnicity"
            )
            df_filter = df_filter.reset_index(drop=True)


if selected == "Home":
    st.title('Introduction of the Database')
    st.markdown("This database is a study made on the consumption of various drugs, the survey asked diffents individuals from different gender, age, education, country and ethnicity")
    st.markdown("Participants were questioned concerning their use of 18 legal and illegal drugs")
    st.markdown("For each drug they have to select one of the answers: never used the drug, used it over a decade ago, or in the last decade, year, month, week, or day.")

    table = ['ID','Age','Gender','Education','Country','Ethnicity','Nscore','Escore','Oscore','Ascore','Cscore','Impulsive','SS','Alcohol','Amphet','Amyl','Benzos','Caff','Cannabis','Choc','Coke','Crack','Ecstasy','Heroin','Ketamine','Legalh','LSD','Meth','Mushrooms','Nicotine','Semer','VSA']
    df_init = pd.read_csv("drug_consumption.data", sep =',', header=None,names= table)
    df_init = df_init.drop(columns=["ID"])
    st.write(df_init)

    st.markdown("The data set contains 1885 rows and 32 columns. From the documentation of the dataset, we have decided to delete the rows where they declared to have used semer as Semer is class of fictitious drug Semeron consumption. Moreover, we have also decided to delete chocolate as it is not medically considered as a drug.")

    st.markdown("Data Cleaning")
    st.markdown("In this dataset, we don't have any null values. During the pre-processing process, we noticed that the mean values of all the columns were more or less equal to 0. We concluded that we didn't have the need to impute the data.")
    st.markdown("For each drug in the dataset, we have CL0, CL1, CL2, CL3, CL4, CL5, CL6 and CL7 which represent the frequency of use of the drug. We are going to encode these values to 0, 1, 2, 3, 4, 5, 6 and 7 as it will be easier for the machine learning part.")
    st.markdown("[Link for more information and the download page](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29)")

    st.markdown("Table after Cleaning and Label Encoding")
    st.write(df_analysis)

    st.markdown("Here we have the description of the table you can see the mean, and the quartiles as well as the max and min values")
    st.write(df_ML.iloc[:,12:].describe())
    

if selected == "Data Analysis":
    st.title('Data Analysis')
    st.markdown('---')
    if len(df_filter) != 0:
        st.header('Filtered Database')
        st.subheader(f'{len(df_filter)} Individuals')
        st.markdown("You can filter this table on the sidebar and if you click on one of the variable on the table, you can order it by the values in the column")
        st.markdown("To get the initial order you have to click on the index column")
        st.write(df_filter.iloc[:,:5])
    
    option = st.selectbox('Display Pie and Bar Chart', ['Age','Gender','Education','Country','Ethnicity'])

    st.markdown("Here we will show two graphs (pie and bar charts) to illustrate 5 the five categorical variables.")
    st.markdown("You just have to select the variable you wish to get a vizualisation in the selectbox")

    df_tot = df_analysis[option].value_counts().rename_axis('unique_values').reset_index(name=f"{option}")

    pie_chart = px.pie(
        df_tot, 
        values=option, 
        names="unique_values",
        title= f"{option} Percentage"
        )
    pie_chart.update_traces(pull=0.1)

    bar_chart = px.bar(df_tot, x='unique_values', y=option)
    bar_chart.update_layout(
        xaxis_title=option,
        yaxis_title="number of individuals"
    )


    #Gender distribution in different countries
    df_gender = df_analysis[['Gender', 'Country']].value_counts().reset_index(name='count')
    df_age = df_analysis[['Age', 'Country']].value_counts().reset_index(name='count')
    df_gender_age = df_analysis[['Gender', 'Age']].value_counts().reset_index(name='count')
    fig_gender_country = px.bar(df_gender, x='Country', y="count", color="Gender")
    fig_age_country = px.bar(df_age, x='Country', y="count", color="Age")
    fig_gender_age = px.bar(df_gender_age, x='Age', y="count", color="Gender")


    #Drug consumption analysis
    dict_drugs = {}

    for drug in drugs:
        dict_drugs[drug]=(df_analysis[drug]>=4).sum()

    df_drugs = pd.DataFrame(list(dict_drugs.items()),columns = ['Drug','Users last month']).sort_values(by=['Users last month'])


    drugs_used_last_month = px.bar(df_drugs, x = 'Users last month', y = 'Drug', title="Drugs Consumed Last Month")
    drugs_used_last_month2 = px.bar(df_drugs[:-3],x = 'Users last month', y = 'Drug', title="Drugs Consumed Last Month (without Alcohol, Caffeine, Nicotine)")

    st.header(f'{option} Pie chart')
    st.plotly_chart(pie_chart, use_container_width=True)
    
    st.header(f'{option} Bar Chart')
    st.plotly_chart(bar_chart, use_container_width=True)

    if option == "Age":
        st.markdown("We can see that more than 50 % of the people questioned are between 18-34 years old. Which is comparatively young audience.")
    elif option == "Gender":
        st.markdown("We can see that we have approximately 50 % of male and 50% of female.")
    elif option == "Education":
        st.markdown("College or Universities student with or without degree make up to more that 50% of the people that were questioned")
    elif option == "Country":
        st.markdown("Most of the questioned people are from the UK.")
    elif option == "Ethnicity":
        st.markdown("The ethnicity of the questioned users is not well distributed as we have 91.37% of white people. We can say that the dataset is biased.")

    st.header("Gender distribution in Countries")
    st.plotly_chart(fig_gender_country, use_container_width=True)
    st.markdown("We can see that in UK we asked more people who identify as female")

    st.header("Gender distribution by Age")
    st.plotly_chart(fig_gender_age, use_container_width=True)

    st.header("Age distribution in Countries")
    st.plotly_chart(fig_age_country, use_container_width=True)

    st.plotly_chart(drugs_used_last_month, use_container_width=True)
    st.markdown("The most used items are caffeine, alcohol and Nicotine but these are not illegal and are widely used. We can say that the most used drug last month is Cannabis.")
    st.plotly_chart(drugs_used_last_month2, use_container_width=True)
    st.markdown("If we take out the legal drugs we can see that the most consumed ones are : Cannabis, Benzos and Extasy.")
    st.markdown("We do not take into consideration Legalh which is considered as legal high consumption, but it is still in the category of hard drugs")

    st.header("Heatmaps")

    #Heatmaps
    df_map = df_analysis.groupby(['Age', 'Education'], as_index=False).count()
    df_map = df_map[['Age', 'Education', 'LSD']]
    df_map = df_map.pivot(index='Age', columns='Education', values="LSD")

    education_order = ['Left school before 16 years',
                    'Left school at 16 years',
                    'Left school at 17 years',
                    'Left school at 18 years',
                    'Some college or university, no certificate or degree',
                    'Professional certificate/ diploma',
                    'University degree',
                    'Masters degree',
                    'Doctorate degree']

    df_map = df_map[education_order]

    heatmap1 = displayHeatmap(df_map, True, 'YlGnBu')
    
    st.markdown("Education and Age distribution")
    st.plotly_chart(heatmap1, use_container_width=True)
 
    # HeatMap2
    df_copy=df_analysis[['Age','Gender','Education','Alcohol','Amphet','Amyl','Benzos','Caff','Cannabis','Coke','Crack','Ecstasy','Heroin','Ketamine','Legalh','LSD','Meth','Mushrooms','Nicotine','VSA']].copy()
    for drug in drugs:
        df_copy[drug] = np.where(df_analysis[drug] >= 3, 1, 0) 

    df_test=Map('Age','Cannabis')

    for drug in drugs:
        df_test[drug] = Map('Age',drug)[drug]

    st.markdown("Drug consumption and Age distribution")
    heatmap2 = displayHeatmap(df_test, True, 'YlGn')
    
    st.plotly_chart(heatmap2, use_container_width=True)

    # HeatMap3
    df_test2=Map('Education','Cannabis')
    df_test2 = df_test2.reindex(index = education_order)

    for drug in drugs:
        df_test2[drug] = Map('Education',drug)[drug]
    
    
    heatmap3 = displayHeatmap(df_test2, False, 'PuRd')
    heatmap3.update_layout(
        xaxis = go.layout.XAxis(
            tickangle = 45)
    )

    st.markdown("Drug consumption and Education distribution")
    st.plotly_chart(heatmap3, use_container_width=True)

    st.markdown("Without any surprises we have Alcohol, caffeine and nicotine widely used among some college or university students. Cannabis, Legalh, Ecstasy, Amphet, Benzos and mushrooms are the principal drugs used by some college, university undergraduate students in the last year.")
    st.markdown("We had the impression that those who left school would be drug users, but here we can see that drugs are mainly used by students.")
    
    # Boxplot
    st.header("Boxplot")
    
    df_analysis['Hard drug user'] = False

    hard_drugs = ['Amphet',
                'Coke',
                'Crack',
                'Ecstasy',
                'Heroin',
                'LSD',
                'Meth']

    for hard_drug in hard_drugs:
        df_analysis['Hard drug user'] = np.where((df_analysis['Hard drug user']==True) | (df_analysis[hard_drug]>=4), True, False)
    
    df_score = df_analysis[['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS', 'Hard drug user']]
    score_l = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
    for score in score_l:
        boxplot1 = px.box(df_score[[score, 'Hard drug user']], y=score, x='Hard drug user')

        st.plotly_chart(boxplot1, use_container_width=True)
        boxplot1.update_layout(yaxis_title=score)
    
    st.markdown("We see that hard drug users are more neurotic, open to experience and impulsive. Furthermore, they are less agreeable and conscientious.The effects of hard drug use on behaviour are very negative.")
    
if selected == "Machine Learning":
    st.title('Machine Learning')
    st.markdown("After the encoding of the data in the drug columns we get this database for testing the machine learning models :")
    st.write(df_ML)

    st.header("Risk of being a certain drug's consumer")
    st.markdown("We will now test 6 different models of Classification and try to predict, if the individual has consumed the drug in question")
    st.markdown("First we will try without binarizing the drug we want to predict. When we binarize, the model will predict if the individual has consumed the drug in question over the last 12 months (1 year)")
    st.markdown("It might take a while to perform 6 models on 17 drug colums.")
    st.markdown("The purpose is to find the best model for each drug, to evaluate this we will calculate the accuracy of each model")
    st.markdown("Click on the button to start the analysis !")
    if st.button('Start'):

        df_results = models_for_drug_consumer(df_ML, binarizer = False)
        df_binarized_results = models_for_drug_consumer(df_ML, binarizer = True)
        
        resultHeatmap = px.imshow(df_results.transpose())
        resultHeatmap.update_layout(title = 'Accuracy of each model tested on each drug')
        BinResultHeatmap = px.imshow(df_binarized_results.transpose())
        BinResultHeatmap.update_layout(title = 'Accuracy of each model tested on each drug (binarized data)')

        df_plots = df_results.copy()
        df_plotsB = df_binarized_results.copy()

        df_plotsT = df_plots.transpose()
        df_plotsB_T = df_plotsB.transpose()

        LineGraphDrug = go.Figure()

        for drug in drugs:
            LineGraphDrug.add_trace(go.Scatter(
            x=list(models.keys()),
            y=df_plotsT[drug].sort_values(),
            name=drug,
            ))
        LineGraphDrug.update_layout(title='Best predicted drugs')

        LineGraphModel = go.Figure()

        for model in list(models.keys()):
            LineGraphModel.add_trace(go.Scatter(
            x=drugs,
            y=df_plots[model].sort_values(),
            name=model
            ))
        LineGraphModel.update_layout(title='Best model performance ')

        LineGraphDrugB = go.Figure()

        for drug in drugs:
            LineGraphDrugB.add_trace(go.Scatter(
            x=list(models.keys()),
            y=df_plotsB_T[drug].sort_values(),
            name=drug,
            ))
        LineGraphDrugB.update_layout(title='Best predicted drugs (Binarized data)')

        LineGraphModelB = go.Figure()

        for model in list(models.keys()):
            LineGraphModelB.add_trace(go.Scatter(
            x=drugs,
            y=df_plotsB[model].sort_values(),
            name=model
            ))
        LineGraphModelB.update_layout(title='Best model performance (Binarized data)')

        st.subheader("General Analysis : Heatmaps")

        st.plotly_chart(resultHeatmap)
        st.plotly_chart(BinResultHeatmap)

        st.markdown("For the non binarized database, the interval of accuracy lies between 29 % and 86%, which is a big interval. We can add to this that the accuracy of certain models isn't great")
        st.markdown("The accuracy changes when we try to binarize the drug we want to find. The accuracy rises and we get the interval of accuracy between 68% and 97 %.")
        st.markdown('It is much better')
        
        st.subheader("General Analysis : Line Charts")

        st.plotly_chart(LineGraphDrug)
        st.markdown("Without binarization the best predicted drugs are crack and heroin.")

        st.plotly_chart(LineGraphModel)

        df_mean=pd.DataFrame(df_results.mean(),columns=["mean"])
        st.markdown("If we do the mean of each model's performance we get this table")
        st.write(df_mean)
        fig_mean = px.line(df_mean, title='Mean of the accuracy for each model')
        st.markdown("To put it visually :")
        st.plotly_chart(fig_mean)
        st.markdown("The best performing model is Random Forest classifier with an average accuracy of 69%")

        st.subheader("Binarized data")

        st.plotly_chart(LineGraphDrugB)
        st.markdown("The best predicted drug is caffeine for the binarized data but if we only take 'hard dugs' it is Heroin.")
        
        st.plotly_chart(LineGraphModelB)

        df_bin_mean=pd.DataFrame(df_binarized_results.mean(),columns=["mean"])
        st.markdown("If we do the mean of each model's performance we get this table")
        st.write(df_bin_mean)
        fig_bin_mean = px.line(df_bin_mean, title='Mean of the accuracy for each model')
        st.markdown("To put it visually :")
        st.plotly_chart(fig_bin_mean)
        st.markdown("The best predicted model for the binarized data is also random forest but we can see that the accuracy is higher : 89%.")

        st.subheader("Conclusion")
        st.markdown("After this analysis we can conclude that our best model to predict all drug consumers is on average Random Forest")
        st.markdown("However, we can apply a different model that performed the best on the drug in question. For instance, Logistic Regression for Meth or Support Vector Machine for LSD")



