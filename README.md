# Final-project-python
### ARTAUD Lucas, DOLLET Maryam, SIVASUBRAMANIAM Iswarya DIA 1

### Streamlit : (https://maryam-dollet-pythonfinalprojectm1-webapp-f7c7d9.streamlit.app/)
<!-- Table of Contents -->
# :notebook_with_decorative_cover: Table of Contents

- [About the Project ](#star2-about-the-project)
- [Libraries](#star2-libraries)
- [Dataset](#star2-dataset)
- [ Data cleaning and encoding ](#star2-data-cleanig-and-encoding)
- [ Data Visualisation](#star2-datat-visualisation)
- [ Machine learning model ](#star2-machine-learning)
- [API](#star2-API)
- [Conclusion](#star2-conclusion)

  
  
 <!-- About the Project -->
## :star2: About the Project

We are going to use the drug consumption data set to answer to our problem. (https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29)

### Our problem 
#### How can we target drug users to limit drug consumption? We should prevent the usage of drugs among the population that is why we are here, members of the drug control committee, to show the extent of this problem. But also to show how you, government, can make a difference in this.
![image](https://img.freepik.com/premium-vector/stop-drugs_73729-84.jpg?w=360)


 <!-- Libraries -->
## :star2: Libraries

#### For this project we are going to use several libraries:
- pandas
- numpy
- matplotlib.pyplot
- seaborn
- bokeh
- sklearn
- streamlit 
- streamlit_option_menu
- pyplot.express
- plotly.graph_objects

 <!-- Dataset -->
## :star2: Dataset

The data set contains 1885 rows and 32 columns. 
From the documentation of the dataset, we have decided to delete the rows where they declared to have used semer as Semer is class of fictitious drug Semeron consumption. Moreover, we have also decided to delete chocolate as it is not medically considered as a drug.


 <!-- Dataset cleaning and encoding -->
## :star2: Dataset cleaning and encoding

In this dataset, we don't have any null values.
During the pre-processing process, we noticed that the mean values of all the columns were more or less equal to 0. 
We concluded that we didn't have the need to impute the data.


For each drug in the dataset, we have CL0, CL1, CL2, CL3, CL4, CL5, CL6 and CL7 which represent the frequency of use of the drug.
We are going to encode these values to 0, 1, 2, 3, 4, 5, 6 and 7 as it will be easier for the machine learning part.


 <!-- Data visualisation -->
## :star2: Data visualisation

For the analysis part, we had to replace the values in the imputed dataset in order to make understandable visualisation.

This part's goal was to draw a typical user profile.

From this part we concluded that the typical profile was :
- Age : 18-34

![image](https://cdn.discordapp.com/attachments/1019278025981972592/1049256150496972820/image.png)

- Country : UK
 ![image](https://cdn.discordapp.com/attachments/1019278025981972592/1049257286209654855/image.png)

- Ethnicity : White
![image](https://cdn.discordapp.com/attachments/1019278025981972592/1049257343935852654/image.png)

- Gender : Male
![image](https://cdn.discordapp.com/attachments/1019278025981972592/1049257147734687758/image.png)

- Education : Some college or university, no certificate or degree
![image](https://cdn.discordapp.com/attachments/1019278025981972592/1049257073986248774/image.png)


- Used drug :      
- legal : Alcohol, Nicotine
- Illegal : Cannabis, Ecstasy, Amphet


We also noticed that the dataset was biased as there were more white people questioned.



 <!-- Machine Learning model -->
## :star2: Machine Learning

We decided to test 6 most common machine learning algorithms for our classification
- 'Logisitc Regression'
- 'Support Vector Machines'
- 'Random Forest Classifier'
- 'Ridge Classifier'
- 'K Neighbors Classifier'
- 'Decision Tree Classifier'

![image](https://cdn.discordapp.com/attachments/1019278025981972592/1049260598623473665/image.png)

![image](https://cdn.discordapp.com/attachments/1019278025981972592/1049260678256545802/image.png)


We can see that the average accuracies are between 58 and 70%. 

In order to improve our model we decided to binarize the data and considering that the person is a drug user for a particular drug if he used it in the last year.

![image](https://cdn.discordapp.com/attachments/1019278025981972592/1049262005191704586/image.png)

![image](https://cdn.discordapp.com/attachments/1019278025981972592/1049262102516351009/image.png)

We can see that the average accuracies are higher with binarization.

 <!--API -->
## :star2: API

In our API, we will ask the age, gender, education, country, ethnicity and a drug.

We made some tests in order to chose the best model for our API and it was logistic regression.

We developed our API with streamlit.

 <!--Conclusion -->
## :star2: Conclusion

It’s important to focus on cannabis which is the first illicit drug and also on young people, especially on men.

Moreover, to have a detailed report we should do another survey and diversify the people we questioned (especially their ethnicity and age) in order to have a more precise model to predict the drug used from the details given. 

We have to do some prevention and also implement more drug rehabilitation centers near schools and colleges.






