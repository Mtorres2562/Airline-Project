# **Airline Satisfaction Prediction Project**  

## **Overview**  
In this project, I analyzed and predicted airline customer satisfaction using machine learning models. I worked with the "Airline Passenger Satisfaction" dataset from Kaggle, focusing on data cleaning, exploratory data analysis (EDA) with Plotly, and predictive modeling using K-Nearest Neighbors (KNN), Random Forest, and Logistic Regression. The results were then presented through an interactive Streamlit app.  

## **Dataset**  
The dataset used in this project is the "Airline Passenger Satisfaction" dataset, which can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data). I specifically worked with the `train.csv` file, which includes key passenger information such as flight distance, seat comfort, and inflight entertainment.  

## **Project Workflow**  

### **1. Data Cleaning**  
I started by reading the `train.csv` file into a Jupyter Notebook and performed the following data cleaning tasks:  

- Handled missing and inconsistent data to ensure a reliable dataset.  
- Converted categorical variables into numerical values where necessary.  
- Removed irrelevant or redundant columns, explaining the reasoning behind each decision.  

### **2. Exploratory Data Analysis (EDA) with Plotly**  
Using Plotly, I created interactive visualizations to uncover patterns and trends affecting customer satisfaction. Key insights included:  

- **Histograms:** Showcased distributions of key numerical features like flight distance.  
- **Scatter Plots:** Explored relationships between passenger features and satisfaction levels.  
- **Box Plots:** Highlighted variations in ratings for seat comfort and inflight services.  
- **Bar Charts:** Compared categorical data, such as satisfaction levels across different customer demographics.  

### **3. Predictive Modeling**  
I built and evaluated multiple machine learning models to predict airline passenger satisfaction:  

- **K-Nearest Neighbors (KNN)** – Used to classify satisfied vs. unsatisfied customers.  
- **Logistic Regression** – A baseline model to establish predictive performance.  
- **Random Forest** – Leveraged to improve accuracy through ensemble learning.  

Each model’s performance was assessed using metrics like accuracy, precision, and recall. I also compared them against a baseline model and performed hyperparameter tuning to enhance results.  

### **4. Streamlit App Development**  
To make the project more interactive, I developed a **Streamlit app** that allows users to explore EDA results and model predictions. Features include:  

- **User-friendly interface** with clear visualizations and summaries.  
- **Interactive elements**, such as sliders and selection boxes, for adjusting model inputs.  
- **Engaging design**, incorporating headers, emojis, and images for an intuitive experience.  
