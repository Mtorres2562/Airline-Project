import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


air = pd.read_csv('data/train.csv')
air = air.drop(columns=["Unnamed: 0", "id"], errors="ignore")
# Drop NaN's
air = air.dropna()
air['Arrival Delay in Minutes'] = air['Arrival Delay in Minutes'].astype('int64')
# Set Page title and icon
st.set_page_config(page_title="Airline Satisfaction Explorer", page_icon="‍✈️")

# Sidebare Navigation
page = st.sidebar.selectbox("Select a Page",["Home", "Data Overview", "Exploratory Data Analysis", "Model Training and Evaluation", "Make Predictions"])

# Home page
if page == "Home":
    st.title("✈️ Airline Satisfaction Explorer")
    st.subheader("Welcome to our Airline Satisfaction Explorer App")

    st.divider()

    st.write("""
             This app analyzes the Airline Satisfaction Dataset to uncover what makes passengers satisfied or dissatisfied. 
             We’ll explore key features, visualize trends that help airlines soar or struggle, and examine relationships between different factors.
             Plus, we’ll even make predictions on new data to better understand passenger satisfaction.
             """)
    
    st.divider()

    st.image('https://imgc.artprintimages.com/img/print/boeing-747-the-world-s-largest-and-fastest-jetliner-at-the-boeing-manufacturing-plant_u-l-p73wl30.jpg?background=F3F3F3')
    st.markdown("Image used from [ART.com](https://www.art.com/products/p15568316-sa-i3787938/boeing-747-the-world-s-largest-and-fastest-jetliner-at-the-boeing-manufacturing-plant.htm?upi=P73WL30&PODConfigID=4990704&sOrigID=20314)")

# Data Overview
elif page == "Data Overview":
    st.title("📋 Data Overview")
    st.subheader("About the Data")
    st.write("""
             Flying is a major part of travel. In the United States, the average person takes approximately 1.4 trips by air each year. 
             In this dataset, we will analyze 103,594 passengers based on various factors such as gender, travel class, customer type, and more. 
             Additionally, we will examine passenger satisfaction through various rating questions (scored from 0 to 5) to determine what contributes to a satisfying or dissatisfying flight experience
             """)
    
    st.divider()

    st.image('https://i.pinimg.com/originals/68/27/3f/68273f919da8f1d47b781b67fd4ea35e.jpg')
    st.markdown("Image used from [Pinterest](https://in.pinterest.com/pin/767019380330060686/)")

    st.divider()

    # Shape of data
    st.subheader("Quick Glance of Data")
    st.markdown(f"""The Airline Satisfaction Dataset consists of {air.shape[0]} rows and {air.shape[1]} columns and 
                is sourced from [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data).  
                Click one below to view.""")
    # Dataframe
    if st.checkbox("Show DataFrame"):
        st.dataframe(air)
    # Dictionary
    if st.checkbox("Show Dictionary"):    
        st.markdown(
        """
        | Columns    | Definition |
        | ---------- | :--------- |
        | Gender | Gender of the passengers (Female, Male) |
        | Customer Type  | The customer type (Loyal customer: 1, disloyal customer: 0)  | 
        | Age   | The actual age of the passengers   | 
        | Type of Travel    | Purpose of the flight of the passengers (Business Travel: 1, Personal Travel: 0)    |
        | Class  | Travel class in the plane of the passengers (Business: 2, Eco Plus: 1, Eco: 0)  | 
        | Flight Distance   | The flight distance of this journey   | 
        | Inflight Wifi Service    | Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)    |
        | Departure/Arrivale Time Concenient  | Satisfaction level of Departure/Arrival time convenient  | 
        |  Ease of Online booking  | Satisfaction level of online booking   | 
        | Gate location    | Satisfaction level of Gate location    |
        | Food and drink  | Satisfaction level of Food and drink  | 
        | Online boarding   | Satisfaction level of online boarding   | 
        | Seat comfort    | Satisfaction level of Seat comfort    |
        | Inflight entertainment  | Satisfaction level of inflight entertainment  | 
        | On-board service   | Satisfaction level of On-board service   | 
        | Leg room service    | Satisfaction level of Leg room service    |
        | Baggage handling  | Satisfaction level of baggage handling  | 
        | Check-in service   | Satisfaction level of Check-in service   | 
        | Inflight service    | Satisfaction level of inflight service    |
        | Cleanliness  | Satisfaction level of Cleanliness  | 
        | Departure Delay in Minutes   | Minutes delayed when departure   | 
        | Arrival Delay in Minutes    | Minutes delayed when Arrival    |
        | Satisfaction  | Airline satisfaction level(Satisfaction: 1, neutral or dissatisfaction: 0)  |     
        """,
        unsafe_allow_html=True)

# Exploratory Data Analysis (EDA)
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.sidebar.subheader("Select the type of visualtization You'd like to expore.")
    eda_type = st.sidebar.selectbox("Visualization Options", ['Histograms', 'Scatterplots', 'Box Plots', 'Bar Charts'])

    obj_cols = air.select_dtypes(include = 'object').columns.tolist()
    num_cols  = air.select_dtypes(include ='number').columns.tolist() 

    # Histogram plot
    if 'Histograms' in eda_type:
        st.subheader("Histograms-Visualizing Numerical Distribution")
        h_selected_col = st.selectbox("Select a numerical column for the histogram:", num_cols)
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace('_',' ')}"
            if st.checkbox("Show by Gender"):
                st.plotly_chart(px.histogram(air, x = h_selected_col, color = 'Gender', title = chart_title, barmode= 'overlay'))
            elif st.checkbox("Show by Customer Type"):
                st.plotly_chart(px.histogram(air, x = h_selected_col, color = 'Customer Type', title = chart_title, barmode= 'overlay'))
            elif st.checkbox("Show by Type of Travel"):
                st.plotly_chart(px.histogram(air, x = h_selected_col, color = 'Type of Travel', title = chart_title, barmode= 'overlay'))
            elif st.checkbox("Show by Class"):
                st.plotly_chart(px.histogram(air, x = h_selected_col, color = 'Class', title = chart_title, barmode= 'overlay'))
            elif st.checkbox("Show by Satisfaction"):
                st.plotly_chart(px.histogram(air, x = h_selected_col, color = 'satisfaction', title = chart_title, barmode= 'overlay'))
            else:
                st.plotly_chart(px.histogram(air, x = h_selected_col, title = chart_title))
    
    # Scatter Plot
    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols)
        if selected_col_x and selected_col_y:
            chart_title = f"{selected_col_x.title().replace('_', ' ')} vs. {selected_col_y.title().replace('_', ' ')}"
            if st.checkbox("Show staisfaction"):
                st.plotly_chart(px.scatter(air, x=selected_col_x, y=selected_col_y, color='satisfaction', title=chart_title))
            elif st.checkbox("Show Class"):
                 st.plotly_chart(px.scatter(air, x=selected_col_x, y=selected_col_y, color='Class', title=chart_title))
            elif st.checkbox("Show Type of Travel"):
                 st.plotly_chart(px.scatter(air, x=selected_col_x, y=selected_col_y, color='Type of Travel', title=chart_title))
            elif st.checkbox("Show Customer Type"):
                 st.plotly_chart(px.scatter(air, x=selected_col_x, y=selected_col_y, color='Customer Type', title=chart_title))
            elif st.checkbox("Show Gender"):
                 st.plotly_chart(px.scatter(air, x=selected_col_x, y=selected_col_y, color='Gender', title=chart_title))
            else:
                 st.plotly_chart(px.scatter(air, x=selected_col_x, y=selected_col_y, title=chart_title))
    # Box plot
    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for the box plot:", num_cols)
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            if st.checkbox("Show staisfaction"):
                st.plotly_chart(px.box(air, x=b_selected_col, y='satisfaction', title=chart_title, color='satisfaction'))
            elif st.checkbox("Show Class"):
                 st.plotly_chart(px.box(air, x=b_selected_col, y='Class', title=chart_title, color='Class'))
            elif st.checkbox("Show Type of Travel"):
                 st.plotly_chart(px.box(air, x=b_selected_col, y='Type of Travel', title=chart_title, color='Type of Travel'))
            elif st.checkbox("Show Customer Type"):
                 st.plotly_chart(px.box(air, x=b_selected_col, y='Customer Type', title=chart_title, color='Customer Type'))
            elif st.checkbox("Show Gender"):
                 st.plotly_chart(px.box(air, x=b_selected_col, y='Gender', title=chart_title, color='Gender'))
            else:
                 st.plotly_chart(px.box(air, x=b_selected_col, title=chart_title))        
    # Bar Chart
    if 'Bar Charts' in eda_type:
        st.subheader("Bar Chart - Visualizing Categorical Distribution")
        selected_col = st.selectbox("Select a categorical variable:", obj_cols)
        if selected_col:
            chart_title = f'Distribution of {selected_col.title()}'
            st.plotly_chart(px.bar(air, y=selected_col, color=selected_col, title=chart_title))

# Model Training and Evaluation
elif page == "Model Training and Evaluation":
    st.title("🛠️ Model Training and Evaluation")

    # First clean data
    # Change the column into a numerical with 1 and 0
    air['satisfaction'] = air['satisfaction'].map({'satisfied':1, 'neutral or dissatisfied':0})
    # Change the column into a numerical with 1 and 0
    air['Type of Travel'] = air['Type of Travel'].map({'Business travel':1, 'Personal Travel':0})
    # Change the column into a numerical with 1 and 0
    air['Customer Type'] = air['Customer Type'].map({'Loyal Customer':1, 'disloyal Customer':0})
    # Change the column into a numerical with 2, 1, and 0
    air['Class'] = air['Class'].map({'Business':2, 'Eco Plus':1, 'Eco':0})

    # Sidebar for model selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X = air.drop(columns = ['satisfaction', 'Gender'])
    y = air['satisfaction']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.number_input("Select the number of neighbors (k) from 1-20", min_value=1, max_value=20, value=3)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    # Train the model on the scaled data
    model.fit(X_train_scaled, y_train)

    # Display training and test accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.2f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Greys')
    st.pyplot(fig)
    if model_option == "Random Forest":
        st.write(f"""We can see that the Test Accuracy: {model.score(X_test_scaled, y_test):.2f} far exceeds that of K-Nearest Neighbors, 
                 indicating that Random Forest is the best-suited model for this dataset.
                 """)

# Make Predictions Page
elif page == "Make Predictions":
    st.title("✈️ Make Predictions")

    # First clean data
    # Change the column into a numerical with 1 and 0
    air['Type of Travel'] = air['Type of Travel'].map({'Business travel':1, 'Personal Travel':0})
    # Change the column into a numerical with 1 and 0
    air['Customer Type'] = air['Customer Type'].map({'Loyal Customer':1, 'disloyal Customer':0})
    # Change the column into a numerical with 2, 1, and 0
    air['Class'] = air['Class'].map({'Business':2, 'Eco Plus':1, 'Eco':0})

    st.subheader("Adjust the values below to make predictions on the Iris dataset:")

    # User inputs for prediction
    customer_type = st.slider("Customer Type - Loyal Customer: 1, Disloyal Customer: 0", min_value=0, max_value=1, value=1)
    age = st.slider("Age", min_value=5, max_value=85, value=53)
    travel_type = st.slider("Type of Travel - Business travel: 1, Personal Travel: 0", min_value=0, max_value=1, value=1)
    travel_class = st.slider("Class - Business: 2, Eco Plus: 1, Eco: 0", min_value=0, max_value=2, value=0)
    flight_distance = st.slider("Flight Distance", min_value=0, max_value=5000, value= 2000)
    inflight_wifi = st.slider("Inflight wifi service", min_value=0, max_value=5, value=2)
    time_convenient = st.slider("Departure/Arrival time convenient", min_value=0, max_value=5, value=3)
    online_booking = st.slider("Ease of Online booking", min_value=0, max_value=5, value=1)
    gate_location = st.slider("Gate location", min_value=0, max_value=5, value=0)
    food_drink = st.slider("Food and drink", min_value=0, max_value=5, value=2)
    online_boarding = st.slider("Online boarding", min_value=0, max_value=5, value=3)
    seat_comfort = st.slider("Seat comfort", min_value=0, max_value=5, value=1)
    entertainment = st.slider("Inflight entertainment", min_value=0, max_value=5, value=3)
    on_board_service = st.slider("On-board service", min_value=0, max_value=5, value=2)
    leg_room = st.slider("Leg room service", min_value=0, max_value=5, value=3)
    baggage_handling = st.slider("Baggage handling", min_value=0, max_value=5, value=5)
    checkin_service = st.slider("Checkin service", min_value=0, max_value=5, value=0)
    inflight_service = st.slider("Inflight service", min_value=0, max_value=5, value=2)
    cleanliness = st.slider("Cleanliness", min_value=0, max_value=5, value=3)
    departure_delay = st.slider("Departure Delay in Minutes", min_value=0, max_value=5, value=3)
    arrival_delay = st.slider("Arrival Delay in Minutes", min_value=0, max_value=5, value=1)

    # User input dataframe
    user_input = pd.DataFrame({
        'Customer Type': [customer_type],
        'Age': [age],
        'Type of Travel': [travel_type],
        'Class': [travel_class],
        'Flight Distance': [flight_distance],
        'Inflight wifi service': [inflight_wifi],
        'Departure/Arrival time convenient': [time_convenient],
        'Ease of Online booking': [online_booking],
        'Gate location': [gate_location],
        'Food and drink': [food_drink],
        'Online boarding': [online_boarding],
        'Seat comfort': [seat_comfort],
        'Inflight entertainment': [entertainment],
        'On-board service': [on_board_service],
        'Leg room service': [leg_room],
        'Baggage handling': [baggage_handling],
        'Checkin service': [checkin_service],
        'Inflight service': [inflight_service],
        'Cleanliness': [cleanliness],
        'Departure Delay in Minutes': [departure_delay],
        'Arrival Delay in Minutes': [arrival_delay]
    })

    st.write("### Your Input Values")
    st.dataframe(user_input)

     # Use KNN (k=9) as the model for predictions
    model = KNeighborsClassifier(n_neighbors=9)
    X = air.drop(columns = ['satisfaction', 'Gender'])
    y = air['satisfaction']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Train the model on the scaled data
    model.fit(X_scaled, y)

     # Make predictions
    prediction = model.predict(user_input_scaled)[0]

     # Display the result
    st.write(f"The model predicts that the passenger is: **{prediction}**")