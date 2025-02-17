import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Heart Disease App")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Analysis", "Prediction"])

df = pd.read_csv("dataset.csv")
x = df.drop(columns=['target'])
y = df['target']

x =pd.get_dummies(x) # convert alfabet into numerical data 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
model = RandomForestClassifier(n_estimators=100, random_state=42) # 2. define 
model.fit(X_train, y_train)#3. fit on trainig data model sikh gaya data se 

y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)


if page == "Introduction":
    st.title("Welcome to the Heart Disease Prediction Model")
    st.write("Top five rows of the dataset")
    st.write(df.head())
    st.write("### Dataset Statistics")
    st.write(df.describe())

elif page == "Analysis":
    st.title("Dataset Analysis")
    st.write("Use the options below to explore the dataset and model results.")

    if st.button("Show Confusion Matrix"):
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_test)
        fig, ax = plt.subplots(figsize=(2,2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Normal", "Heart disease"], yticklabels=["Normal", "Heart Disease"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig)

    if st.button("Show Dataset Statistics"):
        st.subheader("Dataset Statistics")
        st.write(df.describe())

elif page == "Prediction":
    st.title("Patient's Report")
    st.write("Enter the input data of the patient to check whther the patient have heart disease or not")

    st.write(f"The accuracy of the model on testing data is  : {accuracy}")


    input_data = {}
    for col in x.columns:
        input_data[col] = st.number_input(f"Enter value for {col}", value=0.0, format="%.2f")

    
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        st.success(f"Predicted Report: {'Yes' if prediction[0] == 1 else 'No'}")
 