import matplotlib.pyplot as plt
import pandas as pd

# Constants for Action Labels
ACTION_LABELS = {
    0: 'No Action Required',
    1: 'Action Required (Not Urgent)',
    2: 'Urgent Action Required'
}

# Constants for Age Brackets
AGE_BRACKETS = ['11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90']


# Function to plot bar chart
def plot_bar_chart(labels, frequencies, title):
    plt.figure(figsize=(8, 6))
    plt.bar(labels, frequencies, color='skyblue')
    plt.xlabel('Risk Level')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()


# Function to plot pie chart
def plot_pie_chart(labels, frequencies, title):
    plt.figure(figsize=(8, 6))
    plt.pie(frequencies, labels=labels, autopct='%1.1f%%', startangle=140,
            colors=['lightgreen', 'lightcoral', 'skyblue'])
    plt.title(title)
    plt.show()


# Exploratory Data Analysis (EDA)
def perform_eda(data):
    # Capitalize 'Gender' column values
    data['Gender'] = data['Gender'].str.capitalize()

    frequencies = [data[data['Risk'] == label].shape[0] for label in ACTION_LABELS.values()]
    plot_bar_chart(ACTION_LABELS.values(), frequencies, 'Frequency of Different Risk Levels')
    total_samples = len(data)
    percentages = [f"{(f / total_samples) * 100:.2f}%" for f in frequencies]
    plot_pie_chart(ACTION_LABELS.values(), frequencies, 'Percentage of Different Risk Levels')

    # Gender Pie Chart
    gender_counts = data['Gender'].value_counts()
    plot_pie_chart(gender_counts.index, gender_counts.values, 'Gender Distribution')

    # Bar Chart for Total Samples by Gender
    plot_bar_chart(gender_counts.index, gender_counts.values, 'Total Samples by Gender')

    # Age Bracket Pie Chart
    age_counts = pd.cut(data['Age'], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90], labels=AGE_BRACKETS).value_counts()
    plot_pie_chart(age_counts.index, age_counts.values, 'Age Distribution')

    # Bar Chart for Total Samples by Age
    plot_bar_chart(age_counts.index, age_counts.values, 'Total Samples by Age Bracket')

    # Total number of samples in male and female categories
    male_samples = data[data['Gender'] == 'Male'].shape[0]
    female_samples = data[data['Gender'] == 'Female'].shape[0]
    print(f"Total Number of Samples in Male Category: {male_samples}")
    print(f"Total Number of Samples in Female Category: {female_samples}")


# Data Cleaning
def data_cleaning(data):
    # Check for missing values
    missing_file_names = data['FileName'].isnull().sum()
    missing_risk_labels = data['Risk'].isnull().sum()
    missing_abnormality = data['Abnormality'].isnull().sum()
    print('Checking for missing values')
    print(
        f'Missing values found in Column: FileName {missing_file_names}, Risk: {missing_risk_labels}, Abnormality: {missing_abnormality}')

    data['FileName'] = data['FileName'].astype(str)
    data['Risk'] = data['Risk'].astype(str)
    data['Abnormality'] = data['Abnormality'].astype(str)

    # Concatenate columns 'FileName', 'Risk', and 'Abnormality'
    data['Concatenated'] = data[['FileName', 'Risk', 'Abnormality']].agg('-'.join, axis=1)

    # Check for duplicates based on the concatenated column
    duplicate_rows = data[data.duplicated(subset=['Concatenated'], keep=False)]

    # Count the number of duplicates
    duplicate_count = len(duplicate_rows)

    if duplicate_count > 0:
        print(f'Duplicate rows found based on the concatenated column: {duplicate_count}')
        # Remove duplicates, keeping the first occurrence
        data = data.drop_duplicates(subset=['Concatenated'], keep='first')
        print('Duplicate rows removed.')
    else:
        print('No duplicate rows based on the concatenated column.')

    return data


# Read CSV file
def read_data(filename):
    return pd.read_csv(filename)


if __name__ == "__main__":
    # File name
    filename = '/home/algoryc/Projects/wellnest-ecg-ai/ECGResults.csv'

    data = read_data(filename)  # Read Data
    perform_eda(data)  # Exploratory Data Analysis (EDA)
    data = data_cleaning(data)  # Data Cleaning
