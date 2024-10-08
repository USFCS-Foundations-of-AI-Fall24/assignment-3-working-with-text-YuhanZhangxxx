import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def chapter_one(df):
    print("Classification Column:")
    print(df['class'])
    return df['class']


def chapter_two(df):
    most_common_class = df['class'].mode()[0]
    proportion = df['class'].value_counts(normalize=True)[most_common_class]
    print(f"Most common class (ZeroR): {most_common_class} ({proportion:.2%})")
    return most_common_class, proportion


def chapter_three(df):
    # Filter for recurrence events
    filtered = df[df['class'] == 'recurrence-events']
    # Get most common age and menopause status
    most_common_age = filtered['age'].mode()[0]
    most_common_menopause = filtered['menopause'].mode()[0]
    print(f"Most common age for recurrence: {most_common_age}")
    print(f"Most common menopause status for recurrence: {most_common_menopause}")
    return most_common_age, most_common_menopause


def chapter_four(df):
    # Filter for recurrence events
    recurrence_df = df[df['class'] == 'recurrence-events']
    # Remove entries with missing 'age' values
    recurrence_df = recurrence_df.dropna(subset=['age'])
    # Define the correct age group order
    age_order = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
    age_counts = recurrence_df['age'].value_counts().reindex(age_order).fillna(0).astype(int)

    # Total count check
    total_recurrences = age_counts.sum()
    expected_recurrences = len(recurrence_df)

    if total_recurrences != expected_recurrences:
        print("Warning: The total counts do not match!")
    else:
        print("Total counts match.")

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x=age_counts.index, y=age_counts.values, order=age_order, color="blue")
    plt.title("Number of Recurrences for Each Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Number of Recurrences")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def print_results(chapter_name, function, df):
    print(f"{chapter_name}:")
    result = function(df)
    # Avoid printing large outputs for Chapter 1 and Chapter 4
    if chapter_name == "Chapter 1":
        print(result.head())  # Show only first few entries
    elif chapter_name != "Chapter 4":
        print(result)
    print("\n")


if __name__ == '__main__':
    # Define column names
    column_names = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps',
                    'deg-malig', 'breast', 'breast-quad', 'irradiat']
    # Read the dataset, handling missing values and specifying delimiter
    df = pd.read_csv('breast-cancer.data', names=column_names, na_values='?', delimiter=',')

    # Total entries before dropping missing values
    total_entries_before = len(df)
    print(f"Total number of entries in dataset before cleaning: {total_entries_before}")

    # Handle missing values in 'class' and 'age'
    df = df.dropna(subset=['class', 'age'])

    total_entries_after = len(df)
    print(f"Total number of entries in dataset after cleaning: {total_entries_after}\n")

    results = {
        "Chapter 1": chapter_one,
        "Chapter 2": chapter_two,
        "Chapter 3": chapter_three,
        "Chapter 4": chapter_four
    }

    for name, func in results.items():
        print_results(name, func, df)





