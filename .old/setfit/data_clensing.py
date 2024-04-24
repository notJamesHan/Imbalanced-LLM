import pandas as pd
import csv


# data clensing
# x is a ..
def data_clensing():
    with open("./dataset/creditcard.csv", "r") as file:
        # Create a CSV reader object
        reader = csv.DictReader(file)

        # Initialize an empty list to store
        column_data = []
        class_data = []

        # Counter for rows processed
        count = 0

        # Iterate over each row in the reader object
        for row in reader:
            count += 1
            # Break after 10 rows
            if count == 10:
                break

            # Create an empty list to store the formatted column values
            col_temp_data = []

            # Iterate over the columns (keys) in the row dictionary
            for column, value in row.items():
                if column == "Class":
                    class_data.append(value)
                else:
                    # Format the column name and value as a string
                    formatted_value = f"{column} is a {value}"
                    col_temp_data.append(formatted_value)

            # Join the formatted column values into a single string
            formatted_row = ", ".join(col_temp_data)

            # Append the formatted row to the column_data list
            column_data.append(formatted_row)

        return column_data, class_data


# Print the formatted data
left, right = data_clensing()
print(left)

print(right)

# Divdes into test data and train data

# Training
# the a is x, ... is the class A or B?
# predicts
# backpropagate


# Testing
# Give me X number of class A data.
# Spits out.

# Turn into Tabular data

# predicts, check if correct or not

# DecisionTreeClassifier()
