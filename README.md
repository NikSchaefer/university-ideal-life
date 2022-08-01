
# University Ideal Life Analysis & Prediction

Machine Learning Model used to analyze and predict student satiscation and stress levels






## Usage/Examples

Enter your data and run the model
```py
# Input your own data here to predict your target variable.
# Use visualize.py to see the categorical types that best fit you
sample = {
    "Career": "UGRD",
    "Citizenship": "Country Citzen",
    "Nationality": "Singapore",
    "Year since Matriculation": 4,
    "Year of Study": 4,
    "Primary Programme": "Bachelor of Computing",
    "Gender": "M",
    "Department": "School of Science",
    "Housing Type": "Out of Campus",
    "Q1-How many events have you Volunteered in ?": 4,
    "Q2-How many events have you Participated in ?": 4,
    "Q3-How many activities are you Interested in ?": 6,
    "Q4-How many activities are you Passionate about ?": 5,
    "Q7-How much effort do you make to interact with others ?": 3.0,
    "Q8-About How events are you aware about ?": 3.0,
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = model.predict(input_dict)

prob = tf.nn.sigmoid(predictions[0])

print("probability you are satisfied with your student life: ", prob)
```


## How it works

### Preprocessing

The data the survey provides is of several types including both categorical and numerical data. During preprocessing we aim to normalize this data. We do this by creating an input layer for each column of data. For numerical features we use a simple normailzation layer to place the values between 0 and 1 while still maintain a proportional deviance.

For the categorical data we use either a StringLookup or IntegerLookup in order to map the categorical values into a multi-hot encoding layer. It works by turning the string or integer representation into an array representation of either 1s or 0s. This makes it much easier to analyze the data.

Now that each column of data has their own preprocessing layers and normalization complete we can move on the model

### Model

The model is comprised of 7 layers alternating between Dense and Dropout. We start with a larger dense layer to begin to pick apart our data. Their are many dropout layers because much of our data is likely irrelevant to stress levels. As we continue the Dense layers get smaller until we finally create an output layer with 1 unit. This is the percentage we are sure the student is satisfied in the case we are prediction student stress levels.

We combine the inputs and outputs to build our model. In this case we use binary_crossentropy because we are mapping to a stress level whether being stressed or not and then taking the pecentage based from that. We use an adam optimizer in this case to optimize the learning of our model.

## Output

The model achieves 81.4% accuracy on the test set predicting whether students are stressed.

## File Structure

```
/save
ideal-university-life.ipynb
main.py
statistics.png
survey_responses.csv
visualize.py
```




## License

[MIT](https://choosealicense.com/licenses/mit/)

