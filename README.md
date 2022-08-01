
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
    # "Q5-What are your levels of stress ?": 1,
    # "Q6-How Satisfied You are with your Student Life ?": 1,
    "Q7-How much effort do you make to interact with others ?": 3.0,
    "Q8-About How events are you aware about ?": 3.0,
    # "Q9-What is an ideal student life ?": 1,
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = model.predict(input_dict)

prob = tf.nn.sigmoid(predictions[0])

print("probability you are satisfied with your student life: ", prob)
```


## How it works

### Preprocessing
First we generate the vocab in which is used in the scripts. We then map it to a StringLookup layer in order to map each character into a number. We also create ids_from_chars in order to revert back to text near the end. We do this because it is easier for the model to process the numbers rather than individual characters.

Next we configure the dataset. We split the script into a number of sequences that we can use to train the model. These are split with split_input_sequence into a train side and a test side for each sequence. We then batch and shuffle the dataset to prepare for training.


### Model

The model we train on the dataset consists of 3 layers. We use an Embedding layer, a GRU layer, and a dense layer. The embedding layer first maps the input into weights that we can modify during training. The next layer is a GRU or Gated Recurrent Unit is a type of RNN similar to a LSTM but with only 2 gates. It works by determining what information is important and what can be forgotten within the text. Finally there is a dense layer to select an id within the vocab set. There is one logit for each character in the vocab. We then can map this id back to a string.

We use a sparse categorical crossentropy loss because we are labeling the logits we recieve from the dataset. We use sparse categorical crossentropy instead of categorical crossentropy because there is a signifigant amount of logits and not enough relation between them. Finally the adam optimizer is used to optimize our model.


### Text Generation

Finally after we have trained our model we aim to generate a script. To acomplish this we take a starting value of `MATT:` and plug it into our model to predict one letter at a time and then to continue to predict on the new text with the data. This process is repeated many times to generate a full script. First we define a mask to catch any faulty text from being generated. Then we run the convert the input into tensors by mapping it through our StringLookup. We then run the input through the models layers and make sure to save the state. We then repeat this process enough times to generate a full script.


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

