# Keras GPT Copilot: Integrating the Power of Large Language Models in Deep Learning Model Development
<img src="https://github.com/fabprezja/keras-gpt-copilot/assets/87379098/bb66f7fb-fb76-4ac4-acd5-4d06b486ecee.png" alt="logo" width="55%">

Keras GPT Copilot is the first Python package designed to integrate an LLM copilot within the model development workflow, offering iterative feedback options for enhancing the performance of your Keras deep learning models. Utilizing the power of OpenAI's GPT models, Keras GPT Copilot can use any of the compatible models (GPT4 is recommended). However, the prompt-only mode allows for compatibility with other large language models.

> **Note:** The Keras GPT Copilot was showcased at the Google Startup Campus in Seoul, South Korea, as part of the Keras Community Day! Thank you! [Event details here](https://festa.io/events/4000).

## Features

- Generates copilot feedback from gathering model configuration, optimizer details, and experiment results during model development
- Interacts with OpenAI's LLMs, such as GPT-4
- Can be used with non-OpenAI LLMs to generate suggestions
- Offers options to downsample and/or smoothen validation curves to accommodate large (and/or noisy) results within the copilot prompt
- Provides flexibility in customizing the copilot prompt, allowing for the addition of extra information.
- Supports follow-up questions for extended guidance, such as requesting specific code changes based on previous recommendations

## Installation

To install the kerasGPTcopilot library, run:
```
pip install keras-gpt-copilot
```
- Note 1: This is an experimental package; I aim to expand capabilities over time.
- Note 2: GPT4 or similarly capable models tend to perform best.

## Usage
... with Open A.I API
```python
from tensorflow.keras.models import Model
from kerasGPTcopilot.core import ModelAdvisor

model = Model(...)  # Your Keras model
history = model.fit(...)  # The Keras History object
api_key = "your_openai_api_key"

advisor = ModelAdvisor(model, history, api_key)
suggestion = advisor.get_suggestions(model="gpt-4", print_cp_prompt=True, print_cp_response=True)
```

... with other LLMs (copilot prompt only)
 ```python
from tensorflow.keras.models import Model
from kerasGPTcopilot.core import ModelAdvisor

model = Model(...)  # Your Keras model
history = model.fit(...)  # The Keras History object

advisor = ModelAdvisor(model, history, None)  # No API key
copilot_prompt = advisor.get_suggestions(cp_prompt_only=True, print_cp_prompt=True)
```
... follow-up for implementing suggestions
```python
follow_up_question = "Can you show me Python code for implementing the suggested changes?"
follow_up_suggestion = advisor.get_follow_up_suggestion(initial_suggestion=suggestion, 
                                                         follow_up_question=follow_up_question,
                                                         model="gpt-4")
```
## How to cite:

If you find the work usefull in your project, please cite:

```bibtex
@misc{fabprezja_2023,
  author = {Fabi Prezja},
  title = {Keras GPT Copilot: Integrating the Power of Large Language Models in Deep Learning Model Development},
  month = may,
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/fabprezja/keras-gpt-copilot}},
  doi = {10.5281/zenodo.7935183},
  url = {https://doi.org/10.5281/zenodo.7935183}
}
```
## Use Example
### Before Keras GPT Copilot (Intentionally Over-capacitated Model)
![image](https://github.com/fabprezja/keras-gpt-copilot/assets/87379098/0231df5b-2afd-4b5e-9c17-7fb1ed7a5881)

```
Test score at best epoch:
Test Accuracy: 0.9491999745368958
```
### After Keras GPT Copilot (1 Round)
![image](https://github.com/fabprezja/keras-gpt-copilot/assets/87379098/6ec28e14-7637-469f-88e0-7589bf31bbbd)
```
Test score at best epoch:
Test accuracy: 0.9810000061988831
```
Let's go through the code used to produce these results.

### Step 1: Intentionally Over-capacitated Model

 ```python

model = Sequential([
    Dense(360, activation='relu', input_shape=(625,)),
    Dense(525, activation='relu'),
    Dense(212, activation='relu'),
    Dense(1, activation='sigmoid')])

model.compile(optimizer=Adam(learning_rate=1e-5,weight_decay=1e-5), 
              loss='binary_crossentropy', metrics=['accuracy'])
              
# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=11, batch_size=128, callbacks=[model_checkpoint])
                    
```
### Step 2: Keras GPT Copilot

 ```python
from kerasGPTcopilot.core import ModelAdvisor
advisor = ModelAdvisor(model, history, None, test_loss=test_loss,test_metric=("test_accuracy", test_accuracy))
suggestion = advisor.get_suggestions(cp_prompt_only=True,print_cp_prompt=True)
```

### Step 2.1: Keras GPT Copilot Generated Prompt

```
Input prompt:
 I have a deep learning model with the following configuration:
Model Configuration:
{
  "name": "sequential_4",
  "layers": [
    {
      "class_name": "InputLayer",
      "config": {
        "batch_input_shape": [
          null,
          625
        ],
        "dtype": "float32",
        "sparse": false,
        "ragged": false,
        "name": "dense_16_input"
      }
    },
    {
      "class_name": "Dense",
      "config": {
        "name": "dense_16",
        "trainable": true,
        "dtype": "float32",
        "batch_input_shape": [
          null,
          625
        ],
        "units": 360,
        "activation": "relu",
        "use_bias": true,
        "kernel_initializer": {
          "class_name": "GlorotUniform",
          "config": {
            "seed": null
          }
        },
        "bias_initializer": {
          "class_name": "Zeros",
          "config": {}
        },
        "kernel_regularizer": null,
        "bias_regularizer": null,
        "activity_regularizer": null,
        "kernel_constraint": null,
        "bias_constraint": null
      }
    },
    {
      "class_name": "Dense",
      "config": {
        "name": "dense_17",
        "trainable": true,
        "dtype": "float32",
        "units": 525,
        "activation": "relu",
        "use_bias": true,
        "kernel_initializer": {
          "class_name": "GlorotUniform",
          "config": {
            "seed": null
          }
        },
        "bias_initializer": {
          "class_name": "Zeros",
          "config": {}
        },
        "kernel_regularizer": null,
        "bias_regularizer": null,
        "activity_regularizer": null,
        "kernel_constraint": null,
        "bias_constraint": null
      }
    },
    {
      "class_name": "Dense",
      "config": {
        "name": "dense_18",
        "trainable": true,
        "dtype": "float32",
        "units": 212,
        "activation": "relu",
        "use_bias": true,
        "kernel_initializer": {
          "class_name": "GlorotUniform",
          "config": {
            "seed": null
          }
        },
        "bias_initializer": {
          "class_name": "Zeros",
          "config": {}
        },
        "kernel_regularizer": null,
        "bias_regularizer": null,
        "activity_regularizer": null,
        "kernel_constraint": null,
        "bias_constraint": null
      }
    },
    {
      "class_name": "Dense",
      "config": {
        "name": "dense_19",
        "trainable": true,
        "dtype": "float32",
        "units": 1,
        "activation": "sigmoid",
        "use_bias": true,
        "kernel_initializer": {
          "class_name": "GlorotUniform",
          "config": {
            "seed": null
          }
        },
        "bias_initializer": {
          "class_name": "Zeros",
          "config": {}
        },
        "kernel_regularizer": null,
        "bias_regularizer": null,
        "activity_regularizer": null,
        "kernel_constraint": null,
        "bias_constraint": null
      }
    }
  ]
}
Optimizer Configuration:
{
  "name": "Adam",
  "weight_decay": 1e-05,
  "clipnorm": null,
  "global_clipnorm": null,
  "clipvalue": null,
  "use_ema": false,
  "ema_momentum": 0.99,
  "ema_overwrite_frequency": null,
  "jit_compile": false,
  "is_legacy_optimizer": false,
  "learning_rate": 9.999999747378752e-05,
  "beta_1": 0.9,
  "beta_2": 0.999,
  "epsilon": 1e-07,
  "amsgrad": false
}
The validation curves over epochs are:
{
  "loss": [
    0.30621,
    0.16135,
    0.12647,
    0.10301,
    0.08611,
    0.07187,
    0.05993,
    0.04941,
    0.03879,
    0.02993,
    0.02391
  ],
  "accuracy": [
    0.8664,
    0.941,
    0.95653,
    0.96793,
    0.97527,
    0.98073,
    0.98553,
    0.9892,
    0.99287,
    0.9956,
    0.99667
  ],
  "val_loss": [
    0.19374,
    0.16817,
    0.16604,
    0.15831,
    0.15778,
    0.15312,
    0.15342,
    0.15701,
    0.15996,
    0.1623,
    0.15952
  ],
  "val_accuracy": [
    0.9254,
    0.9344,
    0.9382,
    0.9386,
    0.9428,
    0.9434,
    0.946,
    0.946,
    0.944,
    0.9452,
    0.9484
  ]
}
The test loss is 0.1487419605255127 and the test_accuracy is 0.9491999745368958.
How can I improve the model performance based on this information?
```
### Step 3: GPT4 Response (Suggestions):
```
Based on the information provided, it seems that your model is already performing quite well on the training dataset. However, the performance on the validation dataset does not improve significantly after a certain point, and there is a gap between training and validation accuracy. This suggests that the model might be overfitting the training data. Here are some suggestions to improve the model performance:

1. Add dropout layers: Dropout is a regularization technique that helps to prevent overfitting. You can add dropout layers after each dense layer in your model. Start with a dropout rate of 0.5 and experiment with different rates to find the best value.

2. Early stopping: You can implement early stopping to halt the training process when the validation loss stops improving. This will prevent the model from overfitting the training data.

3. Experiment with the architecture: You can try different configurations for the number of layers and the number of units in each layer. You may also experiment with other activation functions, like LeakyReLU or ELU, instead of ReLU.

4. Try different optimizers: You can try different optimizers like RMSprop, SGD with momentum, or AdaGrad, and see if they yield better results.

5. Adjust the learning rate: Experiment with different learning rates for your optimizer. You can also use a learning rate scheduler to reduce the learning rate during training, which can help in converging to better optima.

6. Use data augmentation: If your dataset is small or not diverse enough, you can apply data augmentation techniques to generate more varied training samples. This will help your model generalize better to unseen data.
```
### Step 3.1: Copilot Follow-up:
```
"Can you show me Python code for implementing the suggested changes?"
```
### Step 4: GPT4 Follow-up Response (Code Implemented Suggestions):
```
Sure! Here's an example of how you can implement the suggested changes using TensorFlow and Keras. I'll demonstrate adding dropout layers, early stopping, adjusting the learning rate, and adding L2 weight regularization:

```Python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# Model configuration
input_shape = (625,)
n_units_1 = 360
n_units_2 = 525
n_units_3 = 212
dropout_rate = 0.5
l2_reg = 1e-4

# Create the model
model = models.Sequential()
model.add(layers.InputLayer(input_shape=input_shape))
model.add(layers.Dense(n_units_1, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(n_units_2, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(n_units_3, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(1, activation='sigmoid'))

# Optimizer configuration
learning_rate = 1e-4
optimizer = optimizers.Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
```

# Documentation & Notes
In this section, we will see the documentation and related notes; Documentation will migrate from here as the library grows.

## ModelAdvisor Class
The ModelAdvisor class has several optional parameters to customize the copilot prompt:

- `model`: (Model) A Keras model object.
- `history`: (History) A Keras History object.
- `api_key`: (str) The API key from OpenAI.
- `model_summary`: (bool, optional) Whether to include the model summary in the copilot prompt. Defaults to False.
   > _**Note:** This argument is intended for larger models that are difficult to include without exceeding token limits with model_config (and require to set it to False).
- `model_config`: (bool, optional) Whether to include the model configuration in the copilot prompt. Defaults to True.
- `optimizer_info`: (bool, optional) Whether to include the optimizer information in the copilot prompt. Defaults to True.
- `add_info`: (str, optional) Additional information to include in the copilot prompt. Defaults to None.
   > _**Note:** This field can provide additional information, it is generally adviced to include some sort of additional details.
- `test_loss`: (float, optional) Test loss value to include in the copilot prompt. Defaults to None.
   > _**Note:** usually points to the keras test result.
- `test_metric`: (Union[float, Tuple[str, float]], optional) Test metric name and value to include in the copilot prompt. Defaults to None.
   > _**Note:** e.g. test_metric=("test_accuracy", test_accuracy)
- `un_trainable`: (bool, optional) Whether to include untrainable layer names in the copilot prompt. Defaults to False.
- `window_size`: (int, optional) The window size for smoothing the validation curves. Defaults to None.
   > _**Note:** This argument is intended for visualizing extremely noisy curves; generally it is not recommended. A moving average is used for smoothing. The window_size determines the number of data points (in this case, epochs) to be considered in the moving average. The larger the window size, the smoother the resulting curve will be. If the window_size is set to None, no smoothing will be applied to the validation curves.
- `downsample_factor`: (int, optional) The downsampling factor for the validation curves. Defaults to None.
   > _**Note:** This argument is intended for a large number of epochs that may exceed token limits. If downsample_factor is set to 2, every second data point will be selected, effectively reducing the number of data points in the validation curve by half.
- `suffix_text`: (str, optional) Additional text to include at the end of the copilot prompt. Defaults to None.
- `round_vc`: (int, optional) The number of decimal places to round the validation curves values. Defaults to 5.

## ModelAdvisor.get_suggestions method
The .get_sugestions method has the following arguments:

- `print_cp_response`: (bool, optional) Whether to print the copilot suggestion to the console. Defaults to True.
- `print_cp_prompt`: (bool, optional) Whether to print the copilot prompt to the console. Defaults to True.
- `model`: (str, optional) The name of the OpenAI model to use. Defaults to "gpt-4".
	> _**Note:** A full list of models: https://platform.openai.com/docs/models/overview ; Remember, that gpt-4 is in limited beta: https://platform.openai.com/docs/models/gpt-4;
- `temperature`: (float, optional) The temperature to control the randomness of the copilot suggestion. Defaults to 0.5.
- `max_tokens`: (int, optional) The maximum number of tokens in the copilot suggestion. Defaults to 400.
- `save_to_file`: (str, optional) The path to save the copilot prompt and suggestion to a file. Defaults to None.
- `copy_to_clipboard`: (bool, optional) Copy the copilot prompt to the clipboard. Defaults to True.
	> _**Note:**  The copilot prompt is automatically copied to the clipboard.
- `cp_prompt_only`: (bool, optional) Return only the copilot prompt without API suggestions. Defaults to False.
	> _**Note:** This argument can be used to transfer the copilot prompt into another LLM. Generally, GPT4-like LLMs, tend to perform well.

## ModelAdvisor.get_follow_up_suggestion method
The .get_follow_up_suggestion method has the following arguments:

- `initial_suggestion`: (str) The initial copilot suggestion.
- `follow_up_question`: (str) The follow-up copilot question.
- `model`: (str) The name of the OpenAI model to use. Defaults to "gpt-4".
- `temperature`: (float, optional) The temperature to control the randomness of the copilot follow-up suggestion. Defaults to 0.5.
- `max_tokens`: (int, optional) The maximum number of tokens in the copilot follow-up suggestion. Defaults to 600.
- `print_cp_response`: (bool, optional) Print the copilot follow-up suggestion to the console. Defaults to True.
