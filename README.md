
# SefAgent

llm agent which can learn human characterisitcs from user conversation. The key idea is to human personality traits into 13 types, such as openness, conscientiousness, honestness, etc. Based on the converstaion between users and llm agent, it can a vector representation to simulate human personality while responding back to users. 


A Python implementation which can help agent itself to have human personality traits. It has two parts:
(1) train a multi-classifier which can classification conversations or sententences into human personality traits, 
    -- prepare training data, (text, label) pairs
    -- run 
        ```python
        python train_behaviorclassifier.py
        ```

(2) run llmserver.py to learn human perosnality from conversation
    -- they are personalized_characteristics and global_characteristics, where global_characteristics feature the agent personality, while personalized_characteristics will be the reciprocal trait mapping for user behavior
    -- run server as below
        ```python
        python llmserver.py
        ```

## Usage

### Human personality traits classifier

Here's how you'd instantiate BehaviorClassifier:

```python
#. Load Tokenizer and Model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    NUM_CLASSES = len(df['label'].unique())  #Dynamically determine the number of classes
    model = BehaviorClassifier(MODEL_NAME, NUM_CLASSES+1)
    print(f"Model '{MODEL_NAME}' loaded successfully.")

except Exception as e:
    print(f"Error loading model: {e}")
    exit()

```

### Start LLM server
And here's how you'd start llm server with user conversation:

```python
python llmserver.py
```

### Prompt test
```python
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "hello"}' http://localhost:5000/generate
```



### Example Data

behavior_data.csv: an example to format the training data for human behavior classifier



### Samples

user message = "what are the top 10 customers by sales"
![result slide](output.pptx)


### Library Dependences

huggingface transformer
openai client api


### License

MIT
