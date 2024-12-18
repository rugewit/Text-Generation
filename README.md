## Генерация текста



Необходимо натренировать и сравнить качество нескольких генеративных текстовых моделей на одном из заданных текстовых датасетов.

Необходимо исследовать следующие нейросетевые архитектуры:

1. Simple RNN с посимвольной и по-словной токенизацией
2. Однонаправленная однослойная и многослойная LSTM c посимвольной токенизацией и токенизацией по словам и [на основе BPE](https://keras.io/api/keras_nlp/tokenizers/byte_pair_tokenizer/)
3. Двунаправленная LSTM

Датасет - Английская литература с сайта [Project Gutenberg](https://www.gutenberg.org/)



Выведем первые 1000 символов из датасета (Библия)
``````
The Project Gutenberg eBook of The King James Version of the Bible
    
This ebook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this ebook or online
at www.gutenberg.org. If you are not located in the United States,
you will have to check the laws of the country where you are located
before using this eBook.

Title: The King James Version of the Bible

Release date: August 1, 1989 [eBook #10]
                Most recently updated: April 6, 2024

Language: English



*** START OF THE PROJECT GUTENBERG EBOOK THE KING JAMES VERSION OF THE BIBLE ***
The Old Testament of the King James Version of the Bible
The First Book of Moses: Called Genesis
The Second Book of Moses: Called Exodus
The Third Book of Moses: Called Leviticus
The Fourth Book of Moses: Called Number
``````

Функции для преобразования char в индексы
```python
unique_chars = sorted(set(file_text))

chars_to_ids = tf.keras.layers.StringLookup(vocabulary = list(unique_chars), mask_token = None)
ids_to_chars = tf.keras.layers.StringLookup(vocabulary = chars_to_ids.get_vocabulary(), invert = True, mask_token = None)

def ids_to_text(ids):
  return tf.strings.reduce_join(ids_to_chars(ids), axis=-1)
```

Преобразуем датасет в пару x-y

```python
def split_into_x_y(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text
```

Создание подготовленного датасета для обучения

```python
BATCH_SIZE = 256

BUFFER_SIZE = 10000

dataset = (
    dataset_xy
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)) 
```

Параметры

```python
vocab_size = len(chars_to_ids.get_vocabulary())

embedding_dim = 64

rnn_units = 64
```

#### Simple RNN модель

```python
def build_chars_simple_rnn_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=(batch_size, None)),        
    tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, stateful=True),
    tf.keras.layers.Dense(vocab_size)
    ])

    return model
```

Результат

```
Epoch 1/10
172/172 [==============================] - 13s 61ms/step - loss: 3.0049
Epoch 2/10
172/172 [==============================] - 11s 56ms/step - loss: 2.1649
Epoch 3/10
172/172 [==============================] - 11s 57ms/step - loss: 1.9663
Epoch 4/10
172/172 [==============================] - 10s 55ms/step - loss: 1.8602
Epoch 5/10
172/172 [==============================] - 10s 54ms/step - loss: 1.7853
Epoch 6/10
172/172 [==============================] - 11s 59ms/step - loss: 1.7299
Epoch 7/10
172/172 [==============================] - 10s 54ms/step - loss: 1.6883
Epoch 8/10
172/172 [==============================] - 11s 57ms/step - loss: 1.6558
Epoch 9/10
172/172 [==============================] - 10s 54ms/step - loss: 1.6292
Epoch 10/10
172/172 [==============================] - 10s 54ms/step - loss: 1.6087
```

![](https://github.com/rugewit/Text-Generation/blob/main/report_images/1.png)

Пример сгенерированного текста

```
Heaven his and deenses of Philis saiveles of the your bomeitenglact there of that him.

11:8 But the staber of hath that secve, tnom the Goked anay that he wan of the given that orer thit shall with eviled, but the broldin lish the ry.

4:18 Septind:
and dierd a goth therest theth which thou God, and down it thou the tigaine, ye lot, jedurding fearl, and the phildres, grod, for the LORD peary every aboow, and thy man shall lovesteme, and hand of the plethes, and a set me the gon with the man whic
```

#### LSTM модель

```python
def build_chars_lstm_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=(batch_size, None)),        
    tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True),
    tf.keras.layers.Dense(vocab_size)
    ])

    return model
```



Подготовка датасета и параметров обучения

```python
lstm_batch = 64

BUFFER_SIZE = 10000

dataset_lstm = (
    dataset_xy
    .shuffle(BUFFER_SIZE)
    .batch(lstm_batch, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)) 

lstm_embedding_size = 256
lstm_rnn_units = 512
```

Результат
```
Epoch 1/10
688/688 [==============================] - 11s 13ms/step - loss: 1.8857
Epoch 2/10
688/688 [==============================] - 9s 11ms/step - loss: 1.3277
Epoch 3/10
688/688 [==============================] - 8s 11ms/step - loss: 1.2030
Epoch 4/10
688/688 [==============================] - 8s 10ms/step - loss: 1.1429
Epoch 5/10
688/688 [==============================] - 9s 11ms/step - loss: 1.1063
Epoch 6/10
688/688 [==============================] - 8s 10ms/step - loss: 1.0797
Epoch 7/10
688/688 [==============================] - 8s 10ms/step - loss: 1.0590
Epoch 8/10
688/688 [==============================] - 8s 10ms/step - loss: 1.0423
Epoch 9/10
688/688 [==============================] - 8s 11ms/step - loss: 1.0286
Epoch 10/10
688/688 [==============================] - 8s 11ms/step - loss: 1.0170
```

![](https://github.com/rugewit/Text-Generation/blob/main/report_images/2.png)

Пример сгенерированного текста

```
Heaveng the river of
Babylon: and if a man do we laid up to the innorth the
power of all nd the fathers hald been collemies or a falter with
strength winds; and she was perpetual and grace, and
he was the shield, he shall not reserve tranks in the blood,
which he rewarded me.

13:7 The LORD is the Egyptians, who is alones.

109:30 He breaketh bardself, and a fiery furnace of blaspheme, and priests and
unrighteousness: who then saith, I was made in the midst of the nations.

1:17 For whosoe
```
