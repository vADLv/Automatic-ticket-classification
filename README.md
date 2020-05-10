# AIR project
## Automatic ticket service classification in the IT department

There is a ticket system in the IT Department at Innopolis University. Every month IT Dep. has to deal with 500+ various tickets from diverse users on both English and Russian languages. For every ticket, the IT administrator should define related service, type of ticket and owner person by hand.

![Alt Text](img/my.gif)

The project’s purpose is to automate the definition of related service, decrease reaction time and increase IT-administrators’ efficiency.
> "No ticket - no service"

This can be done,e.g by using text classification with neural networks taking into account a large set of previously collected data (7000+ raw messages).

The project consist of following steps:
  - Crawler
  - Preprocessing
  - Classifier training
  - Ticket updater script
  - Testing


### Tech

Project's technology stack':

* Python (Keras, NLTK, sklearn).
* SQL.
* [OTRS](https://it.university.innopolis.ru/otrs/customer.pl) - Open-Source Ticket Request System (Version 6).
* [PyOTRS](https://buildmedia.readthedocs.org/media/pdf/pyotrs/latest/pyotrs.pdf) - Python wrapper for accessing OTRS using the (GenericInterface) REST API.


## Using
### Prepare
In order to predict the service we need to train classifier in advance. In the [colab python notebook](https://colab.research.google.com/drive/1ddkqWp1YHoxTFaNLUA4KfNTPVUlTP8ZK#scrollTo=FVW_94T42zWj) you can see data *crawling*, *preprocessing* and model *training*. From the colab we get saved classifier: **lstm_model.json** and model weights **lstm_model.h5** files with **tokenizer.pickle**. Also we need to provide the same preprocessing steps for the new ticket. Then we'll be able to predict ticket service. There are different data preprocessing and visualizing approaches was done in the Colab notebook as well as different classifier model was used (RandomForest, Simple NN, CNN, LSTM).

You are able to train your own classifier on the raw data. To get a raw dataset with all ticket message you can use **main_query.txt** sql request in the SQL-box panel of OTRS web interface. The next step is OTRS web service importing from the **GenericTicketConnectorREST.yml** file.

### Run
Then just put your credentials into **credentials.json** file and run the **OTRSServiceUpdate.py** script. The script file is well commented, you can specify such configs as ticket updating time, path, etc ...
Script blocks devided by following steps:
- model loading
- new ticket without service retrieving
- ticket preprocessing and service prediction
- ticket update
- repeat steps 2-5 after some time
You can see the information about update and predict events in the terminal window or just check OTRS ticket panel.

### Enjoy!!!


<v.palenov@innopolis.ru>
may, 2020
