# Score function templates for drift detection model generation

Users are expected to author custom score functions that needs to be supplied as a input while generating drift detection model using ibm-ai-openscale python client. This page has some templates of score functions that can be used for reference. 

### Input to score function:
  - **training_data_frame :** Dataframe of the training data

### Output from score function:
 - **predicted label (aka decoded-target) numpy array :**
    * The data type of the values in this array should be same as dataset class label data type 
    * **Example :** predicted label numpy array : [A,B,C,D,A,B,D,......]
    
 
 - **probability numpy array :** 
    * The number of entries in each element of this array should be same as the unique class labels of the dataset 
    * **Example:** probability numpy array: [[50,20,15,15] , [60,10,5,25], .......]

### Contents
- [WML Model Engine](#WML)
- [Azure Model Engine](#Azure)
   * [Azure Studio](#AzureStudio)
   * [Azure ML Service](#AzureMLService)
- [AWS SageMaker Model Engine](#AWS)
- [Custom Model Engine](#Custom)

## WML Model Engine: <a name="WML"></a>
This section provides the score function templates for model deployed in WML. There are 2 formats specified (local model , online model) and user is free to choose any of the formats . **The templates specified below are common for binary / muticlass classification cases**.
  - **Format-1:** Using the local mode (where in model store in WML is loaded and scored locally)
   ```
    WML_CREDENTAILS = {
             <EDIT THIS>
    }
    
    def score(training_data_frame):
        #Supply the model id
        model_id = <EDIT THIS>
        
        #Retain feature columns from user selection
        feature_columns = list(training_data_frame.columns)
        
        #Load the WML model in local object
        from watson_machine_learning_client import WatsonMachineLearningAPIClient
        from watson_machine_learning_client.wml_client_error import WMLClientError

        wml_client = WatsonMachineLearningAPIClient(WML_CREDENTAILS)
        try:
          model = wml_client.repository.load(model_id)
        except WMLClientError as err:
         raise Exception("{}. Please try using WML scoring_url snippet or train in watson openscale during drift configuration".format(str(err))) 
        
        #Predict the training data locally 
        #Example of a spark based model ( the below set of lines to be customized based on model framework)
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.master("local").appName("drift").getOrCreate()
        spark_frame = spark.createDataFrame(training_data_frame)
        spark_frame.printSchema()
        
        score_predictions = model.transform(spark_frame)
        score_preditions_pd = score_predictions.select("*").toPandas()

        probability_column_name = <EDIT THIS>
        prediction_column_name = <EDIT THIS>
        
        import numpy as np
        probability_array = np.array(
            [x.tolist() for x in score_preditions_pd[probability_column_name]])
        prediction_vector = np.array(
            [x for x in score_preditions_pd[prediction_column_name]])
        
        return probability_array, prediction_vector

```  
  *  _**Limitations for running in WML local mode:**_
        - If a model is generated and deployed using WML Model Builer - the local model does not work as WML python client does not support this context.

 - **Format-2:** Using scoring-url . This snippet uses the online scoring endpoint of a WML model using WML API client . **As this is online scoring , a cost is associated with the same .**
  ```
      WML_CREDENTAILS = {
             <EDIT THIS>
    }

    def score(training_data_frame):
        #To be filled by the user
        scoring_url = <EDIT_THIS>
        
        #The data type of the label column and prediction column should be same .
        #User needs to make sure that label column and prediction column array should have the same unique class labels
        prediction_column_name = <EDIT_THIS>
        probability_column_name = <EDIT_THIS>
        
        feature_columns = list(training_data_frame.columns)
        training_data_rows = training_data_frame[feature_columns].values.tolist()
        
        payload_scoring = {
            "fields": feature_columns,
            "values": [x for x in training_data_rows]
        }
        
        probability_array = None
        prediction_vector = None
    
        from watson_machine_learning_client import WatsonMachineLearningAPIClient
        wml_client = WatsonMachineLearningAPIClient(WML_CREDENTAILS)
        score_predictions = wml_client.deployments.score(scoring_url, payload_scoring)
        
        
        prob_col_index = list(score_predictions.get('fields')).index(probability_column_name)
        predict_col_index = list(score_predictions.get('fields')).index(prediction_column_name)
        
        if prob_col_index < 0 or predict_col_index < 0:
            raise Exception("Missing prediction/probability column in the scoring response")
            
        import numpy as np
        probability_array = np.array([value[prob_col_index] for value in score_predictions.get('values')])
        prediction_vector = np.array([value[predict_col_index] for value in score_predictions.get('values')])
        
        return probability_array, prediction_vector
 ```
## Azure Model Engine: <a name="Azure"></a>
### Azure Studio: <a name="AzureStudio"></a>
This section provides the score function templates for model deployed in Azure Model Engine. User needs to consider that online scoring endpoints of Azure Studio will be used. **As this is online scoring, a cost is associated with the same .**

- **Case-1: Binary Classifier**
 ```
   def score(training_data_frame):
    azure_scoring_url = <REQUEST RESPONSE URL FROM AZURE MODEL>
    token = <PRIMARY_KEY FROM AZURE MODEL>

    input_values = training_data_frame.values.tolist()
    feature_cols = list(training_data_frame.columns)
    input1 = [{field: value  for field,value in zip(feature_cols, input_value)} for input_value in input_values]

    #Payload
    import requests
    from datetime import datetime, timedelta

    payload = {
      "Inputs": {
        "input1": input1
      },
      "GlobalParameters": {}
    }

    headers = {'Authorization': 'Bearer ' + token}
    start = datetime.utcnow()
    response = requests.post(azure_scoring_url, json=payload, headers=headers)

    #Extract results part
    results = response.json()["Results"]["output1"]

    #Get Score label from first entry
    score_label = results[0]["Scored Labels"]
    print(score_label)

    score_prob_1 = float(results[0]["Scored Probabilities"])
    main_label = True
    if score_prob_1 < 0.5:
        #The score label is not main label of interest
        main_label = False

    output = [[value["Scored Labels"], 1 - float(value['Scored Probabilities']) if (value["Scored Labels"] == score_label and  not main_label) else float(value['Scored Probabilities'])] for value in results]
    print(len(output))

    import numpy as np
    #Construct predicted_label array
    predicted_vector = np.array([value[0] for value in output])

    #Construct probabilites array
    probability_array = np.array([[value[1],(1-value[1])] for value in output])

    return probability_array, predicted_vector
 ```

- **Case-2: Multiclass Classifier**
```
def score(training_data_frame):
    azure_scoring_url = <REQUEST RESPONSE URL FROM AZURE MODEL>
    token = <PRIMARY_KEY FROM AZURE MODEL>

    input_values = training_data_frame.values.tolist()
    feature_columns = list(training_data_frame.columns)
    input1 = [{field: value  for field,value in zip(feature_columns, input_value)} for input_value in input_values]

    #Payload
    import requests
    from datetime import datetime, timedelta

    payload = {
      "Inputs": {
        "input1": input1
      },
      "GlobalParameters": {}
    }

    headers = {'Authorization': 'Bearer ' + token}
    start = datetime.utcnow()
    response = requests.post(azure_scoring_url, json=payload, headers=headers)
    response_time = (datetime.utcnow() - start).total_seconds() * 1000
    print(response_time)

    #Extract results
    results = response.json()["Results"]["output1"]

    # Compute for all values
    score_label_list = []
    score_prob_list = []

    for value in results:
        score_label_list.append(str(value["Scored Labels"]))

        #Construct prob
        score_prob_values = [float(prob) for key,prob in value.items() if key.startswith('Scored Probabilities for Class',0)]
        score_prob_list.append(score_prob_values)

    import numpy as np
    #Construct predicted_label bucket
    predicted_vector = np.array(score_label_list)

    #Scored probabilities
    probability_array = np.array(score_prob_list)

    return probability_array, predicted_vector
```

### Azure ML Service: <a name="AzureMLService"></a>
This section provides the score function templates for model deployed in Azure ML Service. User needs to consider that online scoring endpoints of Azure ML Service will be used. The below snippet is valid for both multiclass and binary classfication model. **As this is online scoring, a cost is associated with the same .**

- **Format-1: Online Mode**
 ```
   def score(training_data_frame):
    az_scoring_uri = <EDIT THIS>
    api_key = <DEPLOYMENT API KEY>
    
    headers = {'Content-Type':'application/json',  'Authorization':('Bearer '+ api_key)} 
    
    input_values = training_data_frame.values.tolist()
    feature_cols = list(training_data_frame.columns)
    scoring_data = [{field: value  for field,value in zip(feature_cols, input_value)} for input_value in input_values]
    
    payload = {
     "input": scoring_data
    }
    
    import requests
    import json
    import numpy as np
    import time
    
    start_time = time.time()  
    response = requests.post(az_scoring_uri, json=payload, headers=headers)
    response_time = int((time.time() - start_time)*1000)
    
    response_dict = json.loads(response.json())
    results = response_dict['output']
    prediction_vector = np.array([x["Scored Labels"] for x in results])
    probability_array = np.array([x["Scored Probabilities"] for x in results])
    return probability_array, prediction_vector
 ```

## AWS SageMaker Model Engine: <a name="AWS"></a>
This section provides the score function templates for for model deployed in SageMaker Model Engine. User needs to consider that online scoring endpoints of SageMaker will be used.**As this is online scoring, a cost is associated with the same .** Please note that we are the below snipets are created with a assumption that input datasets for AWS are **one hot encoded for categorical columns** and label-encoded for **label column**

- **Case-1: Binary Classifier**
 ```
 SAGEMAKER_CREDENTIALS = {
    "access_key_id": <EDIT THIS>,
    "secret_access_key": <EDIT THIS>,
    "region": <EDIT THIS>
}

def score(training_data_frame):
    #User input needed
    endpoint_name = <EDIT THIS>

    access_id = SAGEMAKER_CREDENTIALS.get('access_key_id')
    secret_key = SAGEMAKER_CREDENTIALS.get('secret_access_key')
    region = SAGEMAKER_CREDENTIALS.get('region')

    #Covert the training data frames to bytes
    import io
    import numpy as np
    train_df_bytes = io.BytesIO()
    np.savetxt(train_df_bytes, training_data_frame.values, delimiter=',', fmt='%g')
    payload_data = train_df_bytes.getvalue().decode().rstrip()

    #Score the training data
    import requests
    import time
    import json
    import boto3

    runtime = boto3.client('sagemaker-runtime', region_name=region, aws_access_key_id=access_id, aws_secret_access_key=secret_key)
    start_time = time.time()

    response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='text/csv', Body=payload_data)
    response_time = int((time.time() - start_time)*1000)
    results_decoded = json.loads(response['Body'].read().decode())

    #Extract the details
    results = results_decoded['predictions']

    #Find the winning label
    predicted_label_0 = results[0]['predicted_label']
    prob_0 = results[0]['score']

    is_winning_label = True
    if prob_0 < 0.5:
        is_winning_label = False


    output = [[value['predicted_label'], 1 - float(value['score']) if (value["predicted_label"] == predicted_label_0 and  not is_winning_label) else float(value['score'])] for value in results]

    import numpy as np
    predicted_vector = np.array([value[0] for value in output])
    probability_array = np.array([[value[1],1-value[1]] for value in output])

    return probability_array, predicted_vector
```

- **Case-2: Multiclass Classifier**
```
SAGEMAKER_CREDENTIALS = {
    "access_key_id": <EDIT THIS>,
    "secret_access_key": <EDIT THIS>,
    "region": <EDIT THIS>
}

def score(training_data_frame):
    #User input needed
    endpoint_name = <EDIT THIS>

    access_id = SAGEMAKER_CREDENTIALS.get('access_key_id')
    secret_key = SAGEMAKER_CREDENTIALS.get('secret_access_key')
    region = SAGEMAKER_CREDENTIALS.get('region')
    
    #Covert the training data frames to bytes
    import io
    import numpy as np
    train_df_bytes = io.BytesIO()
    np.savetxt(train_df_bytes, training_data_frame.values, delimiter=',', fmt='%g')
    payload_data = train_df_bytes.getvalue().decode().rstrip()

    #Score the training data
    import requests
    import time
    import json
    import boto3

    runtime = boto3.client('sagemaker-runtime', region_name=region, aws_access_key_id=access_id, aws_secret_access_key=secret_key)
    start_time = time.time()

    response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='text/csv', Body=payload_data)
    response_time = int((time.time() - start_time)*1000)
    results_decoded = json.loads(response['Body'].read().decode())

    #Extract the details
    results = results_decoded['predictions']

    predicted_vector_list = []
    probability_array_list = []

    for value in results:
        predicted_vector_list.append(value['predicted_label'])
        probability_array_list.append(value['score'])

    #Conver to numpy arrays
    probability_array = np.array(probability_array_list)
    predicted_vector = np.array(predicted_vector_list)

    return probability_array, predicted_vector
```

## Custom Model Engine: <a name="Custom"></a>
This section provides the score function template for model deployed in a custom engine. The online scoring end point of custom engine will be used for scoring. The below snippets holds good for binary/multiclass. **As this is online scoring, a cost is associated with the same .**

 ```
    CUSTOM_ENGINE_CREDENTIALS = {
    "url": <EDIT THIS>,
    "username": <EDIT THIS>,
    "password": <EDIT THIS>
}

def score(training_data_frame):
    #To be filled by the user - model scoring url
    scoring_url = <EDIT THIS>

    #The data type of the label column and prediction column should be same .
    #User needs to make sure that label column and prediction column array should have the same unique class labels
    prediction_column_name = <EDIT THIS>
    probability_column_name = <EDIT THIS>

    feature_columns = list(training_data_frame.columns)
    training_data_rows = training_data_frame[feature_columns].values.tolist()

    payload_scoring = {
        "fields": feature_columns,
        "values": [x for x in training_data_rows]
    }

    #Retain username and password for custom
    username = CUSTOM_ENGINE_CREDENTIALS.get("username")
    password =  CUSTOM_ENGINE_CREDENTIALS.get("password")

    import requests
    import time

    start_time = time.time()
    response = requests.post(scoring_url, json=payload_scoring, auth=(username, password))
    response_time = int((time.time() - start_time)*1000)

    #Convert response to dict
    import json
    score_predictions = json.loads(response.text)

    prob_col_index = list(score_predictions.get("fields")).index(probability_column_name)
    predict_col_index = list(score_predictions.get("fields")).index(prediction_column_name)

    if prob_col_index < 0 or predict_col_index < 0:
        raise Exception("Missing prediction/probability column in the scoring response")

    probability_array = None
    prediction_vector = None

    import numpy as np
    probability_array = np.array([value[prob_col_index] for value in score_predictions.get('values')])
    prediction_vector = np.array([value[predict_col_index] for value in score_predictions.get('values')])

    return probability_array,prediction_vector
 ```
