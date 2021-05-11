## Gender analysis of Master Theses from GU and Chalmers
This project aims to analyse differences in writing style and language between male and female authors in an academic setting.
The gender distribution in the departments and faculties is typically skewed, and the topic of the thesis naturally influences the wording.
Because of this, a model trained to predict gender might pick up on the more accessible topic differences rather than the subtle differences 
in writing style between genders. We explore what features are relevant for predicting author gender, and different ways of dealing with topic bias. 

Each notebook contains some documentation and short analyses of the results. 

### Description of code and other files

```Exjobb_downsample.ipynb``` - Code for restricting the data to include theses from four main topics, and downsampling the data set by randomly removing theses whose author gender is in majority in the department of the thesis.
The result is a set of master theses with an equal number of male and female authors for each included department, limited to the following departments and main topics:

* **humaniora (humanities)**  - Institutionen för filosofi, lingvistik och vetenskapsteori, Institutionen för litteratur, idéhistoria och religion, Institutionen för språk och litteraturer, Institutionen för svenska språket
* **arkitektur (architecture)**  - Institutionen för arkitektur och samhällsbyggnadsteknik
* **ekonomi (economy)** - Företagsekonomiska institutionen
* **naturvetenskap (natural science)** - Institutionen för biologi och bioteknik, Institutionen för fysik, Institutionen för kemi och kemiteknik, Institutionen för rymd- och geovetenskap

English and Swedish theses are downsampled separately, and the resulting datasets are saved to ```data/df_english_downsampled.csv``` and ```data/df_swedish_downsampled.csv```.
Reads from the meta data files ```/data/Examensarbeten/GU-metadata-210310.json``` and ```/data/Examensarbeten/GU-metadata-210310.json```.

\
```compute_frequencies.py``` -  This script precomputes the frequencies of part-of-speach ngrams and function words of a given set of master theses.
It produces a csv-file with one row for each document, where the first column contains the identifier of the document, and the remaining columns represent the frequencies of all pos-tag unigrams, the 100 most common pos-tag bigrams, the 500 most common pos-tag trigrams, and the 200 most common tokens (function words) and pronouns. The following variables need to be specified (in the code):
-    ```included_documents``` specifies the csv-file that lists the master theses to include (for example the csv-files produced by Exjobb_downsample.ipynb), with the required column "id" with the identifiers of the theses.
-    ```file_path``` specifies where to find the files that contain the pos-tagged texts (available at ```/data/Examensarbeten/sparv/exjobb-en/export/csv```).
-    ```tag_kind``` specifies the tag for which to compute the frequencies (default is 'msd'. Change this to 'pos' for the English theses).
-    ```frequency_file``` specifies where to save the computed frequencies (in .csv format).
 
 The script produces a csv-file with one row for each document, where the first column
 contains the identifier of the document, and the remaining columns represent the frequencies of all pos-tag unigrams, 
 the 100 most common pos-tag bigrams, the 500 most common pos-tag trigrams, and the 200 most common tokens (function words) and pronouns.
 
 \
```preprocess_texts.py``` - Code for tokenizing and cleaning text documents.

\
```preprocess_inputs.py``` - Code for scaling and preprocessing input frequencies.

\
```Exjobb_train_test_topic.ipynb``` - Evaluates the results of training a logistic regression model on the theses in one topic and testing on the other topics.


As input to the model, it uses the function word and pos-tag frequencies computed by ```compute_frequencies.py``` (e.g. ```frequencies_swe.csv``` and ```frequencies_eng.csv```) and dataset produced by ```Exjobb_downsample.ipynb``` 
(e.g. ```df_swedish_downsampled.csv``` and ```df_english_downsampled.csv```).


```Exjobb_reduce_topic.ipynb``` 

In this notebook, we train two logistic regression models on gender and on topic, compare the accuracies and list the important features. 

Furthermore, we identify the features with high topic correlation, and create reduced feature sets where features are dropped if their corresponding coefficient in the logistic regression model trained on topics has a coefficient that is higher than some threshold.
We plot the accuracies of the models trained on the reduced feature sets, which show show a faster decline in accuracy for topic than for gender.

```Exjobb_process_data.ipynb``` (no longer used)

This code is no longer needed, as the inferred author gender and language has been added to the metadata files, and the texts are read from the pos-tagged data using the script in ```compute_frequencies.py```.

Includes code for guessing the gender of the authors and the language that each thesis is written in.

Generate tables listing the number of male/female/unknown authors in English and Swedish for each department and faculty at Chalmers and GU.

Generates two csv-files, one for Chalmers and one for GU, containing the fetched texts of each thesis, inferred author gender and language. 

```parse_theses.Rmd``` 

Code for parsing theses from pdf and docx format (no longer used)
