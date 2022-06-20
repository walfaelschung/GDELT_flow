# Event Data Validation using GDELT - Supplement 
GDELT data collection and processing

## Disclaimer
This document contains supplementary material to the manuscript "From event data to protest events: Lifting the veil on the use of big data news repositories in protest event analysis". It lays out a step-by-step guideline of data collection, processing, and analysis as conducted by the researchers. Where possible, we share snippets of code to be amended and re-used by researchers and other interested parties for their own purposes and projects. As the entire workflow contains important steps that involve "manual" labelling of data and crucial decisions, this document is explicitly not meant as a script to be uncritically run, but as documentation of our approach to GDELT-data, which unveils some of the otherwise invisible steps and decisions we took, to aid and inspire others in their own projects. The steps taken do not necessarily represent the exact order of our project, but are assembled in the order that we deemed most practical after evaluating our own approach. Please note that all Python and R code was written by social scientists, not programmers, so feel free to improve code efficiency and parsimony.

----------------

## Step I - Event-Data Collection from Master-CSV
Since GDELT's Event Exporter (https://analysis.gdeltproject.org/module-event-exporter.html), which promises easy access to the Events Data is defunct, it is possible to use SQL-queries GDELT via Google BigQUery (not free). Free, easy but limited access to GDELT Event Data is also provided through ICore (https://icore.mnl.ucsb.edu/event/). For our purpose, there is not alternative to querying the raw comma-separated-value (CSV) files from GDELT's Master List with Python. 

We use the following packages:

    from pandas.errors import EmptyDataError
    import pandas as pd
    import zip file
    from zipfile import BadZipFile
    import requests
    from io import BytesIO
    import time
    import pickle
    
We create a list of urls that contain links to all zip-files that contain event data:

    master = pd.read_csv("http://data.gdeltproject.org/gdeltv2/masterfilelist.txt", sep=" ", header=None) # note: separator is blank space
    master = master[master[2].str.contains('export.CSV', na=False)] # column 2 contains the files. file names containing "export" refer to events data
    urls = list(master[2])
    
This is a list of the Event-Record collected in 15-minute intervals and looks like this:
    ![grafik](https://user-images.githubusercontent.com/34031060/167093068-55281a07-3049-47ba-8362-ff51f5cb42e6.png)


Next, we define the countries we are interested in. These are later used to filter the "ActionGeo_CountryCode" variable, which contains info on the country in which the event took place. 

**Note:** GDELT claims to use FIPS country-codes (https://en.wikipedia.org/wiki/List_of_FIPS_country_codes), which differ from ISO-codes. Make sure to double-check

    fipscodes = ['DA','GM','HU','IT','RO','UK']

We also define the variables we are interested in and create two empty dataframes that will be filled in the next steps: results_df saves all events, protest_df will be used to select only protest events.

    colnames = ['GlobalEventID', 'Day','MonthYear','Year','FractionDate',
                'Actor1Code','Actor1Name','Actor1CountryCode','Actor1KnownGroupCode',
                'Actor1EthnicCode','Actor1Religion1Code','Actor1Religion2Code',
                'Actor1Type1Code','Actor1Type2Code','Actor1Type3Code',
                'Actor2Code','Actor2Name','Actor2CountryCode','Actor2KnownGroupCode',
                'Actor2EthnicCode','Actor2Religion1Code','Actor2Religion2Code',
                'Actor2Type1Code','Actor2Type2Code','Actor2Type3Code',
                'IsRootEvent','EventCode','EventBaseCode','EventRootCode',
                'QuadClass','GoldsteinScale','NumMentions','NumSources',
                'NumArticles','AvgTone',
                'Actor1Geo_Type','Actor1Geo_Fullname',
                'Actor1Geo_CountryCode','Actor1Geo_ADM1Code','Actor1Geo_ADM2Code',
                'Actor1Geo_Lat','Actor1Geo_Long','Actor1Geo_FeatureID',
                'Actor2Geo_Type','Actor2Geo_Fullname',
                'Actor2Geo_CountryCode','Actor2Geo_ADM1Code','Actor2Geo_ADM2Code',
                'Actor2Geo_Lat','Actor2Geo_Long','Actor2Geo_FeatureID',
                'ActionGeo_Type','ActionGeo_Fullname',
                'ActionGeo_CountryCode','ActionGeo_ADM1Code','ActionGeo_ADM2Code',
                'ActionGeo_Lat','ActionGeo_Long','ActionGeo_FeatureID',
                'DATEADDED','SOURCEURL'
                ]
    result_df = pd.DataFrame(columns=colnames)
    protest_df = pd.DataFrame(columns=colnames)

We archive finished files and errors in two lists:

    finished_files =[]
    file_errors = []

The following script loops through the URLs and extracts all protest events in all countries, and filters these in a second step to only countries defined in the fipscodes-list. As this might take a long time, we output the passed time at every 10,000 events and save the collected results at regular intervals.
We need to unpack the zip files to access the CSV within. Note we filter events by "EventBaseCode" == 14, which contains protest events as defined in GDELTS CAMEO classification (http://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf).

    COUNT = 0
    start_time = time.time()
    TIMECOUNTER = 0
    for a in urls:
        TIMECOUNTER = TIMECOUNTER +1 
        if TIMECOUNTER == 10000:

            now_time = time.time()
            print('Another 10,000 files processed. Script running for {0} seconds.'.format(now_time - start_time))
            print('Writing a backup copy to csv...')
            result_df.to_csv('results.csv',index=False,sep='\t')
            protest_df.to_csv('results_all_protest.csv',index=False,sep='\t')
            TIMECOUNTER = 0

        print('Handling file '+str(COUNT)+' of '+str(len(urls)))
        COUNT = COUNT+1
        obj = requests.get(a).content
        try:
            zf = zipfile.ZipFile(BytesIO(obj), 'r')
            filename = zf.namelist()

            with zf.open(filename[0], 'r') as csvfile:
                try:
                    df = pd.read_csv(csvfile, sep="\t", header=None, dtype={26: str,27:str,28:str})
                    df.columns=colnames
                    protestdf = df.loc[df.EventBaseCode.str.startswith('14', na=False)]  # EVENT-FILTER HERE
                    protest_df = protest_df.append(protestdf)
                    df_to_add = protestdf.loc[protestdf.ActionGeo_CountryCode.isin(fipscodes)] # COUNTRY-FILTER HERE
                    result_df = result_df.append(df_to_add)
                except EmptyDataError:
                    print('File was empty, moving on...')
                    file_errors.append(a)

        except BadZipFile:
            file_errors.append(a)
            print('Could not open zip file. Moving on...')
        finished_files.append(a)

We might want to check for duplicate entries, set a date range that we are interested in and run some basic descriptives (Note: If you are interested in smaller Timeframes, it saves time to filter for date earlier, e.g. by using the file names of zip archives as filters):

    results.Year.describe()
    daterange = range(2015,2021,1)
    results = results.loc[results.Year.isin(daterange)]
    results = results.drop_duplicates(subset = ['GlobalEventID'])
    print(results.groupby(['ActionGeo_CountryCode']).size())
    print(results.Day.min())
    print(results.Day.max())
    len(results)
        
 

### Step 1.1
**Note:** The above steps collect data for Events collected from English-language sources. One of GDELT's strengths lies in also collecting (and translating) news in different languages. To obtain event data from this "translingual" dataset, the above steps must be repeated with the following data

        master = pd.read_csv("http://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt", sep=" ", header=None)
        
 Results can be merged and checked for duplicates.
 
---------------------------

## Step 2 - Enrich Event Data with Mentions Data
The high number of false positives in the Event dataset makes a critical inspection necessary - therefore, we want to collect as much information on Events as possible. 
GDELT's coding is based on monitoring of many news sources. However, the Event dataset contains only the URL of a single story. To enrich this with all information that GDELT used, we 'enrich' the Event Data with all Mentions (i.e. story URLs) that report on an Event. This allows us to assess multiple sources when identifying false positives. In addition, having multiple sources per event is advantageous for studies reaching further back in time: the risk of missing data due to broken URLs can thus be reduced.

The following Python code-snippet makes use of the results from Steps 1 and 1.1. (results dataframe)

First, we obtain a list of zip files that contain the Mentions dataset

        master = pd.read_csv("http://data.gdeltproject.org/gdeltv2/masterfilelist.txt", sep=" ", header=None)
        master = master[master[2].str.contains('mentions.CSV', na=False)]
        urls = list(master[2])
        del(master)
        master = pd.read_csv("http://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt", sep=" ", header=None)
        master = master[master[2].str.contains('mentions.CSV', na=False)]
        urls = urls + list(master[2])
        del(master)
        len(urls) != len(set(urls)) # two dupes in the 490k files
        len(set(urls))
        urls = list(set(urls))

Next, we query each of those zip-files for those entries which mention one of the GlobalEventID in our Event Data. Results are stored in an empty pandas dataframe labelled mentions_df:

    colnames = ['GlobalEventID', 'EventTimeDate','MentionTimeDate','MentionType','MentionSourceName',
                'MentionIdentifier','SentenceID','Actor1CharOffset','Actor2CharOffset',
                'ActionCharOffset','InRawText','Confidence',
                'MentionDocLen','MentionDocTone','MentionDocTranslationInfo',
                'Extras']
    mentions_df = pd.DataFrame(columns=colnames)
    COUNT = 0
    for a in urls:

        print('Handling file '+str(COUNT)+' of '+str(len(urls)))
        COUNT = COUNT+1
        obj = requests.get(a).content
        try:
            zf = zipfile.ZipFile(BytesIO(obj), 'r')
            filename = zf.namelist()

            with zf.open(filename[0], 'r') as csvfile:
                try:
                    df = pd.read_csv(csvfile, sep="\t", header=None)
                    df.columns=colnames
                    df_to_add = df.loc[df.GlobalEventID.isin(results_df.GlobalEventID)]
                    mentions_df = mentions_df.append(df_to_add)
                except EmptyDataError:
                    print('File was empty, moving on...')
                    file_errors.append(a)

        except BadZipFile:
            file_errors.append(a)
            print('Could not open zip file. Moving on...')
        finished_files.append(a)


The resulting mentions data frame should look something like this:

![grafik](https://user-images.githubusercontent.com/34031060/167127594-2de921cb-660d-4169-93f7-30a38ff8e1f2.png)

**The results after Step 2 are two datasets: One with Events as the unit of observation and one with the Mentions as units of observation.**
**Note that the Mentions data frame contains a confidence-variable that measures (in steps of 10) how certain GDELT is that a given event is truly mentioned in the story**

See the distribution of confidence scores in our data here:

![grafik](https://user-images.githubusercontent.com/34031060/167844652-b31c140f-d81d-4616-b411-2387fc7f1f73.png)


**At this point, depending on a project's scope, the dataset can be huge and the following steps become costly, labour-intense, or otherwise unfeasible.
Therefore, we decided to filter the Events data by the Number of Articles reporting on a given event (the reasoning being that the more reports on an event exist, the likelihood of a false positive entry in the data is reduced. The mentions data can accordingly be filtered to mentions of remaining events and further filtered by the confidence score, to reduce false-positive mentions. While our project (link) uses specific thresholds, we urge researchers to explore meaningful values depending on their own project**

-------------------------------

## Interlude: Human Coding and insufficient Machine Learning classification

As false positive entries are a well-known issue in GDELT datasets, we wanted to learn about the magnitude if this problem and find ways toward a more valid set of protest events. We checked both the original url and all stories from the Mentions dataset to assess, whether an event is a true positive entry. Defining a true positive as an entry that actually reports on a protest event that happened in the country and at the approximate date that GDELT claims it did, lead us to identify only 448 events out of a random sample of 1,000 events. We concluded that taking GDELT data at face value when doing protest event analysis is not acceptable for our purposes. 
At the same time, it was unfeasible to let humans validate every entry in the dataset. THerefore, we used the random sample of 1,000 events as a test and training set for a Machine Learning classifier, which would ideally be able to identify patterns in GDELT's data that help exlcude false positives.

For that task, we used the sklearn library in Python, and tested several formulae and classification algorithms, none performed suffiencetly. For example, including actors, root_event, quadclass, goldstein_scale,year, average tonality, number of articles and number of sources in the model produced the following results:
 
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        from sklearn import metrics
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        
We found and accuracy of 0.6379310344827587
Logistic regression performed at an accuracy of .69, while support vector machine reached an accuracy of .68.

We decided that the variables included in GDELT did not contain anough information to allow robust classification. We therefore decided to collect additional data from the original news-sources that GDELT analyzed. 

--------------------------------

## Step 3 - Retrieval of article texts and translate non-English texts to English

This part is extremely resource-intense, so consider a batch-wise approach or sourcing the task out so machines with high computational resources. Depending on your own context and project scope, it might be worth exploring different translation models. 
In general, this part requires a data frame that contains the GlobalEventID variable as well as the story-URL (called SOURCEURL in the Event-Data or MentionIdentifier in the Mentions-Data). This data should ideally be split into English and non-English resources, by separating the Events-Data in the translist-Master files from the English-language Master files and by using the "MentionDocTranslationInfo" variable in the Mentions Data (empty for English documents. For the collection of article texts, make sure to follow rules and respect the robots.txt. The code snippet below is one of several approaches we tested, using the urllib, reppy, and newspaper packages in Python. R users might prefer polite to query robots.txt and rvest to collect data. Note that errors stemming from robots.txt disallowing access to an article should be double-checked. For many domains, we found the results that were retrieved to be incorrect.
The Code-Snippets below illustrate how this process can be done. As it's highly dependent on project needs and resources, consider adjusting the parameters to your needs. Note that the translation part is of course not required for English-language documents. For each document that requires translation, the MentionDocTranslationInfo variable should be parsed to identify the source language (in the code example, we suppose three lists of equal length and order: one containing the Event-ID, one containing the story URL, and one containing that stories language as obtained from the Mentions-table). **Note:** GDELT stores language codes in ISO639-2 format, Opus-MT uses ISO 639-1 https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes. In our case, we compared the codes in our dataset and changed the source_language variable where applicable.
Depending on your data and infrastructure, the code may raise different errors. Here, we just handle some that appeared in our application.

        from urllib.parse import urlparse
        from reppy.robots import Robots
        import newspaper
        from newspaper import ArticleException
        from easynmt import EasyNMT

        model = EasyNMT('opus-mt', max_loaded_models=40)
        robotallow = []
        rotslist = []
        resultslist = []
        elist = []


        for a in urllist:
            COUNT = COUNT + 1
            dom = 'http://'+urlparse(a).netloc+'/robots.txt'
            rotslist.append(dom)
            try:

                robots = Robots.fetch(dom)             
                if robots.allowed(a,'*') == True:
                    robotallow.append(True)
                    url_i = newspaper.Article(a)
                    try:
                        url_i.download()
                        url_i.parse()
                        try:
                            t_text = model.translate(url_i.text, source_lang = source_language[COUNT], target_lang = 'en')
                        except:
                            t_text = 'not translated'
                        resultslist.append(t_text)
                    except ArticleException:
                        resultslist.append('er')

                else:
                    robotallow.append(False)
                    resultslist.append('NA')
            except Exception:
                elist.append(a)


 The results may be processed as needed. We recommend cleaning the translated and untranslated texts for stopwords, numbers, double-whitespaces, etc., depending on the research interest (see a great introduction here: https://machinelearningmastery.com/clean-text-machine-learning-python/).
In our case, we were interested only in the appearance of certain keywords within an article, that indicate protest events. 
 
----------------------------------

## Step 5 - Dictionary-based protest identification in the article text 

The variables we used for Machine Learning are based on the intersection of article texts with a dictionary of protest-related terms.
If we define the dictionary like this:

                prodict_eng = ['demonstration', 'protest', 'rally', 'march', 'parade', 'riot',  'strike', 'boycott' , 'sit-in', 'crowd', 'mass', 'picket', 'picket line', 'blockade', 'mob', 'flash mob', 'revolution', 'rebellion',  'demonstrations', 'protests', 'rallies', 'marches', 'parades', 'riots',  'strikes', 'boycotts' , 'sit-ins', 'crowds', 'masses', 'pickets', 'picket lines', 'blockades', 'mobs', 'flash mobs', 'revolutions', 'rebellions','clash','demonstrate','campaign','protester','protesters']

We can implement this in the resultlist from Step 4 like this:

    for text in resultslist:
        matches = [word for word in prodict_eng if  word in re.split('\s+', text.lower())]

This returns a simple list of all terms from the dictionary that were also found in the text. 
Based on this matching, we created several variables on **Event**-level:
1) A "naive" matching variable: True if any dictionary term was found in any story on an event, false otherwise.
2) A majority matching variable: True if a majority of stories on an event contained one of the dictionaries terms.
3) A three-way factorial variable (True, False, No_Info) that determined whether the majority of stories contained one of dictionary terms, did not contain the terms, or did not contain enough info (due to text that are too short of missing)


-----------------------------------

## Step 6 - Train and apply a Machine Learning (ML) classifier

As we imposed additional restrictions on the number of stories per event and the confidence of mentions, applying the same restrictions to the sample coded by humans reduced n from 1,000 to 930. With a 70/30 split, we used this sample as a training and test set for to evaluate different Machine Learning classifiers which used the variables created in step 5 as additional predictors. 

We ran our ML-models in R using the following packages

        library(tidyverse)
        library(caret)
        library(MLeval)

To train the classifier, we begin with a dataframe (df) that contains the original GDELT Event variables (see Step 1), the results of our story retrieval and dictionary search (see Step 5), as well as the result of human validation in the form of a binary variable (1 = Protest Event (PE), 0 = No PE). We propose to identify the GDELT variables that are most meaningful for each project and that contain the least missing values. For our purposes, we subset the data to:

    subset <- 
        df %>%
            select(GlobalEventID,Protest_event, MonthYear, IsRootEvent,GoldsteinScale,AvgTone,fulltext_factor_result) 


Which looks something like this: 
        
        glimpse(subset)
        
        ## Rows: 930
        ## Columns: 7
        ## $ GlobalEventID                    <chr> "410695965", "412532378", "412644348"~
        ## $ Protest_event                    <fct> 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0~
        ## $ MonthYear                        <dbl> 201502, 201502, 201502, 201503, 20150~
        ## $ IsRootEvent                      <dbl> 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1~
        ## $ GoldsteinScale                   <dbl> -7.5, -6.5, -6.5, -6.5, -6.5, -6.5, -~
        ## $ AvgTone                          <dbl> -7.9207921, 2.8277635, -0.9779951, -0~
        ## $ fulltext_factor_result           <chr> "NO", "YES", "NO", "NO", "NO", "YES",~
      
Depending on the data used, pre-processing is advisable. We converted character variables as well as RootEvent and MonthYear to factor, while the caret package allows to center and rescale numeric variables. Note that the EventID must of course be excluded from the classifier as it's unique to each observation.
Next, we split the data into training and test set:

    set.seed(42)
    train_id <- createDataPartition(training_subset_use$Protest_event, list = F, p = .7)
    data_train <- training_subset_use %>% slice(train_id)
    data_test <- training_subset_use %>% slice(-train_id)
    prep <- data_train %>% select(-Protest_event) %>%
            preProcess(method = c("center", "scale"))
    data_train_preped <- predict(prep, data_train)
    data_test_preped <- predict(prep, data_test)
    
The caret package (https://topepo.github.io/caret/index.html) offers excellent documentation of the various functions for data processing, model training and tuning. For us, a Support Vector Machine with linear Kernel provided the best recall values with acceptable precision.
    
        formula_model <- formula("Protest_event ~ IsRootEvent + MonthYear + GoldsteinScale +
                         AvgTone + fulltext_factor_result")                     
        train_params <- trainControl(method = "cv", summaryFunction = twoClassSummary,classProbs = T,savePredictions = T)
        model_svm <- train(formula_model, data = data_train_preped, method = "svmLinear",  trControl = train_params, metric="Sens")
        pred_test_svm <- predict(model_svm, newdata = data_test_preped)
        confusionMatrix(pred_test_svm, data_test_preped$Protest_event, positive = "PE", mode = "prec_recall")
 
 Which produced the following Confusion Matrix:
 
 ![grafik](https://user-images.githubusercontent.com/34031060/167841470-899da0ae-3ee8-4efb-b8de-b1111bd2aee2.png)

       
In other words, without knowing the results of human coding (i.e. ground truth), the classifier identified 129 protest events correctly, failed to identify 27 protest events as such, correctly classified 77 events as not protest events and identified 45 events as protest events that were actually not protest events. While our final step of human validation should sort out these 45 cases, the principal aim is to keep the number of actual protest events that are mislabelled by classifier as low as possible, as humans will later only inspect the observations labelled "PE" by the classifier. This justifies an emphasis on recall over precision - however, for many other applications, researchers might lay a different focus, so make sure to select your classifier and parameters accordingly. In our case, the best results we achieved with a k-nearest neighbours classifier reached precision .69 and recall .71, Naive Bayes .72 and .74.
 
-----------------------


## Step 7 - Human Validation of Protest labels.

Once the classifier was run on the entire dataset, we obtained 91,721 Protest Event and 54,756 No Protest Event Labels with the abovementioned criteria for data selection. Since this is still too much for us to handle in human coding, we shifted the filter for the number of stories reporting on an event from the 60th to the 95th percentile, reasoning that the most impactful events might be the ones who received most attention in the form of news stories. This resulted in a reduction to 6,190 PE and  3,863 No_PE. Human coders inspected the 6,190 events according the the coding rules outlined above. 

![grafik](https://user-images.githubusercontent.com/34031060/167840841-87599776-5fdf-401e-bcd0-95047ba57026.png)

![grafik](https://user-images.githubusercontent.com/34031060/167841180-7778c4be-10d4-4ce1-85bb-a03874f7885b.png)


The results indicate that 3,686 events were labelled true positives after human inspection (around 60 per cent). In (only) 229 cases, none of the urls to an event's stories was working (even with the help of the Internet Archive's Wayback machine) or could not be translated to English. This indicates that mislabelling by GDELT seems to be a far more common source of error than missing information.
