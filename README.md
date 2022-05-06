# Insert smart title here
GDELT data collection and processing

## Discaimer
This documents contains supplementary material to the manuscript "From event data to protest events: Lifting the veil on the use of big data news repositories in protest event analysis". It lays out a step-by-step guideline of data collection, processing, and analysis as conducted by the researchers. Where possible, we share snippets of code to be amended and re-used by researchers and other interested parties for their own purposes and projects. As the entire workflow contains important steps that involve "manual" labelling of data and crucial decisions, this document is explictly not meant as a script to be uncritically run, but as documentation of our approach to GDELT-data, which unveils some of the otherwise invisible steps and decisions we took, to aid and inspire others in their own projects. The steps taken do not necessarily represent the exact order of our project, but are assembled in the order that we deemed most practical after evaluating our own approach. Please note that all Python and R code was written by social scientists, not programmers, so feel free to improve code efficiency and parsimony.

----------------

## Step I - Event-Data Collection from Master-CSV
Since GDELT's Event Exporter (https://analysis.gdeltproject.org/module-event-exporter.html), which promises easy access to the Events Data is defunct, it is possible to use SQL-queries GDELT via Google BigQUery (not free). Free, easy but limited access to GDELT Event Data is also provided through ICore (https://icore.mnl.ucsb.edu/event/). For our purpose, there is not alternative to querying the raw comme-separated-value (CSV) files from GDELT's Master List with Python. 

We use the follwing packages:

    from pandas.errors import EmptyDataError
    import pandas as pd
    import zipfile
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


Next, we define the countries we are interested in. These are later used to filter the "ActionGeo_CountryCode" variable, which contains info on the country in which the Event took place. 

**Note:** GDELT claims to use FIPS country-codes (https://en.wikipedia.org/wiki/List_of_FIPS_country_codes), which differ from ISO-codes. Make sure to double-check

    fipscodes = ['DA','GM','HU','IT','RO','UK']

We also define the variables we are interested an create two empty dataframes that will be filled in the next steps: results_df saves all events, protest_df will be used to select only protest events.

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

The following script loops through the urls and extracts all protest events in all countries, and filters these in a second step to only countries defined in the fipscodes-list. As this might take a long time, we output the passed time at every 10,000 events and save the collected results in regular intervals.
We need to unpack the zip-files to access the csv within. Note by "EventBaseCode" == 14, which contain protest events as defined in GDELTS CAMEO classification (http://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf).

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
        
 
               
We pickle the results and additionally write out to (tab-separated) csv:

    result_df.to_csv('results.csv',index=False,sep='\t')   
    protest_df.to_csv('results_all_protest.csv',index=False,sep='\t')
    pickle.dump(result_df,open('results.p','wb'))
    pickle.dump(protest_df,open('results_all_protest.p','wb'))
    pickle.dump(finished_files,open('finished_files.p','wb'))
    pickle.dump(file_errors,open('file_errors.p','wb'))


### Step 1.1
**Note:** The above steps collect data for Events collected from English-language sources. One of GDELT's strenghts lies in also collecting (and translating) news in different languages. To obtain event data from this "translingual" dataset, the above steps must be repeated with the following data

        master = pd.read_csv("http://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt", sep=" ", header=None)
        
 Results can be merged and checked for duplicates.
 
---------------------------

## Step 2 - Enrich Event Data with Mentions Data
The high number of false positives in the Event dataset makes a critical inspection necessary - therefore, we want to collect as much information on Events as possible. 
GDELT's coding is based on monitoring of many news sources. However, the Event dataset contains only the url of a single story. To enrich this with all information that GDELT used, we 'enrich' the Event Data with all Mentions (i.e. story urls) that report on an Event. This allows us to assess multiple sources when identifying false positives. In addition, having multiple sources per event is advantageous for studies reaching further back in time: the risk of missing data due to broken urls can thus be reduced.

The following Python code-snippet makes use of the results from Steps 1 and 1.1. (results dataframe)

First, we obtain a list of zip-files that contain the Mentions dataset

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

We pickle the resulting dataframe, a list of finished files as well as possible errors for later reference:

    pickle.dump(mentions_df,open('mentions_df.p','wb'))
    pickle.dump(finished_files,open('finished_files_mentions.p','wb'))
    pickle.dump(file_errors,open('file_errors_mentions.p','wb'))

The resulting mentions dataframe should look something like this:

![grafik](https://user-images.githubusercontent.com/34031060/167127594-2de921cb-660d-4169-93f7-30a38ff8e1f2.png)

**The results after Step 2 are two datasets: One with Events as the ubservation and one with the Mentions as units of observation.**
**Note that the Mentions dataframe contains a confidence-variable that measures (in steps of 10) how certain GDELT is that a given event is truly mentioned in the story**

**At this point, depending on a project's scope, the dataset can be huge and the following steps become costly, labor intense, or otherwise unfeasible.
Therefore, we decided to filter the Events data by the Number of Articles reporting on a given event (the reasoning being that the more reports on an event exist, the likelihood of a false positve entry in the data is reduced. The mentions data can accordingly be filtered to mentions of remaining events and further filtered by the confidence score, to reduce false positive mentions. While our project (link) uses specific thresholds, we urge researchers to explore meaningful values depending on their own project**

-------------------------------

## Step 3 - Retrieval of article texts

-------------------------------

## Step 4 - Translation of non-English texts

----------------------------------

## Step 5 - Manually attach a "validation" label to a sample

-----------------------------------

## Step 6 - Train and apply a machine learning (ML) classifier

-----------------------
