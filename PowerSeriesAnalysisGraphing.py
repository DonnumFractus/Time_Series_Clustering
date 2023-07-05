import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import re
import sys

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

import networkx as nx

from scipy.optimize import curve_fit
import string
import pandas as pd

from math import sqrt
from fuzzywuzzy import fuzz
import time

import nltk
from nltk.corpus import stopwords
 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

cred = credentials.Certificate("kentauros-computing-firebase-adminsdk-htedt-07f4c25efb.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore database client
db = firestore.client()

# Specify the collection to retrieve documents from and the number of documents to retrieve
collection_ref = db.collection("Livestreams")

# basic Power_law distribution

def removeStopWords(text):

    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

    return tokens_without_sw

def power_law(x, a, b):
    return a * np.power(x, b)

def power_law_approximation(x, y):
    params, params_covariance = curve_fit(power_law, x, y)
    a = params[0]
    b = params[1]
    return a, b

def get_key_value_lists(dictionary):

    keys = []
    values = []

    for i, (key, value) in enumerate(dictionary.items()):

        keys.append(i)
        values.append(value)

    return keys, values

def plot_word_occurrences(key, value_dict, filename, x_values, a, b):
    # Create lists for labels and values
    labels = list(value_dict.keys())
    values = list(value_dict.values())
    
    new_y_values = [power_law(x, a, b) for x in x_values]

    # Create bar graph
    plt.figure(figsize=(12,4))
    plt.bar(labels, values, color='blue', edgecolor='black')
    plt.plot(labels, new_y_values, color='red')
    plt.ylabel('Occurrences')
    plt.xlabel('Words')
    plt.title(f'Title: {key}')
    plt.xticks(rotation=90)  # rotate the x-axis labels for better readability

    # Add power law equation to the plot
    plt.text(0.5, 0.97, f'y = {a}x^{b}', transform=plt.gca().transAxes)

    # Save the figure to a file
    plt.savefig(filename + ".jpg", bbox_inches='tight', format="jpg", dpi=150)
    plt.close('all')

# Returns a dictionary of unique characters and counted values
def count_string_occurrences(list_of_sets):
    occurrences = defaultdict(int)
    
    # Count the occurrences of each string in the list of sets
    for set_of_strings in list_of_sets:
        for string in set_of_strings:
            occurrences[string] += 1
    
    # Sort the dictionary based on the count of occurrences
    sorted_occurrences = dict(sorted(occurrences.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_occurrences

def count_word_frequency(data):
    frequency_dict = {}
    word_set = data[1]

    for word in word_set:
        frequency_dict[word] = 0

    for word_set in data[2]:
        for word in word_set:
            if word in frequency_dict:
                frequency_dict[word] += 1

    return {data[0]: frequency_dict}

# Load Firebase Database credentials
def retreiveChatData(numberOfVods):

    n_docs_to_retrieve = numberOfVods  # Change this to the desired number of documents to retrieve

    # Iterate over the first n documents in the collection
    docs = collection_ref.limit(n_docs_to_retrieve).stream()

    # Basic lists to store data in
    streamIDList = []
    multipleStreamVODChatList = {}

    # Select a list of document ID's to reference
    for doc in docs:
        doc_id = doc.id
        streamIDList.append(doc_id)

    # Download the ChatLogs as dictionaries and load a variable.
    for ids in streamIDList:

        sub_collection_ref = db.collection("Livestreams").document(ids).collection('ChatLogs')
        chat_logs = sub_collection_ref.stream()

        totalChatLog = []

        for chat_log in chat_logs:
            totalChatLog = totalChatLog + chat_log.to_dict()["Chats"]

        # adjust messages to only have relevant data

        totalChatLog = [chats for chats in totalChatLog if process_string(chats["message"])]

        for chats in totalChatLog:
            chats["message"] = process_string(chats["message"])


        list_size = sys.getsizeof(totalChatLog)
        print(f"List size: {list_size} bytes")
        print(ids)

        # Create a finalList of lists of message objects in
        multipleStreamVODChatList[ids] = totalChatLog


    print(multipleStreamVODChatList.keys()) # Print the dictionary Keys to confirm
    return multipleStreamVODChatList
    # From here on out, we can use the StreamVodChatList to examine multiple unique VOD chats, compare if needed.

# Returns an adjusted string
def process_string(input_string):

    # Remove Stop Works for simplicity
    #input_string = removeStopWords(input_string)
    #print(input_string)
    # Translation table to replace ":" with a space
    translation_table = str.maketrans({":": " "})

    # Create a set to store unique words
    unique_words = set()

    # Iterate over words in the input string
    for word in input_string.translate(translation_table).split():
        # Check if the word has at least one capitalized character
        if any(c.isupper() for c in word):
            # Add the word to the set of unique words
            unique_words.add(word)

    # Return the unique words as a string
    return " ".join(unique_words)

# analyze specific vod intervals
def analyzeSpecificVod(livestreamURL):


    regex_youtube = r"^https:\/\/www\.youtube\.com\/watch\?v=([\w_-]{11}).*$"
    regex_twitch = r"^https:\/\/www\.twitch\.tv\/videos\/(\d{10}).*$"

    vod_id = "invalid URL"

    if re.match(regex_youtube, livestreamURL):

        vod_id = re.match(regex_youtube, livestreamURL)[1]
        website = 'Youtube'

        player_url = f'https://www.youtube.com/embed/{vod_id}?autoplay=1'

    elif re.match(regex_twitch, livestreamURL):

        vod_id = re.match(regex_twitch, livestreamURL)[1]
        website = 'Twitch'

        player_url = f'https://player.twitch.tv/?video=v{vod_id}&parent=kentauroscomputing.com&autoplay=False'

        sub_collection_ref = db.collection("Livestreams").document(vod_id).collection('ChatLogs')
        chat_logs = sub_collection_ref.stream()

        totalChatLog = []

        for chat_log in chat_logs:
            totalChatLog = totalChatLog + chat_log.to_dict()["Chats"]

        return totalChatLog

def retreiveSpecificChat(vodID):

    collection_ref = db.collection("Livestreams").document(vodID).collection('ChatLogs')
    chat_logs = collection_ref.stream()
    # Download the ChatLogs as dictionaries and load a variable.

    totalChatLog = []

    for chat_log in chat_logs:
        totalChatLog = totalChatLog + chat_log.to_dict()["Chats"]

    # adjust messages to only have relevant data

    totalChatLog = [chats for chats in totalChatLog if process_string(chats["message"])]

    for chats in totalChatLog:
        chats["message"] = process_string(chats["message"])

    return totalChatLog
    # From here on out, we can use the StreamVodChatList to examine multiple unique VOD chats, compare if needed.

def convertDurationToBinnedHistogram(startTime,duration,binTime, chatLog):

    durationSeries = list(np.arange(startTime,duration, binTime)) # remove bin lengths
    binnedchatLogs = {binKey: [] for binKey in durationSeries}

    punctuation_without_colon = "".join(c for c in string.punctuation if c != ':')
    translator = str.maketrans("","", punctuation_without_colon) # List of all punctuations that need to be removed

    # iterate through the list and adjust the data to better be analyzed
    for chatMessage in chatLog:

        # adjust the wordlist to be a set of relevant words
        wordList = chatMessage["message"].translate(translator)
        wordList = wordList.split()
        wordList = set(wordList)

        messageBinPosition = chatMessage["timestamp"] - chatMessage["timestamp"] % vodData["binSize"]
        binnedchatLogs[messageBinPosition].append(wordList)

    return chatLog, binnedchatLogs

def convertDurationToBinnedHistogramtimeStamp(startTime, half_range, binTime, chatLog):

    #half_rangeSeries = list(np.arange(startTime,half_range, binTime)) # remove bin lengths
    binnedchatLogs = {msg["timestamp"]: [] for msg in chatLog}

    initTime, finTime = retrieve_timestamps_within_range(timestamp, half_range, vod_startTime, vod_endTime)

    punctuation_without_colon = "".join(c for c in string.punctuation if c != ':')
    translator = str.maketrans("","", punctuation_without_colon) # List of all punctuations that need to be removed

    # iterate through the list and adjust the data to better be analyzed
    for chatMessage in chatLog:

        # adjust the wordlist to be a set of relevant words
        wordList = chatMessage["message"].translate(translator)
        wordList = wordList.split()
        wordList = set(wordList)

        messageBinPosition = chatMessage["timestamp"] - chatMessage["timestamp"] % vodData["binSize"]
        binnedchatLogs[messageBinPosition].append(wordList)

    return chatLog, binnedchatLogs

def returnPowerDistributionConstantData(binnedHistogramWordSets, plotsWanted):

    fileNum = 0
    a_values = []
    b_values = []
    labeledData = []

    for binnedHistogramIndex in binnedHistogramWordSets:
        
        fileName = str(fileNum) + "_" + vodIDStringExpected + '_' + str(binnedHistogramIndex)
        
        x_values, y_values = get_key_value_lists(binnedHistogramWordSets[binnedHistogramIndex])
        x_values = [x + 1 for x in x_values]

        if len(x_values) > 1:

            a,b = power_law_approximation(x_values, y_values)
            a_values.append(a)
            b_values.append(b)

            labeledData.append({"a":a, "b":b,"label":fileNum, "timeStamp":binnedHistogramIndex})

            if (plotsWanted):
               plot_word_occurrences(binnedHistogramIndex, binnedHistogramWordSets[binnedHistogramIndex], fileName, x_values, a, b)

        fileNum = fileNum + 1

    return labeledData, a_values, b_values

def returnPowerDistributionConstantDataAlternative(binnedHistogramWordSets):

    for binnedHistogramIndex in binnedHistogramWordSets:
        
        x_values, y_values = get_key_value_lists(binnedHistogramWordSets[binnedHistogramIndex])
        x_values = [x + 1 for x in x_values]

        if len(x_values) > 1:
            a,b = power_law_approximation(x_values, y_values)
            binnedHistogramWordSets[binnedHistogramIndex] = a
        else:
            binnedHistogramWordSets[binnedHistogramIndex] = 0

    return binnedHistogramWordSets

def binnedChats2PowerDistributionDict(binnedHistogramWordSets):

    for index, binnedHistogramIndex in enumerate(binnedHistogramWordSets):

        binnedHistogramWordSets[index] = count_word_frequency(binnedHistogramIndex)


    return binnedHistogramWordSets

def conditionalPlotPrint(labeledData, a_cutoff, binnedHistogramWordSets, plotsWanted):

    filteredLabeledData = []
    filteredPowerDist = []
    a_filtered = []
    b_filtered = []

    for binSegments in labeledData:

        apass = binSegments["a"] >= a_cutoff
        bpass = binSegments["b"] <= -0.01

        if (apass & bpass):

            filteredPowerDist.append(binnedHistogramWordSets[binSegments["timeStamp"]])

            filteredLabeledData.append(binSegments)
            a_filtered.append(binSegments["a"])
            b_filtered.append(binSegments["b"])

            fileName = str(binSegments["timeStamp"])
            
            x_values, y_values = get_key_value_lists(binnedHistogramWordSets[binSegments["timeStamp"]])
            x_values = [x + 1 for x in x_values]

            if(plotsWanted):
                plot_word_occurrences(binSegments["timeStamp"], binnedHistogramWordSets[binSegments["timeStamp"]], fileName, x_values, binSegments["a"], binSegments["b"])

    return filteredLabeledData, filteredPowerDist, a_filtered, b_filtered 

def CutoffAValue(threshold, a_index_denominator, seriesValues):

    a_without_ones = sorted([constants for constants in seriesValues if constants >= threshold], reverse=True)
    a_index = int(round(len(a_without_ones)/a_index_denominator))
    
    a_cutoff = a_without_ones[a_index]

    return a_cutoff

def plotDualPlots(labeledData, a_values, b_values):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # Create two subplots
    ax1.scatter(a_values, b_values, alpha=0.5)
    for label in labeledData:
        ax1.text(label["a"], label["b"], label["label"], ha='center', va='bottom', fontsize=8)

    ax1.set_title(f"Scatter Plot of binned Data (20 second Increments) For Power Series Approximation over VOD Chat {vodString}")
    ax1.set_xlabel("a-constant value (ax^(b))")
    ax1.set_ylabel("b-constant value (ax^(b))")

    # a_cutoff = CutoffAValue(1, a_values)
    # a_cutoff_values = [a_cutoff] * len(a_values)

    # Line plot
    timeSeries_a_b = []
    for label in labeledData:
        timeSeries_a_b.append(label["timeStamp"])

    ax2.plot(timeSeries_a_b, a_values, label='a_values', alpha=0.5)
    ax2.plot(timeSeries_a_b, b_values, label='b_values', alpha=0.5)
    # ax2.plot(timeSeries_a_b, a_cutoff_values, label='a_cutoff_values', alpha=0.5)

    ax2.set_title("Line Plot of a_values and b_values")
    ax2.set_xlabel("Time Stamp (minutes)")
    ax2.set_ylabel("Value (a,b respectively)")
    ax2.legend()
    plt.show()

def fuzzyCombine(words, threshold):

    combined_words = []
    for word1 in words:
        for word2 in words:
            # Do not compare a word with itself
            if word1 != word2:
                # If the two words are similar according to the threshold
                if fuzz.ratio(word1, word2) > threshold:
                    combined_word = word1 + word2
                    combined_words.append(combined_word)

    return combined_words

def proportionalPowerAlgorithm(powerDict):

    keys = list(powerDict.keys())
    n = len(keys)
    data = [[None]*n for _ in range(n)]

    for i, row_key in enumerate(keys):
        for j, col_key in enumerate(keys):
            if i > j:
                w_n = powerDict[row_key]
                w_1 = powerDict[keys[0]]
                w_m = powerDict[col_key]
                y = sqrt(w_n ** 2 / (w_1 * w_m))
                data[i][j] = round(y,3)

    df = pd.DataFrame(data, index=keys, columns=keys)
    return df

def combine_dataframes(df1, df2):
    # Concatenate the two dataframes
    combined_df = pd.concat([df1, df2]).reset_index()

    # Group by index and transform each group to a list, ignoring NaNs
    combined_df = combined_df.groupby('index').agg(lambda x: list(x.dropna()))

    # Iterate over the combined dataframe to set NaN for the top right triangle
    for i in range(combined_df.shape[0]):
        for j in range(i, combined_df.shape[1]):
            combined_df.iloc[i, j] = np.nan

    return combined_df
# ------Power Algorithm ---------- 
def proportionalPowerAlgorithm(powerDict, PPlength):

    keys = list(powerDict.keys())[:PPlength]
    n = len(keys)

    data = [[[]]*n for _ in range(n)]

    for i, row_key in enumerate(keys):
        for j, col_key in enumerate(keys):
            if i > j:

                w_1 = powerDict[keys[0]]
                w_n = powerDict[row_key]
                w_m = powerDict[col_key]
                
                y = sqrt(w_n ** 2 / (w_1 * w_m))
                data[i][j] = [round(y,3)]
            else:
                data[i][j] = []

    df = pd.DataFrame(data, index=keys, columns=keys)

    return df

def proportionalPowerAlgorithmV2(powerDict, PPlength):
    
    keys = list(powerDict.keys())[:PPlength]
    n = len(keys)

    columnIndex = keys[0]
    rowIndex = keys
    data = [[]*n for _ in range(n)]
    
    for i, key in enumerate(rowIndex):

        a = [round(powerDict[key]/powerDict[columnIndex], 3)]
        data[i] = [a]

    df = pd.DataFrame(data, index=rowIndex, columns=[columnIndex])

    return df

def combine_dictionaries(d1, d2):

    combined_dict = {}
    
    # Iterate over the keys in d1
    for key in d1.keys():

        if key in d2:
            # Combine the lists for matching keys
            combined_dict[key] = d1[key] + d2[key]
        else:
            # If the key is not present in d2, add it with its list from d1

            combined_dict[key] = d1[key]
    
    # Add any keys in d2 that are not in d1
    for key in d2.keys():
        if key not in d1:
            combined_dict[key] = d2[key]
    
    return combined_dict

def extract_unique_dataframes(df1, df2):

    # get the column names of the pandas dataframes find where names are the same
    df1_labels = df1.columns.tolist()
    df1_labels2 = df2.columns.tolist()

    df_intersection = list(set(df1_labels) & set(df1_labels2))

    combined_overlap_frame = {}

    for labels in df_intersection:

        combinedDict = combine_dictionaries(df1[labels].to_dict(), df2[labels].to_dict())
        combined_overlap_frame[labels] = combinedDict

        del df1[labels]
        del df2[labels]

    combined_df = pd.DataFrame(combined_overlap_frame)

    return df1, df2, combined_df

def replace_nan_with_empty_list(dict):
    for key in dict:
        if np.nan(dict[key]):
            dict[key] = []
    return dict

def combine_three_dataframes(df1, df2, df3):

    merged_df = pd.concat([df1, df2], axis=1)
    merged_df = pd.concat([merged_df, df3], axis=1)

    return merged_df

def dataframe_info(df):
    if isinstance(df, pd.DataFrame):
        matrix_values = df.values
        columns = df.columns.tolist()
        row_keys = df.index.tolist()

        return matrix_values, columns, row_keys
    else:
        print("Provided argument is not a DataFrame")
        return None

def remove_nan_From_dataframe(merged_df):
    merged_df = merged_df.fillna(0)
    merged_df_dict = merged_df.to_dict()

    new_data = []
    column_labels = list(merged_df_dict.keys())
    index_labels = list(merged_df.index.tolist())

    for items in list(merged_df_dict.keys()):

        columnList = list(merged_df_dict[items].values())
        columnList = [item if item != 0 else [] for item in columnList]

        new_data.append(columnList)

    df_new = pd.DataFrame(list(zip(*new_data)), columns=column_labels, index=index_labels)

    return df_new

def replace_list_with_mean_or_zero(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: 0 if isinstance(x, list) and len(x) <= 1 else (np.mean(x) if isinstance(x, list) else x))
    return df

def remove_zero_rows_columns(df):
    # Remove all zero columns
    df = df.loc[:, (df != 0).any(axis=0)]
    # Remove all zero rows
    df = df.loc[(df != 0).any(axis=1)]
    return df

def DFS(df, index, used_cols, col_indices):
    if index == len(df):
        return True

    for col in range(len(df)):
        if df.iat[index, col] == 1 and col not in used_cols:
            used_cols.add(col)
            col_indices[index] = col

            if DFS(df, index + 1, used_cols, col_indices):
                return True

            used_cols.remove(col)
            col_indices[index] = -1

    return False

def reorder_columns_as_index(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder the columns of the DataFrame to match the order of the index."""
    if not df.shape[0] == df.shape[1]:
        raise ValueError("DataFrame must be square.")
        
    if set(df.columns) != set(df.index):
        raise ValueError("DataFrame columns and index must have same labels.")
        
    df = df[df.index]
    return df

def get_unique_items(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    
    unique_items = list(set1.symmetric_difference(set2))
    return unique_items

def diagonalize_ones_matrix(new_df):

    overlap = get_unique_items(list(new_df.index), list(new_df.columns))
    index = new_df.index.get_indexer(overlap)

    added_df = pd.DataFrame(0, index=new_df.index, columns=overlap)
    added_df = added_df.apply(lambda row: row.where(row.index != row.name, round(1.0000, 2)))

    combined_df = pd.concat([new_df, added_df], axis=1)
    combined_df = combined_df.reindex(columns=list(combined_df.index))

    return combined_df

def greater_symmetrize(df):
    df_T = df.T
    return df.where(df.gt(df_T), df_T)

def set_diagonal_zero(df):
    for col in df.columns:
        if col in df.index:
            df.loc[col, col] = 0
    return df

def retrieve_timestamps_within_range(timestamp, half_range, vod_startTime, vod_endTime):

    # First define the inital and final time to take from using the range
    time_range = {
        'initialTime': timestamp - half_range if (timestamp - half_range > vod_startTime) else vod_startTime,
        'finalTime':   timestamp + half_range if (timestamp + half_range < vod_endTime) else vod_endTime,
    }

    return time_range['initialTime'], time_range['finalTime']

def numbers_in_range(numbers, num_range):
    start, end = num_range
    return [num for num in numbers if start <= num <= end]

def find_values_with_key(data, key):
    values = []
    for d in data:
        if key in d:
            values.append(d[key])
    return values
# --------------------------------------

vodIDStringMinimum = '1810011271' # The minimum needed for sufficient data to draw conclusions (15 chats per Min)
vodIDStringExpected = '1823236342' # The expected chat logs we'd most likely receive (24 chats per Min)
vodIdString_99Perc = '1816825048' # The Chat that would cover 99.99% of chats (86 chats per Min)

vodString = vodIDStringExpected

chatLog = retreiveSpecificChat(vodString)

vodData = { # Units are in minutes
    'startTime': 0,
    'duration': chatLog[-1]['timestamp'],
    'binSize': (1/3),
}

# clean the chat data, convert chat data into binned times and put in same list
cleanedChatLog, binnedHistogramWordSets = convertDurationToBinnedHistogram(vodData["startTime"], vodData["duration"] + vodData["binSize"], vodData["binSize"], chatLog)

timestamplist = [messages["timestamp"] for messages in cleanedChatLog]

# organize the data into dictionaries with the bin time as key, the value is a dictionary of the word frequency of keywords/emotes found
#EmoteDistributionDict = binnedChats2PowerDistributionDict(binnedHistogramWordSets)

# WARNING, FIX THE OVERLAP PROBLEM. SETS REMOVE CHAT MESSAGES THAT HAVE THE SAME TIME STAMP. FIX IT IN FUTURE
binnedchatLogs = [{msg["timestamp"]: retrieve_timestamps_within_range(msg["timestamp"], (1/3), vodData["startTime"], vodData["duration"])}  for msg in chatLog]
adjustedchatLog = [{msg["timestamp"]: msg["message"]} for msg in chatLog]

# create a set list of common words that need to be removed
removeList = {"I", "I'm", "Tier", "They've", "A", "He", "Her", "The", "We", "THIS","If","The","THE","If", "OF", "NOT", "So", "it's", 'its' ,"You", "They", "They're", "WE", "I've", 'And'}

# interate through list of dictionaries of chat messages
for i_value, elements in enumerate(binnedchatLogs):

    # retrieve the dict key for the element
    dictKey = list(elements.keys())[0]

    # collect the messages within a list
    msg_within_range = []

    # return a list of timestamps that fall within the (1/3) time range.
    timestamps_within_range = numbers_in_range(timestamplist, elements[dictKey])

    #print(elements[dictKey])
    #print(timestamps_within_range)

    # iterate through the found time stamps
    for timestamp in timestamps_within_range:

        # return the list of all possible chatLogs with the same key as an identified timestamp
        concurrent_timestamp_ts = find_values_with_key(adjustedchatLog, timestamp)
        
        #print(timestamp)
        #print(concurrent_timestamp_ts)

        for timestamps_msg in concurrent_timestamp_ts:

            ts_message = timestamps_msg
            ts_message = set(ts_message.split()) - removeList
            ts_message = {str for str in ts_message if "@" not in str}
            
            if len(ts_message) > 0:
                msg_within_range.append(ts_message)


    # rewrite the chat logs so that we have the key phrase set and the cleared set.
    Og_Message_string = set(list(adjustedchatLog[i_value].values())[0].split())

    #print(Og_Message_string)
    OriginalMessage = Og_Message_string

    binnedchatLogs[i_value] = (dictKey, OriginalMessage, msg_within_range)

print(binnedchatLogs)
EmoteDistributionDict = binnedChats2PowerDistributionDict(binnedchatLogs)


newEmoteDistDict = []
a = (1)

for index, dict in enumerate(EmoteDistributionDict):
    
    maxValue = max(list(dict.values())[0].values())
    maxKey = max(list(dict.values())[0])

    if maxValue > 2:

        newDict = {list(dict.keys())[0]: (maxValue ** a, maxKey)}
        newEmoteDistDict.append(newDict)

keys = []
values = []
labels = []

for elements in newEmoteDistDict:

    keys.append(list(elements.keys())[0])
    values.append(list(elements.values())[0][0])
    labels.append(list(elements.values())[0][1])

data = list(zip(keys,values))
data = [list(tup) for tup in data]
data = np.array(data)

# Perform K-Means Clustering
k = 100 # Specify the number of clusters
kmeans = KMeans(n_clusters=k, random_state=0).fit(data)

# Get cluster assignments for each data point
clusters = kmeans.labels_

# Plot the data points, color-coded by cluster ID
plt.scatter(data[:,0], data[:,1], c=clusters)
for i, txt in enumerate(labels):
    plt.annotate(txt, (keys[i], values[i]), fontsize=6)  # this will add the value as label at each point

# BUG TO FIX: NEED TO FIGURE OUT WHY SOME WORDS ARE STILL BEING COUNTED 'TIER', 'ITS' ETC While being removed from the sentence. Why???

plt.ylabel('Linear Proportional data')
plt.xlabel('Time Stamps of the VOD')
plt.title(f'{k} cluster representation of stream {vodString} with proportion score a={a} for y=x^(a), n={len(keys)}')
plt.show()

"""
linked = linkage(data, 'single')

# Define a cutoff value for the clusters, and use fcluster to identify the individual clusters
max_d = 0
clusters = fcluster(linked, max_d, criterion='distance')

# Plotting the hierarchical clustering as a dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
plt.axhline(y=max_d, c='k')

plt.show()

# Plot the data points, color-coded by cluster ID
plt.scatter(data[:,0], data[:,1], c=clusters)

for i, txt in enumerate(labels):
    plt.annotate(txt, (keys[i], values[i]), fontsize=6)  # this will add the value as label at each point

plt.show()
"""
"""
print(keys)
print(values)
print(labels)

plt.scatter(keys, values)

for i, txt in enumerate(labels):
    plt.annotate(txt, (keys[i], values[i]), fontsize=6)  # this will add the value as label at each point


plt.ylabel('A Values')
plt.xlabel('Time Stamps')
plt.title(f'A Relative Values')
plt.xticks(rotation=90)  # rotate the x-axis labels for better readability
plt.show()
"""
"""
timeAValueDict = returnPowerDistributionConstantDataAlternative(EmoteDistributionDict)

timeSeries = list(timeAValueDict.keys())
aSeries = list(timeAValueDict.values())

a_min = CutoffAValue(1, 4, aSeries)
avg_min_values = [a_min]*len(timeSeries)
#plt.plot(timeSeries, aSeries, timeSeries, avg_min_values)
plt.scatter(timeSeries, aSeries)
plt.ylabel('A Values')
plt.xlabel('Time Stamps')
plt.title(f'A Relative Values')
plt.xticks(rotation=90)  # rotate the x-axis labels for better readability
plt.show()
"""


