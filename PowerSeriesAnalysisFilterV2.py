import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import numpy as np
from collections import defaultdict
from scipy.optimize import curve_fit

import re
import sys
import string

def powerSeriesFilterting(vodString, credentialID):

    cred = credentials.Certificate(credentialID)
    firebase_admin.initialize_app(cred)

    # Initialize Firestore database client
    db = firestore.client()

    # Specify the collection to retrieve documents from and the number of documents to retrieve
    collection_ref = db.collection("Livestreams")

    # basic Power_law distribution
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

    def count_string_occurrences(list_of_sets):
        occurrences = defaultdict(int)
        
        # Count the occurrences of each string in the list of sets
        for set_of_strings in list_of_sets:
            for string in set_of_strings:
                occurrences[string] += 1
        
        # Sort the dictionary based on the count of occurrences
        sorted_occurrences = dict(sorted(occurrences.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_occurrences

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

            # Create a finalList of lists of message objects in
            multipleStreamVODChatList[ids] = totalChatLog

        return multipleStreamVODChatList
        # From here on out, we can use the StreamVodChatList to examine multiple unique VOD chats, compare if needed.

    def process_string(input_string):
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

    def returnPowerDistributionConstantData(binnedHistogramWordSets, plotsWanted, vodIDStringExpected):

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

            fileNum = fileNum + 1

        return labeledData, a_values, b_values

    def binnedChats2PowerDistributionDict(binnedHistogramWordSets):

        for binnedHistogramIndex in binnedHistogramWordSets:
            
            binnedHistogramWordSets[binnedHistogramIndex] = count_string_occurrences(binnedHistogramWordSets[binnedHistogramIndex])
            
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

        return filteredLabeledData, filteredPowerDist, a_filtered, b_filtered 

    def CutoffAValue(threshold, a_index_denominator, seriesValues):

        a_without_ones = sorted([constants for constants in seriesValues if constants >= threshold], reverse=True)
        a_index = int(round(len(a_without_ones)/a_index_denominator))
        
        a_cutoff = a_without_ones[a_index]

        return a_cutoff
    # ---------Return Filtered Data -------------
    def return_time_stamp_list(filteredData):

        timeStampfiltered = []
        for i in filteredData:
            timeStampfiltered.append(i['timeStamp'])

        return timeStampfiltered

    def generate_number_ranges(numbers, constant):

        ranges = []
        for num in numbers:
            ranges.append((round(num,4), round(num + constant,4)))
        return ranges

    def combine_overlapping_ranges(ranges):
        combined_ranges = []
        if not ranges:
            return combined_ranges

        ranges.sort(key=lambda x: x[0])  # Sort the ranges based on the starting value

        start, end = ranges[0]
        for i in range(1, len(ranges)):
            next_start, next_end = ranges[i]
            if next_start <= end:
                # There is an overlap, update the end value
                end = max(end, next_end)
            else:
                # No overlap, add the current range to the combined_ranges
                combined_ranges.append((start, end))
                start, end = next_start, next_end

        combined_ranges.append((start, end))  # Add the last range

        return combined_ranges

    def filter_dicts_by_time(a, time_ranges):
        filtered_list = []

        for item in a:
            timestamp = item.get("timestamp")

            if timestamp:
                for start, end in time_ranges:
                    if start <= timestamp <= end:
                        filtered_list.append(item)
                        break

        return filtered_list
    # --------------------------------------

    """
    vodIDStringMinimum = '1810011271' # The minimum needed for sufficient data to draw conclusions (15 chats per Min)
    vodIDStringExpected = '1823236342' # The expected chat logs we'd most likely receive (24 chats per Min)
    vodIdString_99Perc = '1816825048' # The Chat that would cover 99.99% of chats (86 chats per Min)

    vodString = vodIDStringExpected
     """
    
    chatLog = retreiveSpecificChat(vodString)

    vodData = { # Units are in minutes
        'startTime': 0,
        'duration': chatLog[-1]['timestamp'],
        'binSize': (1/3),
    }

    # clean the chat data, convert chat data into binned times and put in same list
    cleanedChatLog, binnedHistogramWordSets = convertDurationToBinnedHistogram(vodData["startTime"], vodData["duration"] + vodData["binSize"], vodData["binSize"], chatLog)

    # organize the data into dictionaries with the bin time as key, the value is a dictionary of the word frequency of keywords/emotes found
    EmoteDistributionDict = binnedChats2PowerDistributionDict(binnedHistogramWordSets)

    # return the power constant data and printed plots
    labeledData, a_values, b_values = returnPowerDistributionConstantData(EmoteDistributionDict, False, vodString)

    a_min = CutoffAValue(1, 4, a_values)
    filteredData, filteredPowerDist, a_filtered, b_filtered  = conditionalPlotPrint(labeledData, a_min, binnedHistogramWordSets, False)

    timeStampfiltered = return_time_stamp_list(filteredData)
    new_range = generate_number_ranges(timeStampfiltered, (1/3))
    new_range = combine_overlapping_ranges(new_range)
    filteredChatLogs = filter_dicts_by_time(cleanedChatLog, new_range)

    reductionValue = round((len(cleanedChatLog) - len(filteredChatLogs))/len(cleanedChatLog) * 100, 2)
    print("chat size reduced by:",reductionValue,"%")

    return filteredChatLogs

filteredChats = powerSeriesFilterting('1823236342', "kentauros-computing-firebase-adminsdk-htedt-07f4c25efb.json")
print(filteredChats)
