from datetime import datetime
from uuid import uuid1

# Returns data in this format "060105"
# This example represents July 1st, 2005
def getDate():
    today_date = str(datetime.today().date())
    today_date = today_date[5:7] + today_date[8:10] + today_date[2:4]
    return today_date

def giveUUID1():
    return str(uuid1())[:8]

def DictToList(dictionary):
    keywordList = [0] * len(dictionary)
    for k, v in dictionary.items():
        keywordList[v] = k
    return keywordList