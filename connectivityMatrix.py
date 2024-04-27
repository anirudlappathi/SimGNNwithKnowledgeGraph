import numpy as np
import csv
import os

from utils import getDate

class ConnectivityMatrix():

    paper_titles = []
    kw_to_index = {}
    allMatrix = []

    def __init__(self, setTopic):

        self.topic = setTopic
        self.paper_titles = []

        self.kw_to_index = {}
        self.connMatrix = np.zeros((len(self.kw_to_index), len(self.kw_to_index)), dtype=int)
        self.cell_added = 0

        self.allMatrix = []
        

    def addToConnMatrix(self, head, tail):

        self.cell_added += 1
        head_index = 0
        tail_index = 0
        if head not in self.kw_to_index:
            head_index = self.connMatrix.shape[0]
            self.kw_to_index[head] = head_index
            right_section = np.zeros(shape=(self.connMatrix.shape[0], 1))
            self.connMatrix = np.hstack((self.connMatrix, right_section))
            bottom_section = np.zeros(shape=(1, self.connMatrix.shape[1]))
            self.connMatrix = np.vstack((self.connMatrix, bottom_section))
        else:
            head_index = self.kw_to_index[head]

        if tail not in self.kw_to_index:
            tail_index = self.connMatrix.shape[0]
            self.kw_to_index[tail] = tail_index
            right_section = np.zeros(shape=(self.connMatrix.shape[0], 1))
            self.connMatrix = np.hstack((self.connMatrix, right_section))
            bottom_section = np.zeros(shape=(1, self.connMatrix.shape[1]))
            self.connMatrix = np.vstack((self.connMatrix, bottom_section))
        else:
            tail_index = self.kw_to_index[tail]

        self.connMatrix[head_index, tail_index] += 1

    def updateStoredSizes(self):
        for i in range(len(self.allMatrix)):
            matrix = self.allMatrix[i]
            n = matrix.shape[0] 
            z = len(self.kw_to_index) 
            enlarged_matrix = np.zeros((z, z))
            enlarged_matrix[:n, :n] = matrix
            self.allMatrix[i] = enlarged_matrix

    # RUN THIS AFTER EACH ARTICLE TO SEPERATE EACH MATRIX
    def storeCurrent(self, title):
        self.paper_titles.append(title)
        storedMatrix = self.connMatrix.copy()
        self.allMatrix.append(storedMatrix)
        self.connMatrix = np.zeros((len(self.kw_to_index), len(self.kw_to_index)), dtype=int)
        self.updateStoredSizes()
        # print(self.cell_added)
        self.cell_added = 0

    def getConnectivityMatrix(self):
        return self.connMatrix
    
    def getAllMatrix(self):
        return self.allMatrix
    
    def saveToCSV(self, uuid, topic_name):
        kw_names = [None] * len(self.kw_to_index)
        for k, v in self.kw_to_index.items():
            kw_names[v] = k

        currDate = getDate()
        folder_path = f'connMatrixCSV/{topic_name}-{currDate}-{uuid}'
        os.makedirs(folder_path)
        for i in range(len(self.allMatrix)):
            matrix = self.allMatrix[i]
            paper = self.paper_titles[i][:-4]

            filename = f'{folder_path}/{paper}.csv'

            with open(filename, mode='w+', newline='') as file:
                print("opened")
                writer = csv.writer(file)

                writer.writerow(['Keywords'] + kw_names)
                for row in range(len(kw_names)):
                    currRowData = [str(int(i)) for i in matrix[row]]
                    writer.writerow([kw_names[row]] + currRowData) 

    def setTopic(self, topic):
        self.topic = topic

    def clearAll(self):
        self.connMatrix = None
        self.kw_to_index = {}

if __name__ == "__main__":
    c = ConnectivityMatrix("Test")
    c.addToConnMatrix("a", "b")
    c.addToConnMatrix("a", "b")
    c.storeCurrent("title")
    c.addToConnMatrix("a", "c")
    c.addToConnMatrix("a", "b")
    c.storeCurrent("title2")
    c.saveToCSV("12312312312")
    # c.saveToCSV("12312312312")
