# for tokenizing into relations
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import os
from os import listdir
from os.path import isfile, join
from dotenv import load_dotenv
load_dotenv()


# from neo4j import GraphDatabase

# gpt3 to use with contextual data
# import openai
# openai.api_key = os.getenv('OPEN_AI_KEY')

# for mysql database
# from mysqlconnection import DBSession, Base
# from keywordsdb import DBKeywords
# from inputdatadb import DBInputData
from utils import getDate, DictToList, giveUUID1
from uuid import uuid1

# to make the DNN
import numpy as np

# to save files
import csv

from connectivityMatrix import ConnectivityMatrix
from tqdm import tqdm

from PyPDF2 import PdfReader

from functools import partial
from multiprocessing import Pool, Manager

# # load pdf into strings
# from langchain_community.document_loaders import PyPDFLoader
# # chunk strings to later process
# from langchain.text_splitter import RecursiveCharacterTextSplitter

class Neo4jConnection:
    '''
    Establishes connection to the Neo4j graph database
    '''
    def __init__(self, set_topic_name, uri='neo4j+s://5b656ba4.databases.neo4j.io:7687', user='neo4j', password=os.getenv('NEO4JPASS')):
        # self.driver = GraphDatabase.driver(uri, auth=(user, password))

        self.relations_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")

        self.gen_kwargs = {
            "max_length": 256,
            "length_penalty": 0,
            "num_beams": 10,
            "num_return_sequences": 10,
        }

        self.local_nodes_counter = {}
        self.local_relation_counter = {}

        # pdf scoped cache
        self.curr_inputdata = None
        self.curr_new_keywords_added = 0
        self.curr_column = np.empty((0,0))

        # global cache for words
        # keyword to index in column
        self.topic_name = set_topic_name
        self.keyword_am_index = {}
        self.data_am_index = []
        self.adjacency_matrix = None

        self.connectivityMatrix = ConnectivityMatrix("Applications of Reinforcement Learning for Biodiversity Analysis")

        # self.keywordsdb = DBKeywords()
        # self.inputdatadb = DBInputData()

    def PDFtoString(self, filepath):
        # print("STARTING: PDFtoString")

        reader = PdfReader(filepath)
        all_words = ""
        for pn in range(len(reader.pages)):
            page = reader.pages[pn]
            all_words += page.extract_text().replace("\n", " ")

        return all_words

    def TextSplitter(self, text, chunk_size=1000, chunk_overlap=20):
        chunks = []
        current_chunk = []
        current_length = 0

        words = text.split()

        for word in words:
            current_chunk.append(word)

            if current_length >= chunk_size and word.endswith('.'):
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_length = 0
            current_length += 1

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

        return chunks

    def TokenizeText(self, split_text):
        extracted_text = ''
        for text in split_text:
            model_inputs = self.tokenizer(text, max_length=254, truncation=True, padding=True, return_tensors = 'pt')

            generated_tokens = self.relations_model.generate(
                model_inputs["input_ids"].to(self.relations_model.device),
                attention_mask=model_inputs["attention_mask"].to(self.relations_model.device),
                **self.gen_kwargs,
            )

            decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            extracted_text += ''.join(decoded_preds)
        return extracted_text
    
    def TripletsAppend(self, triplets, subject, relation, tail):
        subject = subject.strip().lower()
        relation = relation.strip().replace(" ", "_").upper()
        tail = tail.strip().lower()

        for word in [subject, tail]:
            if word[-2:] == "ed":
                word = word[:-2]
            if word[-3:] == "ing":
                word = word[:-3]
            if word[-1:] == "s":
                word = word[:-1]
        self.local_nodes_counter[subject] = self.local_nodes_counter.get(subject, 0) + 1
        self.local_nodes_counter[tail] = self.local_nodes_counter.get(tail, 0) + 1

        self.local_relation_counter[(subject, relation, tail)] = self.local_relation_counter.get((subject, relation, tail), 0) + 1

        triplets.append({'head': subject, 'type': relation,'tail': tail})
    
    def ExtractTriplets(self, text):
        # print("STARTING: ExtractTriplets")
        triplets = []
        relation, subject, relation, tail = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", " ").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    self.TripletsAppend(triplets, subject, relation, tail)
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    self.TripletsAppend(triplets, subject, relation, tail)
                tail = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    tail += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and tail != '':
            self.TripletsAppend(triplets, subject, relation, tail)

        return triplets

    # TO CODE LATER
    def loadCSVData(filename):
        loaded_data = np.genfromtxt(filename, delimiter=',', names=True, dtype=None, encoding=None)

    def setDataInputAdjacency(self, inputdata):
        self.curr_inputdata = inputdata
        self.data_am_index.append(inputdata)
        if self.adjacency_matrix is None:
            self.curr_column = None
        else:
            self.curr_column = np.zeros((self.adjacency_matrix.shape[0], 1), dtype=int)

    def GetRelations(self, pdf_file_path):

        inputdata_filepath = pdf_file_path.split("/")[-1]
        self.setDataInputAdjacency(inputdata_filepath)

        pdf_text = self.PDFtoString(pdf_file_path)
        split_text = self.TextSplitter(pdf_text)
        extracted_text = self.TokenizeText(split_text)
        return self.ExtractTriplets(extracted_text)

    # NOT IN USE RIGHT NOW
    def CreateUniqueEmbeddings(self, triplets):
        
        nameToEmbedding = {}
        # only create embeddings for nodes not in graph and once for all same node name in triplets
        nodeNames = set()
        for triple in triplets:

            if triple['head'] not in nodeNames:
                result = self.driver.execute_query(f"""
                    MATCH (n:Node {{name:$name}})
                    RETURN n                                                   
                """, name=triple['head'],
                database_="neo4j")
                if result[0]['count'] <= 0:
                    nodeNames.add(triple['head'])

            if triple['tail'] not in nodeNames:
                result = self.driver.execute_query(f"""
                    MATCH (n:Node {{name:$name}})
                    RETURN n                                                   
                """, name=triple['tail'],
                database_="neo4j")
                if result[0]['count'] <= 0:
                    nodeNames.add(triple['tail'])
        nodeNames = list(nodeNames)
        embeddings = self.embedding_model.encode(nodeNames, convert_to_tensor=True)
        for nameIndex in range(len(nodeNames)):
            nameToEmbedding[nodeNames[nameIndex]] = embeddings[nameIndex]
        return nameToEmbedding

    def AddToCurrColumn(self, head, tail):
        if self.curr_column is None:
            self.curr_column = np.array([1])
            self.keyword_am_index[head] = self.curr_column.shape[0] - 1
        elif head in self.keyword_am_index:
            self.curr_column[self.keyword_am_index[head]] += 1
        else:
            self.curr_new_keywords_added += 1
            self.curr_column = np.append(self.curr_column, [1])
            self.keyword_am_index[head] = self.curr_column.shape[0] - 1

        if tail in self.keyword_am_index:
            self.curr_column[self.keyword_am_index[tail]] += 1
        else:
            self.curr_new_keywords_added += 1
            self.curr_column = np.append(self.curr_column, [1])
            self.keyword_am_index[tail] = self.curr_column.shape[0] - 1

    def completeDataInputAdjacency(self):
        yOfZeros = len(self.data_am_index) - 1
        xOfZeros = self.curr_new_keywords_added

        KWDataZeros = np.zeros((yOfZeros, xOfZeros)).T
        a = np.array([self.curr_column])
        if self.adjacency_matrix is None:
            self.adjacency_matrix = np.array([self.curr_column.T], dtype=int).T
        else:
            self.adjacency_matrix = np.vstack((self.adjacency_matrix, KWDataZeros))
            self.curr_column = self.curr_column.reshape(-1, 1)
            self.adjacency_matrix = np.hstack((self.adjacency_matrix,self.curr_column))

        # def create_row(self, newPDFName):
        # dataInput_dataId = self.inputdatadb.create_row(self.data_am_index[-1])
        keywordList = DictToList(self.keyword_am_index)

        # def create_row(self, newFromDataID, newFrequency, newKeyword):
        # for kw in range(len(self.curr_column)):
        #     if self.curr_column[kw] != 0:
        #         self.keywordsdb.create_row(dataInput_dataId, int(self.curr_column[kw]), keywordList[kw])

        # reset for next file
        self.curr_new_keywords_added = 0


    def AppendRelationToDB(self, triplets):
        # print("STARTING: AppendRelation")
        
        for triple in triplets:
            head = triple['head']
            relation = triple['type']
            tail = triple['tail']

            self.connectivityMatrix.addToConnMatrix(head, tail)
            self.AddToCurrColumn(head, tail)

            # _, _, _ = self.driver.execute_query(
            # f"""
            #     MERGE (head:Node {{name: $head}})
            #     SET head.count = COALESCE(head.count, 0) + $headcount
            #     MERGE (tail:Node {{name: $tail}})
            #     SET tail.count = COALESCE(tail.count, 0) + $tailcount
            #     MERGE (head)-[r:{relation}]->(tail)
            #     SET r.count = COALESCE(r.count, 0) + $relcount
            # """, 
            # head=head, relation=relation, tail=tail,
            # headcount=self.local_nodes_counter[head], tailcount=self.local_nodes_counter[tail], relcount=self.local_relation_counter[(head, relation, tail)],
            # database_="neo4j")

        self.connectivityMatrix.storeCurrent(self.curr_inputdata)
        self.completeDataInputAdjacency()

    def SaveToCSV(self):
        print("save")
        currDate = getDate()
        uniqueFilename = giveUUID1()

        self.connectivityMatrix.updateStoredSizes()
        self.connectivityMatrix.saveToCSV(uniqueFilename, self.topic_name)

        # NOT USED
        filename = f'adjMatrixCSV/{currDate}-{uniqueFilename}.csv'

        keywordList = DictToList(self.keyword_am_index)

        with open(filename, mode='w+', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(['Keywords'] + self.data_am_index)
            for row in range(len(keywordList)):
                currRowData = [str(int(i)) for i in self.adjacency_matrix[row]]
                writer.writerow([keywordList[row]] + currRowData)

    def getAdjacencyList(self):
        return self.adjacency_matrix


if __name__ == '__main__':
    ag_pg55 = 'agriculturePDF/agriculture_usda_article_p55.pdf'
    ag_pg10 = 'agriculturePDF/agriculture_usda_article_p10.pdf'

    path = "research_papers/finished_papers"

    connection = Neo4jConnection("AgricultureData", uri='neo4j+s://5b656ba4.databases.neo4j.io:7687')
    allFiles = [f for f in listdir(path) if isfile(join(path, f))]
    research_papers = [(connection, join(path, f)) for f in allFiles]

    ag_articles = [f"agriculturePDF/agriculture_usda_article_p{i}.pdf" for i in range(1, 3)]
    ag_articles = [(connection, f) for f in ag_articles]
    def process_file(args):
        connection, file = args
        triplets = connection.GetRelations(file)
        connection.AppendRelationToDB(triplets)

    chosen_input = ag_articles

    # with Pool() as pool:
    #     list(tqdm(pool.imap_unordered(process_file, chosen_input), total=len(chosen_input)))

    for a in tqdm(ag_articles):
        con, aa = a
        process_file((connection, aa))

    print("DONE")
    connection.SaveToCSV()
    npArray = connection.getAdjacencyList()

    # a = DBKeywords()
    # b = DBInputData()
    # a._CLEARALL()
    # b._CLEARALL()
    
