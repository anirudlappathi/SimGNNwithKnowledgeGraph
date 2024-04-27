from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from dotenv import load_dotenv
import os
load_dotenv()

"""
TO MAKE:
turn the dict's to jsons so I can store the current data in the db for easy access in the 
"""


# REMOVE DUPLICATE NAMES
"""
MATCH (n:Node)
WITH n.name AS nodeName, collect(n) AS nodes
WHERE size(nodes) > 1
FOREACH (n IN tail(nodes) |
    SET n.name = nodeName
    REMOVE n:Node
    MERGE (fn:Node {name: nodeName})
    SET firstNode += properties(n)
    DELETE n
)
"""

# Show All
"""
Match (n)-[r]->(m)
Return n,r,m
"""

# Delete All
"""
MATCH (n)-[r]-(m)
DELETE n, r, m
"""


'''
create a dict containing each seen head type tail so we can calculate the frequency of seen relations

create embeddings for everything to see distance between vectors to see normalized similarity

if you remove a graph node, check the performance ability of the entire model to see the importance or connectivity of the node

all text is converted by a number and cosine function
'''

# taken from 
# https://towardsdatascience.com/integrate-llm-workflows-with-knowledge-graph-using-neo4j-and-apoc-27ef7e9900a2

# self.system_prompt = """
#     You are an assistant that helps to generate text to form nice and human understandable answers based.
#     The latest prompt contains the information, and you need to generate a human readable response based on the given information.
#     Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
#     Do not add any additional information that is not explicitly provided in the latest prompt.
#     I repeat, do not add any information that is not explicitly given.
# """

# self.open_ai_key = os.getenv('OPEN_AI_KEY')
# self.embedding_model = SentenceTransformer("GIST-Embedding-v0")

# https://medium.com/@shubhkarmanrathore/mastering-crud-operations-with-sqlalchemy-a-comprehensive-guide-a05cf70e5dea

# try:
#     # Begin a transaction
#     session.begin()

#     # Update user data
#     user.email = "updated_email@example.com"
#     session.commit()  # Commit the transaction
# except:
#     session.rollback()  # Rollback if an error occurs

db_username = 'root'
db_password = os.getenv('MYSQL_PASSWORD')
db_name = 'knowledgebase'

DATABASE_URL = f'mysql+mysqlconnector://{db_username}:{db_password}@localhost/{db_name}'

engine = create_engine(DATABASE_URL, echo=True)

Session = sessionmaker(bind=engine)

DBSession = Session()
Base = declarative_base()