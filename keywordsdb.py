from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.exc import SQLAlchemyError
from mysqlconnection import DBSession, Base

from utils import getDate

class Keywords(Base):
    __tablename__ = 'Keywords'

    keywordID = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    fromDataID = Column(Integer, ForeignKey('InputData.dataID'), nullable=False)
    frequency = Column(Integer)
    keyword = Column(String(255))

class DBKeywords():

    def __init__(self):
        self.date = getDate()

    def create_row(self, newFromDataID, newFrequency, newKeyword):
        try:
            DBSession.begin()

            newKeyword = Keywords(fromDataID=newFromDataID, frequency=newFrequency, keyword=newKeyword)

            DBSession.add(newKeyword)
            DBSession.commit()

            return newKeyword
        except SQLAlchemyError as e:
            DBSession.rollback()
            print(f"ERROR ({__name__}): {e}")

    def update_frequency(self, uKeywordID, uFromDataID, uFrequency):
        try:
            DBSession.begin()

            keywordToUpdate = DBSession.query(Keywords).filter_by(KeywordID=uKeywordID, FromDataID=uFromDataID).get(1)
            keywordToUpdate.frequency = uFrequency

            DBSession.commit()
        except SQLAlchemyError as e:
            DBSession.rollback()
            print(f"ERROR ({__name__}): {e}")

    def _CLEARALL(self):
        try:
            DBSession.begin()
            table = Keywords.__table__
            DBSession.execute(table.delete())
            DBSession.commit()
            print(f"SUCCESS ({__name__}): Deleted Full Table")
        except SQLAlchemyError as e:
            DBSession.rollback()
            print(f"ERROR ({__name__}): {e}")


