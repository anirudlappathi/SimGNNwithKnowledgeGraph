from sqlalchemy import Column, Integer, String
from sqlalchemy.exc import SQLAlchemyError
from mysqlconnection import DBSession, Base

from utils import getDate

class InputData(Base):
    __tablename__ = 'InputData'

    dataID = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    pdfName = Column(String(50))
    dateAdded = Column(String(6))

class DBInputData():

    def __init__(self):
        self.date = getDate()

    def create_row(self, newPDFName):
        try:
            DBSession.begin()

            newInputData = InputData(pdfName=newPDFName, dateAdded=self.date)

            DBSession.add(newInputData)
            DBSession.commit()

            return newInputData.dataID
        except SQLAlchemyError as e:

            DBSession.rollback()
            print(f"ERROR (DBInputData.create_row): {e}")

    def _CLEARALL(self):
        try:
            DBSession.begin()
            table = InputData.__table__
            DBSession.execute(table.delete())
            DBSession.commit()
            print(f"SUCCESS ({__name__}): Deleted Full Table")
        except SQLAlchemyError as e:
            DBSession.rollback()
            print(f"ERROR ({__name__}): {e}")