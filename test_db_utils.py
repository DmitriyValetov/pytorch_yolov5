from sqlalchemy import Table, Column, Integer, ForeignKey, String, Float, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func
from sqlalchemy.exc import OperationalError, IntegrityError

import os
import json


class DBManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.Base = declarative_base()

        class Detections(self.Base):
            __tablename__ = 'Detections'
            
            id = Column(Integer, primary_key=True, autoincrement=True)
            file_path = Column(String)
            detected = Column(String)

            def __str__(self):
                return str(self.toDict())

            def __repr__(self):
                return str(self.toDict())


            def toDict(self):
                return {
                    'file_path': self.file_path,
                    'detected': json.loads(self.detected),
                }

        self.Detections = Detections

        engine = self.get_engine()
        self.Base.metadata.create_all(engine)


    def get_engine(self, echo=False):
        engine = create_engine('sqlite:///{}'.format(self.db_path), echo=echo)
        return engine
    

    def _closeConnection(self, session, engine):
        try:
            session.close()
        except:
            pass

        try:
            engine.dispose()
        except:
            pass


    def add_detections(self, file_path, detected):
        try:
            engine = self.get_engine()
            Session = sessionmaker(bind=engine)
            session = Session()

            new_row = self.Detections(
                file_path=file_path,
                detected=json.dumps(detected),
            )

            session.add(new_row)
            session.commit()
        
        except Exception as e:
            print(f"[dbManager.addDetections] Error: {e}")

        finally:
            self._closeConnection(session, engine)

    def get_all_detections(self):
        try:
            engine = self.get_engine()
            Session = sessionmaker(bind=engine)
            session = Session()
            
            detections = session.query(self.Detections).all()
            detections = [d.toDict() for d in detections]
        
        
        except Exception as e:
            raise e

        finally:
            self._closeConnection(session, engine)

        return detections
    
    
    def get_detections_for_file(self, file_path):
        try:
            engine = self.get_engine()
            Session = sessionmaker(bind=engine)
            session = Session()
            
            detections = session.query(self.Detections).filter(self.Detections.file_path == file_path).all()
            detections = [d.toDict() for d in detections]
        
        
        except Exception as e:
            raise e

        finally:
            self._closeConnection(session, engine)

        return detections
    
    

            
# db = dbManager('test.sqlite')
# db.add_detections('test_path_name', {'data': 'foo'})
# db.get_all_detections()
# db.get_detections_for_file('test_path_name')
# db.get_detections_for_file('unk')