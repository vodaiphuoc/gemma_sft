from pydantic import TypeAdapter
from loguru import logger
import sqlite3
import json
from typing import List, Tuple, Literal, Union, Dict
import re

class ChatHistoryDB(object):
    """
    Database for storage files content
    """
    def __init__(self, db_url = 'db/chat_history.db') -> None:
        try:
            self.connection = sqlite3.connect(db_url)
            
        except sqlite3.Error as error:
            logger.error(f"Cannot connect to {db_url} db, ", error)

        try:
            with self.connection:
                create_prompt = """
CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY, Topic TEXT, Role TEXT, Content TEXT, Timestamp TEXT);
""" 
                self.connection.execute(create_prompt)
        except sqlite3.Error as error:
            logger.error('Cannot create table', error)

    def insert_new_turns(self,
                    topic: str,
                    new_msgs: List[Dict[str, str]]
                    )->None:
        """
        Insert new user and model reponse into db
        Args:
            - topic (str): topic of current conversation
            - new_msgs (List[Dict[str, str]]): new data of conversation
        
        Example:
            - topic = 'Hoi An'
            - new_msgs = [
                {"role": "user", "content": "Hello", 'timestamp': '20-01-2025'},
                {"role": "model", "content": "Great to meet you", 'timestamp': '20-01-2025'}
            ]
        """
        prepare_data = [(topic, msg['role'], msg['content'], msg['timestamp']) for msg in new_msgs]
        try:
            with self.connection:
                insert_prompt = """
                    INSERT INTO chat_history (Topic, Role, Content, Timestamp) VALUES (?,?,?,?);
                """ 
                self.connection.executemany(insert_prompt, prepare_data)

        except Exception as error:
            logger.error(f"Cannot peform insert many files, error: ", error)

    def close(self):
        self.connection.close()

    def get_topics(self)->List[str]:
        try:
            with self.connection:
                query_prompt = f"""SELECT Topic FROM chat_history"""
                records = self.connection.execute(query_prompt).fetchall()
                return list(set([ele[0] for ele in records]))
        
        except Exception as error:
            logger.error(f"Cannot load topics, error: ", error)

    def get_chat_history(self, topic:str)->List[Dict[str,str]]:
        try:
            with self.connection:
                query_prompt = f"""SELECT Role, Content, Timestamp FROM chat_history WHERE Topic LIKE '%{topic}';"""
                records = self.connection.execute(query_prompt).fetchall()
                return [{
                    'role': _rec[0],
                    'content': _rec[1],
                    'timestamp': _rec[2]
                } 
                for _rec in records
                ]
        
        except Exception as error:
            logger.error(f"Cannot load history with topic {topic} error: ", error)        
