
Main script. See README.md for more information

Use python 3
"""
import pandas as pd
import argparse  # Command line parsing
import configparser  # Saving the models parameters
import datetime  # Chronometer
import os  # Files management
import tensorflow as tf
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm  # Progress bar
from tensorflow.python import debug as tf_debug

from chatbot.textdata import TextData
from chatbot.model import Model
from chatbot import paraid
import gensim

class Chatbot:
    """
    Main class which launch the training or testing mode
    """

    class TestMode:
        """ Simple structure representing the different testing modes
        """
        ALL = 'all'
        INTERACTIVE = 'interactive'  # The user can write his own questions
        DAEMON = 'daemon'  # The chatbot runs on background and can regularly be called to predict something

    def __init__(self):
        """
        """
        # Model/dataset parameters
        self.args = None

        # Task specific object
        self.data = 'data/dataset.csv   # Dataset
        self.model = None  # MA_LSTM model
        self


    @staticmethod

    def mainTest(self, sess):
        """ Try predicting the sentences that the user will enter in the console
        Args:
            sess: The current running session
        """
        # TODO: If verbose mode, also show similar sentences from the training set with the same words (include in mainTest also)
        # TODO: Also show the top 10 most likely predictions for each predicted output (when verbose mode)
        # TODO: Log the questions asked for latter re-use (merge with test/samples.txt)

        print('Testing: Launch interactive mode:')
        print('')
        print('Welcome to the interactive mode, here you can ask to Deep Q&A the sentence you want. Don\'t have high '
              'expectation. Type \'exit\' or just press ENTER to quit the program. Have fun.')

        while True:
            question = input(self.SENTENCES_PREFIX[0])
            if question == '' or question == 'exit':
                break

            questionSeq = []  # Will be contain the question as seen by the encoder
            #answer = self.singlePredict(question, questionSeq)
            answer_predict = self.tfidf_predict(question)
            if answer_predict != '':
                print(answer_predict)
                continue
            else:
                print('I do not know')
            
    def tfidf_predict(self, question):
        
        database = pd.Series([
                            " I missed to apply Out of Office (OOO), can I apply now? ",
                             "I have applied OOO still my LWP has been applied in the system.",
                            "Can I take leaves during notice period?",
                            "My LWP has been applied, can I apply for LWP reversal now?",
                            " My salary has been deducted for this month, why?",
                            " I was on Business Trip, why my leave without pay has been applied?",
                            "What is the current attendance or pay cycle?",
                            "Where can I check new policy?",
                            "My manager is on leave, how can I get my leave approve? ",
                            "I have applied leave for wrong date, please correct it?",
                            "Good Morning",
                             "Hi",
                            "What is LWP",
                            "What is ooo", 
                             "What is out of office"]).astype(str)
        Answers = ["Please note that Out of Office (OOO) needs to be applied within 2 days of the applicable date. In case you have not applied, kindly coordinate with your supervisor. He can fill OOO on your behalf.All Out of Office requests need to be approved before attendance cut-off date to avoid unpaid leave.If your date for OOO request is beyond current attendance cycle, you cannot apply OOO."
        ,"This is possible if even after applying Out of Office, your weekly InOut Time is less than 45 hours. If that’s not the case, try checking if the OOO has been applied for the correct date.",
          "You are advised to avoid taking leaves during this period to ensure smooth knowledge transfer. In case of emergencies, you should utilize the remaining Sick Leave Balance.",
          "The current attendance cycle is closed. We are sorry we would not be able to consider your request for now.",
          "There are 2 possible reasons: a.    Missed to apply OOO/Leave/ Comp Off in system before attendance cut-off date. b.    Attendance shortfall in a particular week of the attendance cycle.",
          "Please check whether you have applied BT in HR Management portal & it has been approved by your manager.If it is approved and you still have a query, raise it on Bulletin Board using the below path: Knox Portal-> Samsung Research->Intro->Bulletin Board->Suggestions",
          "Kindly check the path: HR Mgmt->AIMS->Attendance Report - > Payroll",
          "All the policies are uploaded on the path or link mentioned below. You can search for the desired policy from the list of policies.KNOX Portal->Samsung Research->Intro->Policies & Templates.Link",
          "You can cancel your leave application & re-submit it by putting Reviewer 2 in the approval path.In case Reviewer 2 is not available, raise the query on Bulletin Board using the below path:Knox Portal-> Samsung Research->Intro->Bulletin Board->Suggestions",
          " If the applied leave has not been approved, you can cancel it and apply again. If the leave has been approved, raise the query on Bulletin Board using the below path for correction:Knox Portal-> Samsung Research->Intro->Bulletin Board->Suggestions"
          ,"A very Good Morning","Hi!", "LWP is", "ooo", "ooo"]
        
        question = np.full((15), question)
        
        question = pd.Series(question).astype(str)
        
        score = paraid.predict(database,question)
        
        answer_id = np.argmax(score)
        score = score[answer_id]
        print(answer_id)
        print(score)
        answer = Answers[answer_id]
        
        
        if score > 0.3:
            return answer
        return ''
