"""
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


from main import para_id
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
        self.textData = None  # Dataset
        
        # TensorFlow main session
        self.sess = None

        # Filename and directories constants
        self.SENTENCES_PREFIX = ['Q: ', 'A: ']

    def main(self, args=None):
        """
        Launch the training and/or the interactive mode
        """
        print('Welcome to FAQ Bot')
        print()
        print('TensorFlow detected: v{}'.format(tf.__version__))

        # Running session
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,  
            log_device_placement=False) 
        )  

        print('Initialize variables...')
        self.sess.run(tf.global_variables_initializer())


        self.mainTestInteractive(self.sess)
        


    def mainTestInteractive(self, sess):
        """ Try predicting the sentences that the user will enter in the console
        Args:
            sess: The current running session
        """
        print('Testing: Launch interactive mode:')
        print('')
        print('Welcome to the interactive mode, here you can ask to FAQbot the sentence you want. Type \'exit\' or just press ENTER to quit the program. Have fun.')

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
        
        data = pd.read_csv('data/dataset.csv')
        questions = data['Questions']                  
        question = np.full((data.shape[0]), question)
        Answers = data['Answers']
        question = pd.Series(question).astype(str)
        
        score = para_id.predict(questions,question)
        
        answer_id = np.argmax(score)
        score = score[answer_id]
        answer = Answers[answer_id]
        
        if score > 0.3:
            return answer
        return ''
