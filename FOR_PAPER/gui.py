# @Author: Atul Sahay <atul>
# @Date:   2019-09-18T20:52:41+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2019-09-23T12:27:49+05:30



import kivy
import csv
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.core.window import Window

import random
import string

kivy.require("1.11.1")


class Query(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cols = 2  # used for our grid

        self.add_widget(Label(text='TAG', width=50))
        self.folder = TextInput(multiline=False, text='planter_bot') # hint_text="Enter your theme")
        self.add_widget(self.folder)

        self.add_widget(Label(text='TITLE', width=50))
        self.summary = TextInput(multiline=False, text='picam not working') # hint_text="Enter a one line summary")
        self.add_widget(self.summary)

        self.add_widget(Label(text='CONTENT', width=50))
        self.details = TextInput(multiline=True, text='the picam is throwing an error for which i have '
                                                      'attached image below') # hint_text="Enter your query")
        self.add_widget(self.details)

        self.post = Button(text="Post my Question")
        self.post.bind(on_press=self.post_query)
        self.add_widget(Label(width=50))  # just take up the spot.
        self.add_widget(self.post)

    @staticmethod
    def on_act(checkbox):
        return checkbox.text

    def post_query(self, instance):

        qid = self.generate_id()
        summary = self.summary.text
        details = self.details.text
        folder = self.folder.text


        header = ["QUERY_ID", "TITLE", "MAIN_CONTENT", "TAGS"]
        line = [qid, summary, details, folder]

        with open("query.tsv", 'wt') as query:
            tsv_writer = csv.writer(query, delimiter='\t')
            tsv_writer.writerow(header)
            tsv_writer.writerow(line)

        App.get_running_app().stop()
        Window.close()

    @staticmethod
    def generate_id(length=14):
        id = string.ascii_lowercase + string.digits
        return ''.join(random.choice(id) for i in range(length))

class Post(App):
    def build(self):
        return Query()
