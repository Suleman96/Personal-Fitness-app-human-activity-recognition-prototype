from kivy.core.window import Window
from kivy.properties import StringProperty
from kivy.app import App
from kivy.uix.image import Image
#from kivymd.uix.textfield import MDTextFieldRound
from kivy.uix.button import Button
from kivy.metrics import dp
from kivymd.app import MDApp
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivy.uix.widget import Widget
#from sql_functions import write_user,write_movement_data,get_user_data_with_email,update_user_data_with_email,get_movement_data,get_tasks,check_if_email_unique
from kivy.uix.screenmanager import Screen, NoTransition, CardTransition, ScreenManager
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.behaviors import RoundedRectangularElevationBehavior
from kivymd.uix.button import MDRectangleFlatIconButton
#from kivymd.uix.textfield import MDTextFieldRect
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.input.providers.mouse import MouseMotionEvent
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
#from plyer import gravity,accelerometer,notification
from kivy.clock import Clock 
from kivy.uix.switch import Switch
import random
# import ai_model
import pandas as pd
import numpy as np
import matplotlib as tf
# from ai_model import prediction
import keras
#from matplotlib import backend as K
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers import BatchNormalization
from keras.regularizers import L1L2
from keras import optimizers
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from datetime import date, timedelta, datetime

#####################  SQL imports  ################################
import sqlite3
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref


# os.remove('trainer.db')
engine = create_engine('sqlite:///trainer.db', echo=True)
Base = declarative_base()

Session = sessionmaker(bind=engine)
session = Session()
#################################  SQL Tables ####################################
 
class User(Base):
    __tablename__ = "user"

    email = Column(String, primary_key=True)
    password = Column(String, nullable=False)
    name = Column(String)
    age = Column(Integer)
    weight = Column(Integer)
    uheight = Column(Integer)
    gender = Column(String)
    jp = Column(String)
    tasks = relationship('Tasks', backref="user", uselist=False)

    def __init__(self, email, password, name, age=0, weight=0,uheight=0,  gender=" ", jp=" "):
        self.email = email
        self.password = password
        self.name = name
        self.age = age
        self.weight = weight
        self.uheight = uheight
        self.gender = gender
        self.jp = jp
        
    def add_tasks(self, t1, t2, t3, t4, t5):
        self.tasks = Tasks(t1, t2, t3, t4, t5)

    def __repr__(self):
        return "<User(email='%s', name='%s', age='%d', weight='%d',uheight=%d gender='%d', task1='%s', task2='%s, task3='%s, task4='%s, task5='%s)>" % (self.email, self.name, self.age, self.weight, self.uheight, self.gender, self.tasks.task1, self.tasks.task2, self.tasks.task3, self.tasks.task4, self.tasks.task5)

class Tasks(Base):
    __tablename__ = "tasks"

    email = Column(String, ForeignKey(User.email), primary_key=True)
    task1 = Column(String, nullable=False)
    task2 = Column(String)
    task3 = Column(String)
    task4 = Column(String)
    task5 = Column(String)

    def __init__(self, task1, task2, task3, task4, task5):
        self.task1 = task1
        self.task2 = task2
        self.task3 = task3
        self.task4 = task4
        self.task5 = task5

class Motivation(Base):
    __tablename__ = "motivationtasks"

    email = Column(String, ForeignKey(User.email), primary_key=True, nullable=False)
    timestamp = Column(String, nullable=False, primary_key=True)
    mvmntprofile = Column(Integer)
    

    def __init__(self, email, timestamp, mvmntprofile):
        self.email = email
        self.timestamp = timestamp
        self.mvmntprofile = mvmntprofile
 
Base.metadata.create_all(engine)
########################## Motivation Tasks######################################################################

taskm18=["Run 20 mins","Walk 5 mins", "Run 20 mins","Sit 10 mins","Run 6 mins"]
taskm24=["Run 15 mins","Walk 10 mins", "Run 15 mins","Sit 15 mins","Run 4 mins"]
taskm40=["Run 5 mins","Walk 15 mins", "Run 5 mins","Sit 20 mins","Run 2 mins"]
taskf18=["Run 20 mins","Walk 5 mins", "Run 19 mins","Sit 9 mins","Run 3 mins"]
taskf24=["Run 14 mins","Walk 5 mins", "Run 14 mins", "Sit 14 mins","Run 2 mins"]
taskf40=["Run 4 mins", "Walk 5 mins","Run 4 mins","Sit 19 mins","Run 1 mins"]

###########################################################################################################

logged_in_user = None
logged_in_username = None
logged_in_user_age= 0
logged_in_weight = None
logged_in_user_gender=None
logged_in_height = None
logged_in_jp = None

g_time_in_movement= 0
g_sensor_reachable = False
g_number_of_notifications_8h = 4
Sensor_values = []
dirname = os.path.dirname(__file__)

############################### AI MODEL ###############################################################
ACTIVITIES = {
        0: 'sitting',
        1: 'walking',
        2: 'running',
    }

def loading_AI_model():
    model = keras.models.load_model('C:/Users/virtu/Desktop/Embedded/Personal Fitness App Project/model.h5')
    
    #model = keras.models.load_model('C:/Users/Lenovo/Desktop/DIT AI Lectures/MSS-M-1 Case Study Embedded Control Solutions (SS22)/FINAL Project/Models')
    return model

def predict(model, data):
    predicted_model = model.predict([Sensor_values]) 
    ai_values = ACTIVITIES[np.argmax(predicted_model)]
    return ai_values

m = loading_AI_model()
  ########################################################################################################

class LoginScreen(Screen):

    def loginBtn(self):
        email = ObjectProperty(None)
        password = ObjectProperty(None)
        email_input = self.email.text
        password_input = self.password.text
        print(email_input)
        print(password_input)
        valid = session.query(User).filter(
            User.email == email_input, User.password == password_input).count()
        if valid == 0 or (email_input == '' or password_input == ''):
            # if valid == 0:
            print("Invalid email or password")
        else:
            result = session.query(User).filter(
                User.email == email_input, User.password == password_input)
            for user in result:
                print(f"Welcome {user.name}!")
                global logged_in_user
                logged_in_user = user.email
                self.manager.current = "main"


class SignUpScreen(Screen):
    def registerBtn(self):
        email_input = self.email.text
        password_input = self.password.text
        password_confirmation_input = self.password_confirmation.text
        username_input = self.username.text
        age_input = self.age.text
        weight_input = self.weight.text
        height_input = self.uheight.text
        gender_input = self.gender.text
        jp_input = self.jp.text
        age_input=int(age_input)
        weight_input=int(weight_input)
        
        if gender_input != "male" or gender_input != "female":
            print("Password unmatch!!")
        if password_confirmation_input != password_input:
            print("Password unmatch!!")   
        
        else:
            data = User(email_input, password_input, username_input, age_input, weight_input,height_input, gender_input, jp_input)
            
            if jp_input== "Software Engineer":
                if age_input<18 and  gender_input== "female":
                    data.tasks = Tasks("Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins")
                elif age_input>=18 and  age_input<40 and  gender_input== "female":
                    data.tasks = Tasks("Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins")
                elif age_input>40 and  gender_input== "female":
                    data.tasks = Tasks("Run 20 mins", "Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins")
                elif age_input<18 and  gender_input== "male":
                    data.tasks = Tasks("Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins")
                elif age_input>=18 and  age_input<40 and  gender_input== "male":
                    data.tasks = Tasks("Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins")
                elif age_input>40 and  gender_input== "male":
                    data.tasks = Tasks("Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins")         
                    
            else:        
                if age_input<18 and  gender_input== "female":
                    data.tasks = Tasks(taskf18[0],taskf18[1],taskf18[2],taskf18[3],taskf18[4])
                elif age_input>=18 and  age_input<40 and  gender_input== "female":
                    data.tasks = Tasks(taskf24[0],taskf24[1],taskf24[2],taskf24[3],taskf24[4])
                elif age_input>40 and  gender_input== "female":
                    data.tasks = Tasks(taskf40[0],taskf40[1],taskf40[2],taskf40[3],taskf40[4])
                elif age_input<18 and  gender_input== "male":
                    data.tasks = Tasks(taskm18[0],taskm18[1],taskm18[2],taskm18[3],taskm18[4])
                elif age_input>=18 and  age_input<40 and  gender_input== "male":
                    data.tasks = Tasks(taskm24[0],taskm24[1],taskm24[2],taskm24[3],taskm24[4])
                elif age_input>40 and  gender_input== "male":
                    data.tasks = Tasks(taskm40[0],taskm40[1],taskm40[2],taskm40[3],taskm40[4])

            #data.tasks = Tasks('a','b','c','d','e')
            session.add(data)
            session.commit()

class TermsandCondition(Screen):
    pass

class MainScreen(Screen):

    event = ObjectProperty(None)
    event_notification = ObjectProperty(None)
    notification_interval = ObjectProperty(None)
    movement_profile = ObjectProperty(None)
    
    def on_pre_enter(self):
        self.callback()
        # self.get_info()

    def callback(self):
        global logged_in_user
        global logged_in_user_age
        global logged_in_user_gender
        global logged_in_jp
        
        result_user = session.query(User).filter(User.email == logged_in_user)
        
        for user in result_user:
            logged_in_user_age= user.age
            logged_in_user_gender= user.gender
            logged_in_username= user.name
            logged_in_jp= user.jp
        u_age=0
        u_gender=""
        u_jp=""
        mtasks=[]
        u_age=logged_in_user_age
        u_gender=logged_in_user_gender   
        u_jp= logged_in_jp
        
        
        if u_jp== "Software Engineer":
            if u_gender == "male":
                if u_age <18:
                    mtask= ["Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins"]
                if (u_age) > 18 and (u_age) < 40:
                    mtasks = ["Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins"]
                else:
                    mtasks = ["Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins"]
                    
            elif u_gender == "female":
                if u_age <18:
                    mtask= ["Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins"]
                if (u_age) > 18 and (u_age) < 40 :
                    mtasks = ["Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins"]
                else:
                    mtasks = ["Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins"]
        else:
            if u_gender == "male":
                if u_age <18:
                    mtask= ["Run 20 mins","Walk 5 mins", "Run 20 mins","Sit 10 mins","Run 6 mins"]
                if (u_age) > 18 and (u_age) < 40:
                    mtasks = ["Run 15 mins","Walk 10 mins", "Run 15 mins","Sit 15 mins","Run 4 mins"]
                else:
                    mtasks = ["Run 5 mins","Walk 15 mins", "Run 5 mins","Sit 20 mins","Run 2 mins"]
                    
            elif u_gender == "female":
                if u_age <18:
                    mtask= ["Run 5 mins","Walk 15 mins", "Run 5 mins","Sit 20 mins","Run 2 mins"]
                if (u_age) > 18 and (u_age) < 40 :
                    mtasks = ["Run 14 mins","Walk 5 mins", "Run 14 mins", "Sit 14 mins","Run 2 mins"]
                else:
                    mtasks = ["Run 4 mins", "Walk 5 mins","Run 4 mins","Sit 19 mins","Run 1 mins"]
                        
        self.m_task_one.text = mtasks[0]
        self.m_task_two.text = mtasks[1]
        self.m_task_three.text = mtasks[2]
        self.m_task_four.text = mtasks[3]
        self.m_task_five.text = mtasks[4]
        

        session.commit()

    ###########################   CHECK IN/OUT    ###############################################################   


    def AI_callback_get_movement_data(self,dt):
        
        global g_time_in_movement
        
        if g_sensor_reachable:
            Sensor_values.append([accelerometer.accelerometer[0],accelerometer.accelerometer[1],accelerometer.accelerometer[2]])
        else:
            Sensor_values.append([random.random(),random.random(),random.random()])
        print(self.movement_profile.text)
        if self.movement_profile.text not in ACTIVITIES.values():
            self.movement_profile.text = "Reading Sensor Values!"

        if (len(Sensor_values) == 12):        
            last_movement_profile = self.movement_profile.text
            mp = predict(m,Sensor_values)
            self.movement_profile.text = mp
            somethign  = Motivation(logged_in_user, datetime.now(), list(ACTIVITIES.values()).index(mp))
            print(list(ACTIVITIES.values()).index(mp))
            session.add(somethign)
            session.commit()
            
            #write_movement_data(get_user_data_with_email(self.email)["user_id"],datetime.now().replace(microsecond=0),ACTIVITIES[mp])
            Sensor_values.clear()
            
            if last_movement_profile == self.movement_profile.text:
                g_time_in_movement += 1
            else:
                g_time_in_movement = 0
        
            if g_time_in_movement == 3:
                if self.notification:
                    notification.notify(title='test', message=f'Hello {logged_in_username}, you are now {self.movement_profile.text} for 2h. let'+"'"+'s change it')
                    g_time_in_movement = 0
        
    def check_in_deactivate(self, value):
        if value == False:
            self.ids.check_in_button.disabled = True
            self.ids.check_out_button.disabled = False
            self.check_in_start_movement_analysis()
    
    def check_out_deactivate(self, value):
        if value == False:
            self.ids.check_in_button.disabled = False
            self.ids.check_out_button.disabled = True
            self.check_out_stop_movement_analysis()
            
    def check_in_start_movement_analysis(self):

  
        self.event = Clock.schedule_interval(self.AI_callback_get_movement_data, 1/12.)#8.)      
        # age = date.today().year - datetime.strptime(g_user_birthdate, '%d.%m.%Y').year

        # if self.movement_profile not in ['sitting','walking','runing']:
        #     sql_movement_profile = 0
        # else:
        #     sql_movement_profile = ACTIVITIES[self.movement_profile]



    def check_out_stop_movement_analysis(self):
        self.event.cancel()
        self.movement_profile.text = "Inactive!"



    def reminder(self):
    
        self.event_notification = Clock.schedule_once(self.callback_reminder, 2.)#8.)

    #########################################################################################################   

    # def callback(self):
    #     global logged_in_user
    #     result = session.query(Tasks).filter(User.email == logged_in_user)
    #     for tasks in result:
    #         self.m_task_one.text=tasks.task1
    #         self.m_task_two.text = tasks.task2
    #         self.m_task_three.text = tasks.task3
    #         self.m_task_four.text = tasks.task4
    #         self.m_task_five.text = tasks.task5 
    ##############            NOTIFICATION    ###############################################################   
    
    show_notif = True
    def notif_switch_click(self, switchObject, switchValue):
        # show_notif = True
        if(switchValue):
            self.ids.notification_label.text = "Notification ON"
            self.show_notif = True
            self.ids.checkbox_one.disabled = False
            self.ids.checkbox_two.disabled = False
            self.ids.checkbox_three.disabled = False
            self.ids.checkbox_four.disabled = False
            self.ids.checkbox_five.disabled = False
            # self.notification_ON_time()
        else:
            self.ids.notification_label.text = "Notification OFF"
            self.show_notif = False
            self.motiv_task_list = []
            self.ids.checkbox_one.active = False
            self.ids.checkbox_two.active = False
            self.ids.checkbox_three.active = False
            self.ids.checkbox_four.active = False
            self.ids.checkbox_five.active = False
            self.ids.checkbox_one.disabled = True
            self.ids.checkbox_two.disabled = True
            self.ids.checkbox_three.disabled = True
            self.ids.checkbox_four.disabled = True
            self.ids.checkbox_five.disabled = True
            #self.notification_OFF_time()           
             
    notification_interval= ObjectProperty(None)
            
    def notification_ON_time(self):
        self.notification_interval = Clock.schedule_interval(self, notification_popupreminder, 10)
        
    def notification_popupreminder(self):
        notification.notify(title= "Notification is Turned ON!", message= "Start work out!")

    def notification_OFF_time(self):
        self.pop_up_notification()
        self.notification_interval.cancel()
    
    
    # self.trigger1()
    # self.trigger1 = Clock.create_trigger(self.callback_reminder_task1, 10.)
    
    # def push_noticificaton(self,*args):
    #     plyer.notification.notify(title='GetFit!',
    #     message= '2 hour of working! Great! now Lets GetFit!',
    #     app_name='GetFit')

    # def stop_noticification(self):
    #     #print('noticifation stopped')
    #     self.notif_interval.cancel()
             
    def pop_up_notification(self):
        try:
            notification.notify(title='LETS GO!!!!', message="Start You Workout Now! ", timeout=10)
        except:
            self.show_notification_alert_dialog()
 

    def remind_me_notification(self):
        if checkbox_one.value == True:
            print("active baby")
        elif checkbox_one.value== False:
            print("you have not selected any options")
        
      
      
    list_of_mtasks = []
        # collect_the_text=[]        

    def checkbox_click(self,instance, value, mtask):
        if value== True:
            self.list_of_mtasks.append(mtask)
            # txt= ''
            # for x in self.list_of_mtasks:
            #     txt=f'{txt} {x}'
            # self.ids.remind_me_tasks.text= f'You Selected: {txt}'
            
        else:
            self.list_of_mtasks.remove(mtask)
            # txt= ''
            # for x in self.list_of_mtasks:
            #     txt=f'{txt} {x},'
            # self.ids.remind_me_tasks.text = f'You Selected: {txt}'
            # print("none")
        
    #  def notification_ON_time(self):
    #     self.notification_interval = Clock.schedule_interval(self, notification_popupreminder, 10)
        
    # def notification_popupreminder(self):
    #     plyer.notification.notify(title= " Notification is Turned ON!", message= "Start Work out!")   
        
    def reminder_remindme(self):
        txt= ''
        for x in self.list_of_mtasks:
            txt=f'{txt} {x},'
        notification.notify(title= " LETS GO!!", message= f"dont forget to do your tasks! {txt} ") 
        # Clock.schedule_interval(self, notification_popupreminder, 10)
        

                   

class SettingsScreen(Screen):
    pass

class GraphScreen(Screen):
 #   graph_label
    piechartimage = ObjectProperty(None)
    piechartlabel =ObjectProperty(None)


    def on_enter(self):
        self.DPiechart()

    
    def DPiechart(self): 
        
        
        self.piechartlabel.text = "Daily"

        piechartdata = []
        #comare timestamp, to be filtered
        result = session.query(Motivation).filter(Motivation.email == logged_in_user, Motivation.timestamp)
        #datetime.now()
        today = date.today()
        for res in result:
            graph_movementprofile= res.mvmntprofile
            graph_timestamp= res.timestamp
            List_day_of_record = graph_timestamp.split(' ')[0].split('-')
            day_of_record = date(int(List_day_of_record[0]), int(List_day_of_record[1]),int(List_day_of_record[2]))
            if today == day_of_record:
                piechartdata.append(str(graph_movementprofile))
                
        piechartdata_yaxis=[0,0,0]
        piechartdata_yaxis[0] = piechartdata.count('0')
        piechartdata_yaxis[1] = piechartdata.count('1')
        piechartdata_yaxis[2] = piechartdata.count('2')
        piechartdata_xaxis = ['Sitting', 'Walking', 'Running', ]
        plt.pie(piechartdata_yaxis, labels=piechartdata_xaxis,startangle= 90)
        plt.legend(title = "Daily Tasks:")
        # plt.show()
        plt.savefig('images/pie_chart_results.png')
        self.piechartimage.source = os.path.join(dirname, 'images', 'pie_chart_results.png')
        self.piechartimage.reload()
    
    # fix, ax = plt.subplots()
    # ax.bar(values_xaxis, values_yaxis)
    # ax.set_ylabel('minutes')
    # figure = plt.gcf()
    # figure.set_size_inches(3, 3)
    # plt.savefig('plot.png')
    # self.graph.source = os.path.join(dirname, 'plot.png')
    # self.graph.reload()

    # y = np.array([35, 25, 25, 15])
    # mylabels = [ graph_movementprofile[0], graph_movementprofile[1],  graph_movementprofile[2]]
    # plt.pie(y, labels = mylabels)
    # plt.show()

    def WPiechart(self): 
        
        
        self.piechartlabel.text = "Weekly"
        today = date.today()
        piechartdata = []
        #comare timestamp, to be filtered
        result = session.query(Motivation).filter(Motivation.email == logged_in_user, Motivation.timestamp)
        #datetime.now()
        startofweek= today - timedelta(days=today.weekday())
        endofweek = startofweek + timedelta(days=6)
        
        self.piechartlabel.text = f'{str(startofweek)} - {str(endofweek)}'
        
        for res in result:
            graph_movementprofile= res.mvmntprofile
            graph_timestamp= res.timestamp
            List_day_of_record = graph_timestamp.split(' ')[0].split('-')
            day_of_record = date(int(List_day_of_record[0]), int(List_day_of_record[1]),int(List_day_of_record[2]))
            if day_of_record == startofweek:
                print(today,day_of_record)
                piechartdata.append(str(graph_movementprofile))
                
                
        piechartdata_yaxis=[0,0,0]
        piechartdata_yaxis[0] = piechartdata.count('0')
        piechartdata_yaxis[1] = piechartdata.count('1')
        piechartdata_yaxis[2] = piechartdata.count('2')
        piechartdata_xaxis = ['Sitting', 'Walking', 'Running', ]
        plt.pie(piechartdata_yaxis, labels=piechartdata_xaxis,startangle= 90)
        plt.legend(title = "Weekly Tasks:")
        # plt.show()
        plt.savefig('images/pie_chart_results.png')
        self.piechartimage.source = os.path.join(dirname, 'images', 'pie_chart_results.png')
        self.piechartimage.reload()
        
    def MPiechart(self): 
        
        
        self.piechartlabel.text = "Monthly"
        today = date.today()
        piechartdata = []
        #comare timestamp, to be filtered
        result = session.query(Motivation).filter(Motivation.email == logged_in_user, Motivation.timestamp)
        #datetime.now()

        for res in result:
            graph_movementprofile= res.mvmntprofile
            graph_timestamp= res.timestamp
            List_day_of_record = graph_timestamp.split(' ')[0].split('-')
            day_of_record = date(int(List_day_of_record[0]), int(List_day_of_record[1]),int(List_day_of_record[2]))
            if today.month == day_of_record.month and today.year==day_of_record.year:
                piechartdata.append(str(graph_movementprofile))
                
        piechartdata_yaxis=[0,0,0]
        piechartdata_yaxis[0] = piechartdata.count('0')
        piechartdata_yaxis[1] = piechartdata.count('1')
        piechartdata_yaxis[2] = piechartdata.count('2')
        piechartdata_xaxis = ['Sitting', 'Walking', 'Running', ]
        plt.pie(piechartdata_yaxis, labels=piechartdata_xaxis,startangle= 90)
        plt.legend(title = "Monthly Tasks:")
        # plt.show()
        plt.savefig('images/pie_chart_results.png')
        self.piechartimage.source = os.path.join(dirname, 'images', 'pie_chart_results.png')
        self.piechartimage.reload()

    
class NotificationScreen(Screen):
    pass

class UserProfileScreen(Screen):

    # we need to get the user value from the database
    # current_user= session.query(User).filter()

    def on_pre_enter(self):
        self.callback()

    def callback(self):
        global logged_in_user
        result = session.query(User).filter(User.email == logged_in_user)
        for user in result:
            self.email_label.text=user.email
            self.username_label.text = user.name
            self.age_label.text = str(user.age)
            self.weight_label.text = str(user.weight)
            self.height_label.text = str(user.uheight)
            self.gender_label.text = user.gender
            self.jp_label.text = user.jp
            global logged_in_user_age
            logged_in_user= user.age    
            global logged_in_user_gender
            logged_in_user_gender= user.gender
            
class EditUserProfileScreen(Screen):

    def on_pre_enter(self):
        self.callback()

    def callback(self):
        global logged_in_user
        result = session.query(User).filter(User.email == logged_in_user)
        for user in result:
            self.chg_username.text = user.name
            self.chg_age.text = str(user.age)
            self.chg_weight.text = str(user.weight)
            self.chg_height.text = str (user.uheight)
            self.chg_gender.text = user.gender
            self.chg_jp.text = user.jp

    def savechanges(self):
        username_input = self.chg_username.text
        age_input = self.chg_age.text
        age_input = int(age_input)
        weight_input = self.chg_weight.text
        height_input = self.chg_height.text
        weight_input = int(weight_input)
        height_input = int(height_input)
        gender_input = self.chg_gender.text
        jp_input = self.chg_jp.text
        session.query(User).filter(User.email == logged_in_user).update(
            {'name': username_input, 'age': age_input, 'weight': weight_input, "uheight": height_input,
             'gender': gender_input, 'jp': jp_input})


        if jp_input== "Software Engineer":
            if age_input<18 and  gender_input== "female":
                data.tasks = Tasks("Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins")
            elif age_input>=18 and  age_input<40 and  gender_input== "female":
                data.tasks = Tasks("Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins")
            elif age_input>40 and  gender_input== "female":
                data.tasks = Tasks("Run 20 mins", "Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins")
            elif age_input<18 and  gender_input== "male":
                data.tasks = Tasks("Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins")
            elif age_input>=18 and  age_input<40 and  gender_input== "male":
                data.tasks = Tasks("Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins")
            elif age_input>40 and  gender_input== "male":
                data.tasks = Tasks("Run 20 mins","Walk 5 mins","Sit 9 mins","Walk 5 mins","Run 20 mins")  

        else:  
            if age_input<18 and  gender_input== "female" :
                User.tasks = Tasks(taskf18[0],taskf18[1],taskf18[2],taskf18[3],taskf18[4])
            elif age_input>=18 and  age_input<40 and  gender_input== "female":
                User.tasks = Tasks(taskf24[0],taskf24[1],taskf24[2],taskf24[3],taskf24[4])
            elif age_input>40 and  gender_input== "female":
                User.tasks = Tasks(taskf40[0],taskf40[1],taskf40[2],taskf40[3],taskf40[4])
            elif age_input<18 and  gender_input== "male":
                User.tasks = Tasks(taskm18[0],taskm18[1],taskm18[2],taskm18[3],taskm18[4])
            elif age_input>=18 and  age_input<40 and  gender_input== "male":
                User.tasks = Tasks(taskm24[0],taskm24[1],taskm24[2],taskm24[3],taskm24[4])
            elif age_input>40 and  gender_input== "male":
                User.tasks = Tasks(taskm40[0],taskm40[1],taskm40[2],taskm40[3],taskm40[4])
                
        session.commit()
        self.manager.current = "settings"

class SystemSettingsScreen(Screen):
    sensorschecks = []
    def sensorcheckbox_click(self,instance, value, sensors): #value will pass true r pass 
        if value== True:
            SystemSettingsScreen.sensorschecks.append(sensors)
            sen = ''
            for x in self.sensorchecks:
                sen = f'{sen}{x}'
            self.ids.sensorselectiontext.text=f'{sen} has been Activated!'
        else:
            SystemSettingsScreen.sensorschecks.remove(sensors)
            sen = ''
            for x in self.sensorchecks:
                sen = f'{sen}{x}'
                self.ids.sensorselectiontext.text=f'{sen} has been Activated!'

    

sm = ScreenManager()
# sm.add_widget(MDScreen(name="md"))
sm.add_widget(LoginScreen(name="login"))
sm.add_widget(TermsandCondition(name="termsandcondition"))
sm.add_widget(SignUpScreen(name="signup"))
sm.add_widget(MainScreen(name="main"))
sm.add_widget(SettingsScreen(name="settings"))
sm.add_widget(GraphScreen(name="graph"))
sm.add_widget(NotificationScreen(name="notify"))
sm.add_widget(UserProfileScreen(name="userprofile"))
sm.add_widget(EditUserProfileScreen(name="edituserprofile"))
sm.add_widget(SystemSettingsScreen(name="systemsettings"))



Window.size = (320, 550)



# GUI= Builder.load_file("app.kv") #Make Sure this is after all  definitions!
class MainApp(MDApp):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        try:
            accelerometer.enable() # enable the accelerometer
            global g_sensor_reachable
            g_sensor_reachable = True
        except:
            print ("No Accelerometer Detected!")
            
    def build(self):
        #builder = Bzuilder.load_file("app.kv")
        # return #builder
        pass


if __name__ == '__main__':
    MainApp().run()