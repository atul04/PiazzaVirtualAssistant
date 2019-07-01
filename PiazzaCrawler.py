# @Author: Atul Sahay <atul>
# @Date:   2019-01-27T16:51:15+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2019-07-01T18:35:32+05:30



import time
import requests
from urllib.parse import quote_plus
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
import re
import numpy
import pandas as pd
from bs4 import BeautifulSoup
import lxml.html

from selenium.common.exceptions import TimeoutException

import csv

######### Need to wake up the webdriver
path = 'geckodriver-v0.23.0-linux64/geckodriver'

profile = webdriver.FirefoxProfile()
profile.set_preference("webdriver.load.strategy", "unstable")
profile.update_preferences()

driver = webdriver.Firefox(firefox_binary=FirefoxBinary(),executable_path=path,firefox_profile=profile)
t = time.time()
driver.set_page_load_timeout(100)

driver.get("https://piazza.com/")
login_button = driver.find_elements_by_xpath("//button[@id='login_button']")
print(login_button)
usernameStr = 'smitagh@gmail.com'#input("username: ")
passwdStr   = 'howdy&*me'#input("password: ")
login_button[0].click()
username = driver.find_element_by_id('email_field')
username.send_keys(usernameStr)
password = driver.find_element_by_id('password_field')
password.send_keys(passwdStr)
login_button = driver.find_elements_by_id('modal_login_button')
login_button[0].click()
soup = BeautifulSoup(driver.page_source, "html.parser")
tree = lxml.html.fromstring(driver.page_source)
click_button_drop_down_list = driver.find_elements_by_id('network_button')
if(click_button_drop_down_list is not None):
    click_button_drop_down_list[0].click()
time.sleep(3)
click_button_toggle_button = driver.find_element_by_id('inactive_network_toggle')
if(click_button_toggle_button is not None):
    click_button_toggle_button.click()
time.sleep(3)

# Now we will see the track List to traverse
tree = lxml.html.fromstring(driver.page_source)
trackList = tree.findall('.//ul[@id="my_classes"]')
trackList = trackList[0].getchildren()
print("----------------------------Track List----------------------\n")
for num,i in enumerate(trackList[:-1]):
    # print(i)
    print('[{}]'.format(num),i.text_content(),sep="  ")
    print()
track_to_traverse = int(input("Select One : "))
fileName = "_".join(trackList[track_to_traverse].text_content().split())
print(fileName)
b = driver.find_elements_by_id(trackList[track_to_traverse].attrib['id'])
b[0].click()
time.sleep(5)
### Trying to find all the class feed item feeds
QuestionFeeds = tree.xpath('//*[@class="feed_item clearfix note "]')
loadMoreButton = driver.find_elements_by_id('loadMoreButton')
print(loadMoreButton)
if(loadMoreButton is not None):
    loadMoreButton[0].click()


time.sleep(5) #till loading the next contents
tree = lxml.html.fromstring(driver.page_source) #new content
QuestionTitles = tree.findall('.//div[@class="question_group"]')
print(len(QuestionTitles))

#Consisting of all higher headers such as pinned
feedList = []


# Writing to the csv files ( particularly 2 READ and UNREAD )
with open(fileName+".csv",mode='w') as file:
    writer = csv.writer(file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['ID','HEADER','TITLE','MAIN_CONTENT','DATE','TAGS'])
    try:
        for t in QuestionTitles:
            directChildrenList = t.getchildren() # returns all the direct children of the html object
            #particular feed such as pinned details
            perFeedMap = {}
            print(directChildrenList[0].text_content()[2:])
            perFeedMap['HeaderTitle'] = directChildrenList[0].text_content()[2:]
            className = list(directChildrenList[0].attrib['class'].split())
            if(className[0] == 'start_closed'): # if Header not open then open it
                b = driver.find_elements_by_id(directChildrenList[0].attrib['id'])
                b[0].click()
                time.sleep(5)

            perContentList = []
            perFeedMap['Content'] = perContentList
            list_of_contents_in_feed = directChildrenList[1].getchildren()
            feedHeader = perFeedMap['HeaderTitle'] #TO WRITE INTO CSV
            for eachContent in list_of_contents_in_feed:
                eachContentMap = {}
                eachContentMap['id'] = eachContent.attrib['id']
                qId = eachContentMap['id']
                if( "unread" in list(eachContent.attrib['class'].split())):
                    eachContentMap['unread'] = 1
                else:
                    eachContentMap['unread'] = 0
                b = driver.find_elements_by_id(qId)
                b[0].click()
                time.sleep(5)
                tree = lxml.html.fromstring(driver.page_source) #new content
                tagSet = tree.xpath('//span[@data-pats="folders_item"]/a[@class="tag folder"]')
                tagList = []
                for eachTag in tagSet:
                    tagList.append(eachTag.text_content())
                print(tagList)
                eachContentMap['tags']=','.join(tagList)
                questionText = tree.xpath('//div[@id="questionText" and @class="post_region_text"]')[0].text_content()
                eachContentMap['main_content'] = questionText


                questionChildrens = eachContent.getchildren()
                date = questionChildrens[0].getchildren()[0].text_content()
                eachContentMap['date'] = date
                questionTitleRaw = list(questionChildrens[3].getchildren()[0].text_content().split('\n'))
                quesstionTitle = ''
                length = -1
                for eachText in questionTitleRaw:
                    if(len(eachText)>length):
                        questionTitle = eachText
                        length = len(eachText)
                eachContentMap['Title'] = questionTitle

                qId    = eachContentMap['id']
                qTitle = eachContentMap['Title']
                qDate  = eachContentMap['date']
                # if(eachContentMap['unread'] == 0)
                qTags    = eachContentMap['tags']
                qContent = eachContentMap['main_content']
                writer.writerow([qId,feedHeader,qTitle,qContent,qDate,qTags])
                # else:
                    # print("unread\n")

            # feedList.append(perFeedMap)
    except:
        print("error")

#
# for t in QuestionTitles:
#     directChildrenList = t.getchildren() # returns all the direct children of the html object
#     #particular feed such as pinned details
#     perFeedMap = {}
#     print(directChildrenList[0].text_content()[2:])
#     perFeedMap['HeaderTitle'] = directChildrenList[0].text_content()[2:]
#     className = list(directChildrenList[0].attrib['class'].split())
#     if(className[0] == 'start_closed'): # if Header not open then open it
#         b = driver.find_elements_by_id(directChildrenList[0].attrib['id'])
#         b[0].click()
#         time.sleep(5)
#     perContentList = []
#     perFeedMap['Content'] = perContentList
#     list_of_contents_in_feed = directChildrenList[1].getchildren()
#     for eachContent in list_of_contents_in_feed:
#         eachContentMap = {}
#         eachContentMap['id'] = eachContent.attrib['id']
#         qId = eachContentMap['id']
#         if( "unread" in list(eachContent.attrib['class'].split())):
#             eachContentMap['unread'] = 1
#         else:
#             eachContentMap['unread'] = 0
#             b = driver.find_elements_by_id(qId)
#             b[0].click()
#             time.sleep(5)
#             tree = lxml.html.fromstring(driver.page_source) #new content
#             tagSet = tree.xpath('//span[@data-pats="folders_item"]/a[@class="tag folder"]')
#             tagList = []
#             for eachTag in tagSet:
#                 tagList.append(eachTag.text_content())
#             print(tagList)
#             eachContentMap['tags']=','.join(tagList)
#             questionText = tree.xpath('//div[@id="questionText"]')[0].text_content()
#             eachContentMap['main_content'] = questionText
#
#
#         questionChildrens = eachContent.getchildren()
#         date = questionChildrens[0].getchildren()[0].text_content()
#         eachContentMap['date'] = date
#         questionTitleRaw = list(questionChildrens[3].getchildren()[0].text_content().split('\n'))
#         quesstionTitle = ''
#         length = -1
#         for eachText in questionTitleRaw:
#             if(len(eachText)>length):
#                 questionTitle = eachText
#                 length = len(eachText)
#         eachContentMap['Title'] = questionTitle
#         perContentList.append(eachContentMap)
#
#
#
#
#
# # Writing to the csv files ( particularly 2 READ and UNREAD )
# with open("PIAZZA_read.csv",mode='w') as file:
#     writer = csv.writer(file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['ID','HEADER','TITLE','MAIN_CONTENT','DATE','TAGS'])
#     for perFeed in feedList:
#         questionList = perFeed['Content']
#         feedHeader = perFeed['HeaderTitle']
#         for perQuestion in questionList:
#             qId    = perQuestion['id']
#             qTitle = perQuestion['Title']
#             qDate  = perQuestion['date']
#             if(perQuestion['unread'] == 0):
#                 qTags    = perQuestion['tags']
#                 qContent = perQuestion['main_content']
#             else:
#                 print("unread\n")
#             writer.writerow([qId,feedHeader,qTitle,qContent,qDate,qTags])
#
#
driver.quit()
