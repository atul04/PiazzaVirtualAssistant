# @Author: Atul Sahay <atul>
# @Date:   2019-01-27T16:51:15+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2019-05-15T16:16:18+05:30



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

# from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
#
# caps = DesiredCapabilities().FIREFOX
# # caps["pageLoadStrategy"] = "normal"  #  complete
# caps["pageLoadStrategy"] = "eager"  #  interactive
# #caps["pageLoadStrategy"] = "none"

from selenium.common.exceptions import TimeoutException


#
# FileName = "AllPartiesLinkData.csv"
# ########## First Read the Csv file
# df = pd.read_csv(FileName)
# Parties = list(df)
#

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

feedList = []

for t in QuestionTitles:
    directChildrenList = t.getchildren() # returns all the direct children of the html object
    perFeedMap = {}
    print(directChildrenList[0].text_content()[2:])
    perFeedMap['HeaderTitle'] = directChildrenList[0].text_content()[2:]
    perContentList = []
    perFeedMap['Content'] = perContentList
    list_of_contents_in_feed = directChildrenList[1].getchildren()
    for eachContent in list_of_contents_in_feed:
        eachContentMap = {}
        # print(eachContent.attrib['id'])
        eachContentMap['id'] = eachContent.attrib['id']
        if( "unread" in list(eachContent.attrib['class'].split())):
            eachContentMap['unread'] = 1
        else:
            eachContentMap['unread'] = 0
        # print(eachContent.classes)
        feedChildrens = eachContent.getchildren()
        date = feedChildrens[0].getchildren()[0].text_content()
        # print(date)
        eachContentMap['date'] = date
        feedTitleRaw = list(feedChildrens[3].getchildren()[0].text_content().split('\n'))
        feedTitle = ''
        length = -1
        for eachText in feedTitleRaw:
            if(len(eachText)>length):
                feedTitle = eachText
                length = len(eachText)
        # print(feedTitle)
        eachContentMap['feedTitle'] =feedTitle



    # print(list_of_contents_in_feed)


# print(l)


print(QuestionFeeds)
for t in QuestionFeeds:
    print(t.attrib['id'])
    b = driver.find_elements_by_id(t.attrib['id'])
    print(b)
    b[0].click()


#
# for partyName in Parties:
#     # print(list(df[partyName]))
#     linksToCrawl = list(df[partyName])
#     whatsappLinks = []
#     print(partyName)
#     for link in linksToCrawl:
#         try:
#             driver.get(link)
#         except TimeoutException:
#             driver.execute_script("window.stop();")
#         soup = BeautifulSoup(driver.page_source, "html.parser")
#         print(link)
#         for r in soup.find_all("a", href=re.compile("chat.whatsapp.com")):
#             whatsappLinks.append(r['href'])
#             print(r['href'])
#         print(whatsappLinks)
#         print("-----------------------------")
#     modifiedPartyName = "_".join(partyName.split())
#     textFileName = "Links/"+modifiedPartyName+".txt"
#     if(len(whatsappLinks)==0):
#         continue
#     with(open(textFileName,'w')) as writeFile:
#         writeFile.write('\n'.join(whatsappLinks))
#     writeFile.close()

driver.quit()



# for i in list(df.iloc[:,0]):
#     print(i)
