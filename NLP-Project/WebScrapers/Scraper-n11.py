# coding=utf-8
import requests
import csv
import string
from bs4 import BeautifulSoup

import mysql.connector
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="123hal123",
  database="comments",
)

mycursor = mydb.cursor()

main_url="https://www.n11.com/"
notebook_url="https://www.n11.com/bilgisayar/dizustu-bilgisayar/"
page_url="?pg="
page=2

list =[]
listinsert=[]
dict = {}
dictfull ={}

while page < 3:
    try:
        urlcomp=notebook_url+page_url+str(page)
        responser=requests.get(urlcomp)
        soup =BeautifulSoup(responser.text,"html.parser")
        i=0

        for url in soup.findAll("li",{"class":"column"}):

            list.append(url.find("a").get("href"))
            link=url.find("a").get("href")
            title=url.find("a").get("title")
            dict[link]=title
        page=page+1

    except:
        print("cannot fetch")
print("LİST"+"\n"+str(list))

count=len(list)
ind = 0
commentnumber=0
commentlist = []
keys=dict.keys()

while count > 0:

    print("\n" + list[ind] + "\n")
    commentlist.append(list[ind])
    responser = requests.get(list[ind])
    realsoup = BeautifulSoup(responser.text, "html.parser")
    i = 0
    key = dict.get(list[ind])
    counter = 1




    for newx in realsoup.findAll("li", {"class": "comment"}):
        commentnumber=commentnumber+1
        ratingtemp=newx.find("span").get("class")
        a,b =str(ratingtemp).split()
        c,d = b[2:].split("'")
        rating=str(c)
        commentlist.append(newx.find("p").text)
        if counter == 1:
            key =dict.get(list[ind])
        else:
            key=key+str(counter)
        dictfull["link"]=key
        dictfull["comment"]=newx.find("p").text.encode("utf-8")
        dictfull["rating"]=rating
        if int(c) > 60:
            opinion="positive"
        else:
            opinion="negative"
        listinsert.append((key,newx.find("p").text.encode("utf-8"),rating,opinion))

        i=i+1
        counter = counter + 1
    count = count - 1
    ind=ind+1


sql="INSERT INTO commenttableN11lx(productname,comment,rating,opinion) VALUES (%s,%s,%s,%s)"


mycursor.executemany(sql, listinsert)

mydb.commit()

print(mycursor.rowcount, "was inserted.")