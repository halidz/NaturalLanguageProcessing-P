# coding=utf8
import requests
import csv
from bs4 import BeautifulSoup

import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="123hal123",
    database="comments",
)

mycursor = mydb.cursor()

main_url = "https://www.gittigidiyor.com/"
notebook_url = "https://www.gittigidiyor.com/dizustu-laptop-notebook-bilgisayar"
page_url = "?sf="
page = 2

listinsert = []
list = []
productDict = {}
dictfull = {}

while page < 3:
    try:
        urlcomp = notebook_url + page_url + str(page)
        print(urlcomp)
        responser = requests.get(urlcomp)
        soap = BeautifulSoup(responser.text, "html.parser")
        i = 0
        for i in soap.findAll("li", {"class": "gg-uw-6 gg-w-8 gg-d-8 gg-t-8 gg-m-24 gg-mw-24 catalog-seem-cell"}):
            # print(i.find("a").get("href"))
            link = i.find("a").get("href")
            list.append(link)
            title = i.find("h3").get("title")
            productDict[link] = title

        page = page + 1

    except:
        print("cannot fetch")

print(list)
print(productDict)


count=len(list)
ind = 0
commentnumber=0
commentlist = []

while count > 0:
    urlcompx = main_url + list[ind]

    agentx = {"User-Agent": "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"}
    pagerx = requests.get(urlcompx, headers=agentx)
    commentlist.append(list[ind])

    realsoup = BeautifulSoup(pagerx.text, "html.parser")
    i = 0

    key = productDict.get(list[ind])
    counter = 1

    print(realsoup.findAll("li",{"class": "review-item"}))
    for newx in realsoup.findAll("li",{"class": "review-item"}):

        commentnumber=commentnumber+1
        ratingtemp = newx.find("div", {"class": "ratings active"})
        a, b = ratingtemp.get("style").split()
        c, d = b.split("%")
        rating = c
        commentlist.append(newx.find("p").text)
        if counter == 1:
            key=productDict.get(list[ind])
        else:
            key=key+str(counter)

        if int(c) > 60:
            opinion = "positive"
        else:
            opinion = "negative"

        listinsert.append((key, newx.find("p").text.encode("utf-8"), rating, opinion))
        dictfull[key]=newx.find("p").text.encode("utf-8")

        i=i+1
        counter=counter+1
    count = count - 1
    ind=ind+1




try:
    sql="INSERT INTO commenttableHepsid(productname,comment,rating,opinion) VALUES (%s,%s,%s,%s)"


    mycursor.executemany(sql, listinsert)

    mydb.commit()


    print(mycursor.rowcount, "was inserted.")
except:
    print("error while inserting")


