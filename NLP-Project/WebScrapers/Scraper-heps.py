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

main_url = "https://www.hepsiburada.com/"
notebook_url = "https://www.hepsiburada.com/masaustu-bilgisayarlar-c-34"
page_url = "?sayfa="
page = 2
listinsert = []
list = []
productDict = {}
dictfull = {}


try:
    urlcomp ="https://www.hepsiburada.com/laptop-notebook-dizustu-bilgisayarlar-c-98"

    agent = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"}
    pager = requests.get(urlcomp, headers=agent)
    soap = BeautifulSoup(pager.text, "html.parser")
    i = 0
    for i in soap.findAll("li", {"class": "search-item col lg-1 md-1 sm-1 custom-hover not-fashion-flex"}):
        link = i.find("a").get("href")
        list.append(link)
        title = i.find("h3").get("title")
        productDict[link] = title

    page = page + 1

except:
    print("cannot fetch")

count = len(list)
ind = 0
commentnumber = 0
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


    for newx in realsoup.findAll("li", {"class": "review-item"}):
        commentnumber = commentnumber + 1
        ratingtemp = newx.find("div", {"class": "ratings active"})
        a, b = ratingtemp.get("style").split()
        c, d = b.split("%")
        rating = c

        commentlist.append(newx.find("p").text)
        if counter == 1:
            key = productDict.get(list[ind])
        else:
            key = key + str(counter)
        if int(c) > 60:
            opinion = "positive"
        else:
            opinion = "negative"

        listinsert.append((key, newx.find("p").text.encode("utf-8"), rating, opinion))
        dictfull[key] = {newx.find("p").text.encode("utf-8"), str(rating)}


        i = i + 1
        counter = counter + 1
    count = count - 1
    ind = ind + 1



try:
    sql = "INSERT INTO commenttablehepsinew(productname,comment,rating,opinion) VALUES (%s,%s,%s,%s)"

    mycursor.executemany(sql, listinsert)

    mydb.commit()

    print(mycursor.rowcount, "was inserted.")
except:
    print("error while inserting")
