import matplotlib.pyplot as  plt
from matplotlib import pylab

plt.style.use("ggplot")
naivebayes=[0.930,0.926,0.930,0.934,0.945,0.946]

logistic=[0.930,0.935,0.936,0.941,0.947,0.948]

vocabsize=[14603,14200,14100,14100,75251,4116]

process=["pure","lowercase","stopwords","Tf-idf","N-gram","Threshold"]
plt.plot(process,naivebayes,"r-o",label='Naive-Bayes')
plt.plot(process,logistic,"b-o",label="Logistic Regression")
plt.xlabel('Processes')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('fooeven.png')
plt.show()

plt.plot(process,vocabsize,"r-o",label="Vocabulary Size")
plt.xlabel('Processes')
plt.ylabel('Vocabulary size')
plt.legend()

plt.savefig('vocabsizeeven.png')
plt.show()