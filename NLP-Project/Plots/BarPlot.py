import matplotlib.pyplot as  plt
# x-coordinates of left sides of bars
left = [1, 2]

# heights of bars
height = [8040, 1973]


# labels for bars
tick_label = ['pozitif', 'negatif']

# plotting a bar chart
plt.bar(left, height, tick_label = tick_label,
        width = 0.7, color = ['green', 'red'])

# naming the x-axis
plt.xlabel('Yorum duygusu')
# naming the y-axis
plt.ylabel('Yorum sayısı')
# plot title
plt.title('Yorum Dağılımı -10000-')

plt.savefig("yorumdisst.png")
# function to show the plot
plt.show()

heighteven=[1941,2061]

plt.bar(left, heighteven, tick_label = tick_label,
        width = 0.7, color = ['green', 'red'])

# naming the x-axis
plt.xlabel('Yorum duygusu')
# naming the y-axis
plt.ylabel('Yorum sayısı')
# plot title
plt.title('Yorum Dağılımı -4000-')

plt.savefig("yorumdisteven.png")
# function to show the plot
plt.show()



