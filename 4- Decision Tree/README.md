# Decision Tree
### Problem statement:

The data set available for this problem is based on the U.S. congress voting record
from 1984. The data set consists of the votes (yes or no) on sixteen issues for each of
the 435 members of congress. from the voting record. You will use this data to learn a
decision tree that predicts the political party of the representative based on his /her vote.


1. Use the voting data to train a decision tree to predict political party (Democrat or
Republican) based on the voting record. Use 30% of the members of congress for
training and the rest for testing. Rerun this experiment five times and notice the
impact of different random splits of the data into training and test sets. Report the
sizes and accuracies of these trees in each experiment.


2. Measure the impact of training set size on the accuracy and the size of the learned
tree. Consider training set sizes in the range (30-70%). Because of the high variance
due to random splits repeat the experiment with five different random seeds for each
training set size then report the mean, maximum and minimum accuracies at each
training set size. Also measure the mean, max and min tree size.
Start with training data size 30%, 40% .. Until you reach 70%.
The data set contained many missing values , i.e., votes in which a member of
congress failed to participate. To solve those issue insert for each absent
vote the voting decision of the majority.
