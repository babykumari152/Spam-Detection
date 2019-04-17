# Spam-Detection
i choose a text file SMSSPAMCollection that contain 5572 total number of message with ham and spam types of message. each message is tab separated by HAM and SPAM tag.
i used three machine learning algorithm to with accuracy of 94 to 95 percent accuracy.

how i accomplished this:
first created bag of words
then extracted 3000 features for each messages.
trained three model on it which gives good accuracy but low recall values as their record for ham and spam were imbalanced
then balanced the class of record in datasets for spam and ham, using SMOTE
then trained three model on it. which result into good accuracy and precision and recall values.
