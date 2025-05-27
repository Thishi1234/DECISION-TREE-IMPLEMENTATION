# DECISION-TREE-IMPLEMENTATION

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : CHINTHAPARTHI THISHITHA

*INTERN ID* : CT06DM1408

*DOMAIN* : MACHINE LEARNING

*DURATION* : 6 WEEKS

*MENTOR* : NEELA SANTOSH

Building and visualizing a Decision Tree classifier to forecast breast cancer outcomes using the popular Breast Cancer dataset was the main goal of my machine learning project. This project's goal was to categorize tumors as benign or malignant using a variety of medical characteristics. Google Colab, which offered a versatile and effective platform to run Python scripts directly in the cloud without worrying about local configuration, was the coding environment I used to accomplish this. I used resources from GeeksforGeeks, which provided well-structured tutorials and examples that helped me understand the fundamentals and apply them step-by-step, to learn the theoretical concepts behind decision trees and how to implement them practically in Python using scikit-learn.

The project started by importing the required Python libraries, such as matplotlib and seaborn for visualization, sklearn for machine learning modeling and assessment, and numpy and pandas for data handling. Using pandas.read_csv(), the dataset was loaded, and its structure and contents were examined using fundamental exploratory commands like df.shape, df.head(), and len(df). After that, I divided the dataset into labels (y) and features (X). The labels indicated whether a tumor was benign or malignant, and the features included columns pertaining to the properties of the cell nuclei seen in the digital images.

Data preprocessing was the next step, in which I used train_test_split() to separate the dataset into training and testing subsets, using 80% of the data for training and 20% for testing. I used StandardScaler from sklearn.preprocessing to apply feature scaling in order to enhance the machine learning algorithm's performance. This avoided bias toward attributes with higher values by guaranteeing that all feature values were on the same scale.

I created two distinct Decision Tree models after the data was ready. The first model divided nodes according to the Gini impurity, while the second model used entropy, which is based on information gain. To prevent overfitting and guarantee improved generalization, DecisionTreeClassifier was used to train both models with a maximum depth of 5. I used accuracy_score, confusion_matrix, and classification_report to assess the models' performance after fitting them to the training data and making predictions on the test data. Precision, recall, and F1-score for each class were among the metrics that shed light on the model's performance.

I displayed the confusion matrices using seaborn heatmaps to visually evaluate the results, which made it simpler to understand which predictions were right and which weren't. In order to better visualize how the model made decisions at each node, I also created a graphical representation of the decision tree structure using plot_tree() from sklearn.tree. The decision paths taken during classification were better understood thanks to the trees' plotting, which included clearly labeled features and class names.

Lastly, I was able to determine which splitting criterion performed better on this particular dataset by contrasting the outcomes of the Gini and Entropy-based models. Although the accuracy levels of both models were comparable, this experiment demonstrated how crucial it is to assess various configurations in order to determine which is best for a particular issue.

To sum up, this project gave participants practical experience using a machine learning algorithm on an actual healthcare dataset. It addressed every step of the procedure, including data preprocessing, model evaluation and training, and visualization. Learning from GeeksforGeeks made sure that the implementation adhered to best practices, and using Google Colab made experimentation easy. The project was both technically and socially significant because it combined theory and practice to improve my understanding of decision trees and show how machine learning can be used for important tasks like cancer diagnosis.

#OUTPUT 

![Image](https://github.com/user-attachments/assets/a816b330-970b-43c0-8710-5a87d7e467b1)

![Image](https://github.com/user-attachments/assets/7e69b3f6-5a09-47c7-b8db-652f0a507158)

![Image](https://github.com/user-attachments/assets/c4dad254-3510-4d73-abba-3e4fc9369665)
