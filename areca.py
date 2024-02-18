import numpy as np
from PIL import Image
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Prepare the dataset
# Replace 'path_to_dataset_folder' with the actual path to your dataset folder
dataset_folder = 'arecanut\dataset'

# Load and process images from the dataset
def process_image(image_path):
    image = Image.open(image_path)
    # Convert the image to grayscale
    image = image.convert('L')
    # Resize the image to a fixed size if necessary
    # image = image.resize((width, height))
    # Convert the image to a numpy array
    image_array = np.array(image)
    return image_array

# Load and process all images in the dataset
def load_dataset():
    X = []  # Input features (image data)
    y = []  # Output labels (disease or healthy)
    
    # Loop through each image in the dataset folder
    # assuming the subfolders represent different classes
    for class_name in ['disease', 'healthy']:
        class_folder = dataset_folder + '/' + class_name + '/'
        image_files = os.listdir(class_folder)
        
        # Process each image in the class folder
        for image_file in image_files:
            image_path = class_folder + image_file
            image_array = process_image(image_path)
            X.append(image_array)
            y.append(class_name)
    
    # Convert the lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Step 2: Split the dataset into training and testing sets
X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Flatten the image data
# Reshape the image array to a 2D matrix (one row per image)
num_samples_train, height, width = X_train.shape
X_train_flat = X_train.reshape(num_samples_train, height * width)

num_samples_test = X_test.shape[0]
X_test_flat = X_test.reshape(num_samples_test, height * width)

# Step 4: Train the Decision Tree classifier
clf = tree.DecisionTreeClassifier()
clf.fit(X_train_flat, y_train)

# Step 5: Make predictions on the test set
y_pred = clf.predict(X_test_flat)

# Step 6: Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
