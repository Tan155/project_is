import os
import pandas as pd

dataset_path = "C:/Users/hp/Desktop/project_is/datasets"


def load_data(folder):
    data = []
    labels = []

    # check floder
    for label in ["glasses", "no_glasses"]:
        folder_path = os.path.join(folder, label)

        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    file_path = os.path.join(folder_path, filename)
                    data.append(file_path)
                    labels.append(label)

    return pd.DataFrame({"data": data, "label": labels})


# download data
train_data = load_data(dataset_path)

# แสดงตัวอย่างข้อมูล
print(train_data.head())
print(f"📊 All pictures : {len(train_data)}")
print(train_data["label"].value_counts())
