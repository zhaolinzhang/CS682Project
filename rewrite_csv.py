import os
import pandas as pd
import glob


def re_write_cvs(input_path, cvs_file_name):

    file_name_collection = []
    class_name_collection = []

    for path in os.listdir(input_path):
        print("Using: " + os.path.join(path, '*.csv'))
        file_path = os.path.join(input_path, path)
        annotations = pd.read_csv(glob.glob(os.path.join(file_path, '*.csv'))[0], sep=';')
        annotations = annotations.set_index('Filename')
        print(annotations)

        for file_name in os.listdir(file_path):
            # print("Processing file: " + file_name)
            if file_name.endswith('.png'):
                file_name_collection.append(os.path.join(file_path, file_name))
                new_image_name = file_name.split(".")[-2] + ".ppm"
                class_name = annotations.at[new_image_name, 'ClassId']
                class_name_collection.append(class_name)
            elif file_name.endswith('.csv'):
                # os.remove(os.path.join(file_path, file_name))
                pass
    df = pd.DataFrame({'file': file_name_collection,
                       'class': class_name_collection})
    df.to_csv(path_or_buf=os.path.join(input_path, cvs_file_name), index=False)


if __name__ == '__main__':
    re_write_cvs("./germany_test_processed", "./filepath_class_mapping/germany_test_processed.csv")
    re_write_cvs("./germany_training_processed", "./filepath_class_mapping/germany_training_processed.csv")
    re_write_cvs("./italy_test_processed", "./filepath_class_mapping/italy_test_processed.csv")
    re_write_cvs("./italy_training_processed", "./filepath_class_mapping/italy_training_processed.csv")