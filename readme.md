# CS682 Project Improved Traffic Sign Classification by Co-Training
1. [ Data preprocess part. ](#pre)
2. [ Filepath mapping part. ](#file)
3. [ Model build and training part. ](#model)


<a name="pre"></a>
## Data Preprocess
### Summary:  
Input files and folders:  
(belgium_test_raw)[https://btsd.ethz.ch/shareddata/]  
(belgium_training_raw)[https://btsd.ethz.ch/shareddata/]  
(germany_test_raw)[http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads]  
(germany_training_raw)[http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads]  
(italy_training_and_test_raw)[http://users.diag.uniroma1.it/bloisi/ds/dits.html]  

Process file:  
data_preprocess.py  

Output files and folders:  
belgium_test_processed, belgium_training_processed, germany_test_processed,   
germany_training_processed, italy_training_and_test_processed  


### Output Example
Before process:  
![alt text](https://github.com/zhaolinzhang/CS682ProjectPreprocess/blob/master/00000_00000_1.png)

After process:  
![alt text](https://github.com/zhaolinzhang/CS682ProjectPreprocess/blob/master/00000_00000_2.png)  


### Algorithm Used
1.Gaussian Blur by Open CV  
2.Histogram Equalized by Open CV  
3.Normalized by Open CV  


### Library dependencies requirements:
1.glob  
```
sudo pip3 install glob3
```  
2.pandas (should come with virtual environment)  
```
pip install pandas
```  


### Usage:
```
Usage: -c <country> -i <dir_from_path> -o <dir_to_path> -w <output_img_width> -h <output_img_height>
```
country: limited to [ge, be, it].   
'ge' stands for Germany, 'be' stands for belgium, 'it' stands for italy  

dir_from_path: limit to exact 2 upper direction folder of image file.  
For instance,
```
./germany_test_raw/Final_Test
``` 
is a good path  

dir_from_path: no special rule as long as it's valid  

output_img_width: Output image width. Limited to integer  

output_img_height: Output image height. Limited to integer  


### Usage Example:
```
python3 data_preprocess.py -c ge -i ./germany_test_raw/Final_Test -o ./germany_test_processed/Final_Test -w 48 -h 48
python3 data_preprocess.py -c ge -i ./germany_training_raw/Final_Training/Images -o ./germany_training_processed/Final_Training/Images -w 48 -h 48
python3 data_preprocess.py -c be -i ./belgium_test_raw -o ./belgium_test_processed -w 48 -h 48
python3 data_preprocess.py -c be -i ./belgium_training_raw -o ./belgium_training_processed -w 48 -h 48
python3 data_preprocess.py -c it -i ./italy_training_and_test_raw/classification_test -o ./italy_training_and_test_processed/classification_test -w 48 -h 48
python3 data_preprocess.py -c it -i ./italy_training_and_test_raw/classification_train -o ./italy_training_and_test_processed/classification_train -w 48 -h 48
```


### Notes:
The corresponding labels of each image are still in raw data's folder.  
If you are going to run the raw data, in Belgium trainning raw data, files 'GT-00056.csv' and ' 'GT-00057.csv' has duplicate records;   
If you are going to run the raw data, in Belgium test raw data,files 'GT-00035.csv' and 'GT-00038.csv' has duplicate records.  
  
<a name="file"></a>
## Filepath mapping
### Summary:  
Input files and folders:  
[germany_test_processed]  
[germany_training_processed]  
[italy_test_processed]  
[italy_training_processed]  
  
Process file:  
rewrite_csv.py

Output files and folders:  
[filepath_class_mapping]
  
### Usage:
```
python3 rewrite_csv.py
```


<a name="model"></a>
## Model build and training
### Summary:
Input files and folders:  
[germany_test_processed]  
[germany_training_processed]  
[italy_test_processed]  
[italy_training_processed]  
  
Process file:  
cotrain_finalversion.py  
  
Output files:  
cotrain_validation_germany_CNN.txt  
cotrain_validation_italy_CNN.txt  
cotrain_accuracies_germany_CNN.txt  
cotrain_accuracies_italy_CNN.txt  
cotrain_validation_germany_EIGEN.txt  
cotrain_validation_italy_EIGEN.txt  
cotrain_accuracies_germany_EIGEN.txt  
cotrain_accuracies_italy_EIGEN.txt  
cotrain_validation_germany_COTRAIN.txt  
cotrain_validation_italy_COTRAIN.txt  
cotrain_accuracies_germany_COTRAIN.txt  
cotrain_accuracies_italy_COTRAIN.txt  
  
### Library dependencies requirements:  
1. keras
2. sklearn
3. cv2
4. pandas
  
### Usage:
```
python3 cotrain_finalversion.py
```
It will sequentially run CNN model, Eigen Model, and Co-train Model.  
Notice each model might take around 24 hours to process depends on computing performance.  
  

