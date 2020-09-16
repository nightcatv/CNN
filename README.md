# CNN
> For finger vein project
> Still Developing

## Installation
> Use python 3.x
* Suggest use virtual environment of python
```bash
$ virtualenv ve
```
* Run the virtual environment.
	* if os == Linux-based
	```bash
	$ source ve/bin/activate
	```
	* if os == Windows-based
	```bash
	$ ve/Scripts/activate
	```
* Install require packages
```bash
$ pip install -r requirement.txt
```
---
## Usage
```
$ python main.py
```
* Then choose the dataset and model

---
## Explaination
* `main.py`
	* The main program
* `process.py`
	* Define the process procedure
	* There are 2 procedure: train, load\_dataset
* `models.py`
	* Define the CNN model
	* There are 5 models: LeNet, AlexNet, VGG16, GoogLeNet V1, ResNet34
* `output.py`
	* Use plot to draw the chart of training result
