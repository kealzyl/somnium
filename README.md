<div align="center">
<img src="images/logo.png" alt="logo">

# Kealzyl Somnium
**What if AI could have dreams?**

</div>

<br>

# Setup :

1. Install libraries :
```pip
pip install torch torchvision pillow
```
2. Download and extract the zip

3. Open the folder
  
5. Download and extract the dataset from [here]().

6. Place the extracted folder in the downloaded project

7. Train by running :
```python
python train_kealzyl_somnium.py --epochs 150 --batch-size 4 --amp
```

## Usage :
1. For simple usage use just :
```python
python kealzyl_somnium_dream.py --input input.jpg 
```
and replace input.jpg with your image

2. For more realistic results use :
```python
python kealzyl_somnium_dream.py --input input.jpg --layer block3 --intensity 1.0  --steps 60 --octaves 4 --max-side 768
```
