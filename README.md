# Semantic_Segmentation
@Author ZeYuan MA

To get a quick start, just clone and run main.py.

To make contribution, you can make amove in two ways:
    
    1.Contribute your data, Here we need sketches in 224*224 and .png form and its vectorical labels.
        1.1 Concretely,you need to submit a png pic in 224*224, and you need to give us the vectorized path of this 
            picture, the form is like [[24,57,23],[17,90,100]] which meaans the stroke will pass (24,17) (57,90) (23,100)
            these 3 coordinates.
        1.2 If you want to sunmit this kiand of data, please make every piece as a whole csv file.
    2.Contribute your idea about the cnn structure. you can change the pathnet.py and push it to me!

A general statement of PathNet using a flowchart:
    ![](./pathnet_flowchart.png)