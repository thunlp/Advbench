#!bash
dataset_name=('LUN' 'satnews',"amazon_lb","CGFake","HSOL","jigsaw","EDENCE","FAS","assassin","enron") 

for i in {0..9}
do
    python src/rocket --name ${dataset_name[i]} 
done
