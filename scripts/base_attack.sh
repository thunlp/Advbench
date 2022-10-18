#!bash
dataset_name=('LUN' 'satnews',"amazon_lb","CGFake","HSOL","jigsaw","EDENCE","FAS","assassin","enron") 
attakcer_name=("pso", "textfooler", "pwws", "bert","deep", "deep25","deep100", "pso100")
for i in {0..9}
do
    for j in {0..7}
    do
        python src/base_attack.py --dataset ${dataset_name[i]}  --attacker ${attacker_name[j]} 
    done
done
