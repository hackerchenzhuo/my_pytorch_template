#data=0;
##rel=103;
#fact=2791;
#score=1000;
##for score in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,100}
##  do
#  for rel in {3,5,7,10,15,20,25}
##      do
##      for fact in {2,3,5,10,15,20,25}
#        do
#        python joint_test.py --gpu_id 1 --exp_name fusion_prediction_zsl_rel --exp_id rel"${rel}"_fact"${fact}"data_"${data}"score_"${score}" --data_choice "${data}" --top_rel "${rel}" --top_fact "${fact}" --soft_score "${score}" --ZSL 1 --mrr 1
#        done
##      done
##  done


python main.py