python gan.py --mode=process_data --data_path=$1
python gan.py --mode=train --n_epochs=200
python gan.py --mode=inference --model_path=result/models/final/

