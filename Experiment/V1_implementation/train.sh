# cd pds22025/git/pdS22025/Experiment/V1_implementation/
# Terminal 1: Train all modalities configuration
# Using python -u for unbuffered output so logs appear immediately
nohup python -u -m training.main \
  --mode multi_modal \
  --modalities all \
  --num-workers 8 \
  --batch-size 64 \
  --device cuda \
  --patience 10 \
  --n-splits 10 \
  --evaluate-test \
  --force-retrain \
  > train_all.log 2>&1 &

# Terminal 2: Train static+motor
nohup python -u -m training.main \
  --mode multi_modal \
  --modalities static+motor \
  --num-workers 8 \
  --batch-size 64 \
  --device cuda \
  --patience 10 \
  --n-splits 10 \
  --evaluate-test \
  > train_static+motor.log 2>&1 &

# Terminal 3: Train motor_only
nohup python -u -m training.main \
  --mode multi_modal \
  --modalities motor_only \
  --num-workers 8 \
  --batch-size 64 \
  --device cuda \
  --patience 10 \
  --n-splits 10 \
  --force-retrain \
  --evaluate-test \
  > train_motor_only.log 2>&1 &

# Terminal 4: Train static_only
nohup python -u -m training.main \
  --mode multi_modal \
  --modalities static_only \
  --num-workers 8 \
  --batch-size 64 \
  --device cuda \
  --patience 10 \
  --n-splits 10 \
  --evaluate-test \
  > train_static_only.log 2>&1 &

# Terminal 5: Train static+nonmotor
nohup python -u -m training.main \
  --mode multi_modal \
  --modalities static+nonmotor \
  --num-workers 8 \
  --batch-size 64 \
  --device cuda \
  --patience 10 \
  --n-splits 10 \
  --evaluate-test \
  > train_static+nonmotor.log 2>&1 &