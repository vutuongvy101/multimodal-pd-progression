# Terminal 1: Train all modalities configuration
nohup python -m training.main \
  --mode multi_modal \
  --modalities all \
  --num-workers 8 \
  --batch-size 64 \
  --device cuda \
  --max-epochs 70 \
  --patience 10 \
  > train_all.log 2>&1 &

# Terminal 2: Train static+motor
nohup python -m training.main \
  --mode multi_modal \
  --modalities static+motor \
  --num-workers 8 \
  --batch-size 64 \
  --device cuda \
  --max-epochs 70 \
  --patience 10 \
  > train_static+motor.log 2>&1 &

# Terminal 3: Train motor_only
nohup python -m training.main \
  --mode multi_modal \
  --modalities motor_only \
  --num-workers 8 \
  --batch-size 64 \
  --device cuda \
  --max-epochs 70 \
  --patience 10 \
  > train_motor_only.log 2>&1 &

# Terminal 4: Train static_only
nohup python -m training.main \
  --mode multi_modal \
  --modalities static_only \
  --num-workers 8 \
  --batch-size 64 \
  --device cuda \
  --max-epochs 70 \
  --patience 10 \
  > train_static_only.log 2>&1 &

# Terminal 5: Train static+nonmotor
nohup python -m training.main \
  --mode multi_modal \
  --modalities static+nonmotor \
  --num-workers 8 \
  --batch-size 64 \
  --device cuda \
  --max-epochs 70 \
  --patience 10 \
  > train_static+nonmotor.log 2>&1 &