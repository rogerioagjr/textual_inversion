# install diffusers from source
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .

# Return to root directory
cd ..

# Install the textual inversion requirements
pip install -r requirements.txt

echo "REQUIREMENTS INSTALLED!"

# copy the textual inversion in the root directory
# cp diffusers/examples/textual_inversion/textual_inversion.py ./

# Download the cat data
python download_cat_data.py

echo "DATA DOWNLOADED!"

# Run the training script
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./cat"

accelerate launch --config_file ./myconfig.yml textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=5 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="textual_inversion_cat"

echo "TRAINING COMPLETED!"

# Run the inference script
python inference.py

echo "INFERENCE DONE!"
