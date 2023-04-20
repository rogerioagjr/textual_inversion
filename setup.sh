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
echo "SETUP COMPLETED!"
