This is the user guide of running inACP anticancer peptides prediction model.

First of all, download the Python source code, dataset and pretrained model from https://github.com

Next, execute following commands:

unzip inACP.zip

cd inACP

pip install -r pip requirements.txt

wget bergerlab-downloads.csail.mit.edu/bepler-protein-sequence-embeddings-from-structure-iclr2019/pretrained_models.tar.gz

tar -xzvf pretrained_models.tar.gz

mkdir ./src/PretrainedModel/

mv ./pretrained_models/ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav ./src/PretrainedModel/SSA_embed.model

python ensemble.py

And now the inACP program should be running and will output the prediction result for the principal dataset.

