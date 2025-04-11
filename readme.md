# Repo for replicating and extending [Awesome-Align](https://github.com/neulab/awesome-align/tree/master?tab=readme-ov-file)
Motivation for reimplementation:
- their exisisting repo is difficult to make extentions on without effectively reorganizing the entire setup anyway.
- improve ease of experimentation with Hydra integration as well.
- provide organization for other extentions in terms of sentence and book level alignments
- allow for different models to be used other than just bert, as is shown to be effective in that one paper which used labse as an encoding model
- Ideally more easy to add losses and other settings than is currently.

Notes:
- overwritting the logger, (trainer HAS-A logger) is done as follows: 

```bash
python run.py logger@trainer.logger=BasicPrintLogger
```
this is more complex than I would have hoped for, but can easily be seen why it should be done this way, and can be more easily redefined in config classes than on the command line, but easy to make sweeps with the syntax still. The autocomplete doesn't recognize this syntax tho, have to start with logger=\tab, and then go back and add the @ sugar.

- test the models which were trained inside this new repo to ensure prec recall aer and coverage are consistent. Done. on nopretraining they have the same AER prec and recall.

debugging script:
```bash
python run.py +run_modifier=[DebugRunConfig,OfflineRunConfig] dataset@trainer.datasetloaders.train_dataset=JaEnSupervisedAwesomeAlignDatasetEval trainer.max_steps=-1 trainer.max_epochs=1 trainer.log_every_n_steps=10
```

replicate supervised training script:
```bash
python run.py +run_modifier=SupervisedRunConfig
```

https://pytorch.org/get-started/locally/

then install:
pip install python-dotenv
pip install hydra-core
pip install hydra-submitit-launcher --upgrade
pip install ipython
pip install wandb
pip install transformers

# coverage schedule
python run.py -m +run_modifier=FullTrainRunConfig trainer.model.max_softmax_temperature_end=1e-16,1e-20 trainer.model.max_softmax_temperature_start=1 trainer.model.coverage_weight=0.06,0.03,0.01,0.005 trainer.model.coverage_encouragement_type=max_softmax

## above didn't do much, so tyring different temp start, and higher coverage weight.
python run.py -m +run_modifier=FullTrainRunConfig trainer.model.max_softmax_temperature_end=1e-16 trainer.model.max_softmax_temperature_start=1e-6,1e-10,1e-14 trainer.model.coverage_weight=0.06 trainer.model.coverage_encouragement_type=max_softmax

python run.py -m +run_modifier=FullTrainRunConfig trainer.model.max_softmax_temperature_end=1e-16 trainer.model.max_softmax_temperature_start=1 trainer.model.coverage_weight=0.1,0.3,0.7 trainer.model.coverage_encouragement_type=max_softmax

### above also didn't do much. Seems like the bug has changed what values are good to try. Should try many different values potentially just supervised setting first, and then experiment on the pretrain later.
python run.py -m +run_modifier=SupervisedRunConfig "trainer.model={max_softmax_temperature_start:5e-3,max_softmax_temperature_end:5e-3},{max_softmax_temperature_start:1e-3,max_softmax_temperature_end:1e-3},{max_softmax_temperature_start:1e-2,max_softmax_temperature_end:1e-2},{max_softmax_temperature_start:5e-2,max_softmax_temperature_end:5e-2},{max_softmax_temperature_start:1e-1,max_softmax_temperature_end:1e-1},{max_softmax_temperature_start:5e-1,max_softmax_temperature_end:5e-1}" trainer.model.coverage_weight=0.1,0.5,1 trainer.model.coverage_encouragement_type=max_softmax

want to start doing sweeps and reporting those sweeps to wandb.
python run.py -m hydra/sweeper=optuna hydra/sweeper/sampler=random hydra.sweeper.n_trials=60  hydra.sweeper.n_jobs=60 +run_modifier=SupervisedRunConfig "trainer.model.max_softmax_temperature_start=tag(log, interval(1e-5,10))" "trainer.model.max_softmax_temperature_end=tag(log, interval(1e-5,10))" "trainer.model.coverage_weight=tag(log, interval(1e-3,10))" trainer.model.coverage_encouragement_type=max_softmax trainer.get_optimizer.lr=0.00007

supervised_pp_cdms=\(0.0, 0.0, 0.0, 0.0\)_lr=7e-05|supervised_maxCvgTempStartEnd=\(\d.\d{3}\d+

<!-- python run.py -m +run_modifier=SupervisedRunConfig "trainer.model={max_softmax_temperature_start:0.0023,max_softmax_temperature_end:0.304}" trainer.model.coverage_weight=2.4 seed=3,4,5 trainer.model.coverage_encouragement_type=max_softmax -->
python run.py -m +run_modifier=FullTrainRunConfig "trainer.model={max_softmax_temperature_start:0.00026,max_softmax_temperature_end:0.0026}" trainer.model.coverage_weight=0.1 trainer.model.coverage_encouragement_type=max_softmax_linear seed=3,4,5 
python run.py -m +run_modifier=FullTrainRunConfig "trainer.model={max_softmax_temperature_start:0.00026,max_softmax_temperature_end:0.0026}" trainer.model.coverage_weight=0.01 trainer.model.coverage_encouragement_type=max_softmax_linear seed=3,4,5 
python run.py -m +run_modifier=FullTrainRunConfig "trainer.model={max_softmax_temperature_start:1,max_softmax_temperature_end:0.0026}" trainer.model.coverage_weight=1 trainer.model.coverage_encouragement_type=max_softmax_const_cosine seed=3,4,5 hydra.launcher.partition=kargo-lab
python run.py -m +run_modifier=FullTrainRunConfig "trainer.model={max_softmax_temperature_start:1,max_softmax_temperature_end:0.0026}" trainer.model.coverage_weight=1 trainer.model.coverage_encouragement_type=max_softmax_cosine seed=3,4,5 hydra.launcher.partition=kargo-lab

<!-- python run.py -m +run_modifier=FullTrainRunConfig "trainer.model={max_softmax_temperature_start:1,max_softmax_temperature_end:0.0026}" trainer.model.coverage_weight=1 trainer.model.coverage_encouragement_type=max_softmax_const_cosine seed=3,4,5 hydra.launcher.partition=kargo-lab -->
python run.py -m +run_modifier=FullTrainRunConfig "trainer.model={max_softmax_temperature_start:1,max_softmax_temperature_end:0.00026}" trainer.model.coverage_weight=1,10 trainer.model.coverage_encouragement_type=max_softmax_const_cosine seed=3,4,5 #hydra.launcher.partition=kargo-lab


full_train_maxCvgTempStartEnd=(1, 0.0026)_cvgW=1.0_cvgType=max_softmax_const_cosine_lr=2e-05_seed=4
# corruption sweep
python run.py -m +run_modifier=FullTrainRunConfig preprocessing.prob_combine=0.2,0.5,1.0 preprocessing.prob_delete=0.0,0.2,0.5,1.0 preprocessing.prob_swap=0.0,0.2,0.5,1.0

python run.py -m +run_modifier=FullTrainRunConfig run_type=full_tlm_train "preprocessing={prob_combine:1.0, prob_delete:1.0, prob_move:1.0, prob_swap:1.0},{prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0},{prob_combine:0.0, prob_delete:0.0, prob_move:0.0, prob_swap:0.0}" datasetmap@trainer.datasetloaders.val_datasets=nozhPlusPPSupervisedAwesomeAlignDatasetsMapEval seed=2,3,4 trainer.model.train_so=false trainer.model.train_psi=false trainer.model.train_mlm=false


python run.py -m +run_modifier=FullTrainRunConfig seed=2,3,4 "preprocessing={prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0}" datasetmap@trainer.datasetloaders.val_datasets=nozhPlusPPSupervisedAwesomeAlignDatasetsMapEval seed=2,3,4,5,6 hydra.launcher.partition=kargo-lab

python run.py -m +run_modifier=[FullTrainRunConfig] run_type=full_train_no_move_eval datasetmap@trainer.datasetloaders.val_datasets=nozhPlusPPSupervisedAwesomeAlignDatasetsMapEval seed=2,3,4,5,6 hydra.launcher.partition=kargo-lab "trainer.datasetloaders.val_datasets.enfrpp_dataset.preprocessing={_target_:greek.dataset.dataset.preprocessing, _partial_:True, prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0}" "trainer.datasetloaders.val_datasets.deenpp_dataset.preprocessing={_target_:greek.dataset.dataset.preprocessing, _partial_:True, prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0}" "trainer.datasetloaders.val_datasets.roenpp_dataset.preprocessing={_target_:greek.dataset.dataset.preprocessing, _partial_:True, prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0}" "trainer.datasetloaders.val_datasets.jaenpp_dataset.preprocessing={_target_:greek.dataset.dataset.preprocessing, _partial_:True, prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0}"

"trainer.datasetloaders.val_datasets.enfrpp_dataset.preprocessing={_target_:greek.dataset.dataset.preprocessing, _partial_:True, prob_combine:1.0, prob_delete:1.0, prob_move:1.0, prob_swap:1.0}" "trainer.datasetloaders.val_datasets.deenpp_dataset.preprocessing={_target_:greek.dataset.dataset.preprocessing, _partial_:True, prob_combine:1.0, prob_delete:1.0, prob_move:1.0, prob_swap:1.0}" "trainer.datasetloaders.val_datasets.roenpp_dataset.preprocessing={_target_:greek.dataset.dataset.preprocessing, _partial_:True, prob_combine:1.0, prob_delete:1.0, prob_move:1.0, prob_swap:1.0}" "trainer.datasetloaders.val_datasets.jaenpp_dataset.preprocessing={_target_:greek.dataset.dataset.preprocessing, _partial_:True, prob_combine:1.0, prob_delete:1.0, prob_move:1.0, prob_swap:1.0}"



# masking and word masking sweeps



python run.py +run_modifier=[OfflineRunConfig,DebugRunConfig,FullTrainRunConfig] maskedlm@trainer.model.maskedlm=PreTrainedBERTMaskedLMWithLSTM trainer.model.maskedlm.combine_layer=4 trainer.model.maskedlm.expand_layer=10 trainer.datasetloaders.train_dataset.len_per_lang=1000 trainer.model.word_masking=True

python run.py -m +run_modifier=FullTrainRunConfig maskedlm@trainer.model.maskedlm=PreTrainedBERTMaskedLMWithLSTM trainer.model.maskedlm.combine_layer=7 trainer.model.maskedlm.expand_layer=9,10 trainer.model.mlm_probability=0.15,0.3 trainer.model.word_masking=True hydra.launcher.partition=kargo-lab seed=2,3

python run.py -m +run_modifier=FullTrainRunConfig  hydra.launcher.partition=kargo-lab maskedlm@trainer.model.maskedlm=PreTrainedBERTMaskedLMWithLSTM trainer.model.maskedlm.combine_layer=1,2,3,4,5,6 trainer.model.maskedlm.expand_layer=9 trainer.model.mlm_probability=0.15 trainer.model.word_masking=True 

python run.py -m +run_modifier=FullTrainRunConfig seed=3,4 trainer.model.word_masking=True trainer.model.mlm_probability=0.15,0.4,0.7 hydra.launcher.partition=kargo-lab


python run.py -m hydra/sweeper=optuna hydra/sweeper/sampler=random hydra.sweeper.n_trials=60  hydra.sweeper.n_jobs=60 +run_modifier=FullTrainRunConfig "trainer.model.so_weight=tag(log, interval(1e-3,1))" "trainer.model.tlm_weight=tag(log, interval(1e-3,1))" "trainer.model.mlm_weight=tag(log, interval(1e-3,1))" "trainer.model.psi_weight=tag(log, interval(1e-3,1))"

python run.py -m +run_modifier=FullTrainRunConfig seed=2,3,4 trainer.model.word_masking=False trainer.model.mlm_probability=0.15,0.4,0.7 hydra.launcher.partition=kargo-lab

# Classification like image segmentation
python run.py +run_modifier=[ClassificationTrainRunConfig,DebugRunConfig,OfflineRunConfig] seed=2
python run.py -m +run_modifier=ClassificationTrainRunConfig seed=2 trainer.datasetloaders.batch_size=1,2,4,8

try with corrupted data
python run.py -m run_type=classification_tlm +run_modifier=ClassificationTrainRunConfig seed=2 trainer.datasetloaders.batch_size=2 "preprocessing={prob_combine:1.0, prob_delete:1.0, prob_move:1.0, prob_swap:1.0},{prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0},{prob_combine:0.0, prob_delete:0.0, prob_move:0.0, prob_swap:0.0}" dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnPPSupervisedAwesomeAlignDatasetsMapEval trainer.max_epochs=50 trainer.model.train_tlm=true trainer.model.train_tlm_full=true

python run.py -m +run_modifier=SupervisedRunConfig seed=2 "preprocessing={prob_combine:1.0, prob_delete:1.0, prob_move:1.0, prob_swap:1.0},{prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0},{prob_combine:0.0, prob_delete:0.0, prob_move:0.0, prob_swap:0.0}" dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnPPSupervisedAwesomeAlignDatasetsMapEval trainer.max_epochs=50 "trainer.model.maskedlm.pretrained_model_name_or_path=/nethome/jbjorner3/dev/diffusion-fun/text_diffusion/greek/awesome-align/greek/model_replicate_full_out_2"

-- try with clean training data and corrupt eval data? does this make sense? The corrupt training data will give the model a better sense of the eval data, so it doesn't really make sense...

-- trying different layer number, and different model

python run.py -m run_type=classification_tlm +run_modifier=ClassificationTrainRunConfig seed=2 trainer.datasetloaders.batch_size=2 "preprocessing={prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0},{prob_combine:0.0, prob_delete:0.0, prob_move:0.0, prob_swap:0.0}" dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnPPSupervisedAwesomeAlignDatasetsMapEval trainer.max_epochs=50 trainer.model.layer_number=7,8 "trainer.model.maskedlm.pretrained_model_name_or_path=google-bert/bert-base-multilingual-cased" trainer.model.train_tlm=true trainer.model.train_tlm_full=true

python run.py -m run_type=classification +run_modifier=ClassificationTrainRunConfig seed=2 trainer.datasetloaders.batch_size=2 "preprocessing={prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0},{prob_combine:0.0, prob_delete:0.0, prob_move:0.0, prob_swap:0.0}" dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnPPSupervisedAwesomeAlignDatasetsMapEval trainer.max_epochs=50 trainer.model.layer_number=7,8 "trainer.model.maskedlm.pretrained_model_name_or_path=google-bert/bert-base-multilingual-cased"

python run.py -m run_type=classification +run_modifier=ClassificationTrainRunConfig seed=2 trainer.datasetloaders.batch_size=2 "preprocessing={prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0},{prob_combine:0.0, prob_delete:0.0, prob_move:0.0, prob_swap:0.0}" dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnPPSupervisedAwesomeAlignDatasetsMapEval trainer.max_epochs=50 trainer.model.layer_number=7,8

<!-- FIX TLM causes nan loss??? gradient norm is very large maybe related. happens after step 300 in the val set. it seems to be able to recover?
FIX inconcsistent behaviour with predictions these give completely different AER, and training losses. Not sure where the difference stems from...
python run.py -m run_type=cls1,cls2,cls3 +run_modifier=ClassificationTrainRunConfig seed=2 trainer.datasetloaders.batch_size=2 dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnPPSupervisedAwesomeAlignDatasetsMapEval trainer.max_epochs=50 trainer.model.layer_number=7 trainer.datasetloaders.num_workers=0
before first training loop
torch.rand(10)
tensor([0.1608, 0.3434, 0.0996, 0.2070, 0.1051, 0.7362, 0.1932, 0.8068, 0.0974,0.6293])
before second training loop 
torch.rand(10)
tensor([0.6216, 0.6705, 0.7929, 0.7528, 0.7356, 0.4188, 0.4225, 0.3737, 0.3711, 0.3814])
before 100th training loop
torch.rand(10)
tensor([0.9163, 0.8969, 0.6658, 0.7784, 0.1039, 0.9369, 0.3630, 0.3672, 0.2356, 0.6566])  
COULDN'T FIX-->

adding TLM should be done with continuous pretraining, and not with the supervising dataset. Try a checkpoint where the tlm is run on corrupted pretraining data, and then run the supervision.
python run.py -m +run_modifier=FullTrainRunConfig run_type=full_tlm_train "preprocessing={prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0},{prob_combine:0.0, prob_delete:0.0, prob_move:0.0, prob_swap:0.0}" datasetmap@trainer.datasetloaders.val_datasets=nozhPlusPPSupervisedAwesomeAlignDatasetsMapEval seed=2 trainer.model.train_so=false trainer.model.train_psi=false trainer.model.train_mlm=false hydra.launcher.partition=kargo-lab
python run.py -m run_type=classification +run_modifier=ClassificationTrainRunConfig seed=2 trainer.datasetloaders.batch_size=2 "preprocessing={prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0}" dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnPPSupervisedAwesomeAlignDatasetsMapEval trainer.max_epochs=50 trainer.model.layer_number=7,8,9  "trainer.model.maskedlm.pretrained_model_name_or_path=/nethome/jbjorner3/dev/diffusion-fun/text_diffusion/greek/greek/greek_runs/tlm_pp\=1_1_0_1"
python run.py -m run_type=classification +run_modifier=ClassificationTrainRunConfig seed=2 trainer.datasetloaders.batch_size=2 "preprocessing={prob_combine:0.0, prob_delete:0.0, prob_move:0.0, prob_swap:0.0}" dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnPPSupervisedAwesomeAlignDatasetsMapEval trainer.max_epochs=50 trainer.model.layer_number=7,8,9  "trainer.model.maskedlm.pretrained_model_name_or_path=/nethome/jbjorner3/dev/diffusion-fun/text_diffusion/greek/greek/greek_runs/tlm_pp\=0_0_0_0"
replicating normal awesome-align semi-supervised setting on corrupted data for baseline:
python run.py -m +run_modifier=SupervisedRunConfig seed=2 "preprocessing={prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0}" dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnPPSupervisedAwesomeAlignDatasetsMapEval trainer.max_epochs=5,50 "trainer.model.maskedlm.pretrained_model_name_or_path=/nethome/jbjorner3/dev/diffusion-fun/text_diffusion/greek/greek/greek_runs/tlm_pp\=1_1_0_1"
python run.py -m +run_modifier=SupervisedRunConfig seed=2 "preprocessing={prob_combine:0.0, prob_delete:0.0, prob_move:0.0, prob_swap:0.0}" dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnPPSupervisedAwesomeAlignDatasetsMapEval trainer.max_epochs=5,50 "trainer.model.maskedlm.pretrained_model_name_or_path=/nethome/jbjorner3/dev/diffusion-fun/text_diffusion/greek/greek/greek_runs/tlm_pp\=0_0_0_0"

supervise with semi supervised awesome align setting ?? How do I replicate their AER of 30.0? (I think I can just beat it?)
python run.py -m +run_modifier=SupervisedRunConfig seed=2 "preprocessing={prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0}" dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnPPSupervisedAwesomeAlignDatasetsMapEval trainer.max_epochs=5,50 "trainer.model.maskedlm.pretrained_model_name_or_path=/nethome/jbjorner3/dev/diffusion-fun/text_diffusion/greek/greek/greek_runs/super_pp\=1_1_0_1"
python run.py -m +run_modifier=SupervisedRunConfig seed=2 "preprocessing={prob_combine:0.0, prob_delete:0.0, prob_move:0.0, prob_swap:0.0}" dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnPPSupervisedAwesomeAlignDatasetsMapEval trainer.max_epochs=5,50 "trainer.model.maskedlm.pretrained_model_name_or_path=/nethome/jbjorner3/dev/diffusion-fun/text_diffusion/greek/greek/greek_runs/super_pp\=0_0_0_0"

test set replicate results from awesome align and form experiments in dev.ipynb where awesome align does better???
python run.py +run_modifier=SupervisedRunConfig seed=2 dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnSupervisedAwesomeAlignDatasetsMapTest trainer.max_epochs=5 
python run.py -m run_type=classification +run_modifier=ClassificationTrainRunConfig seed=2 trainer.datasetloaders.batch_size=2 dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnSupervisedAwesomeAlignDatasetsMapTest trainer.max_epochs=50 trainer.model.layer_number=7,8

try with greek:
python run.py +run_modifier=SupervisedRunConfig seed=2 dataset@trainer.datasetloaders.train_dataset=GeEnSupervisedAwesomeAlignDatasetEval datasetmap@trainer.datasetloaders.val_datasets=GeEnSupervisedAwesomeAlignDatasetsMapTest trainer.max_epochs=5 
python run.py -m run_type=classification +run_modifier=ClassificationTrainRunConfig seed=2 trainer.datasetloaders.batch_size=2  dataset@trainer.datasetloaders.train_dataset=GeEnSupervisedAwesomeAlignDatasetEval datasetmap@trainer.datasetloaders.val_datasets=GeEnSupervisedAwesomeAlignDatasetsMapTest  trainer.max_epochs=50 trainer.model.layer_number=7,8,9 "trainer.model.maskedlm.pretrained_model_name_or_path=google-bert/bert-base-multilingual-cased,/nethome/jbjorner3/dev/diffusion-fun/text_diffusion/greek/greek/greek_runs/tlm_pp\=0_0_0_0,/nethome/jbjorner3/dev/diffusion-fun/text_diffusion/greek/awesome-align/greek/model_replicate_full_out_2"

"trainer.model.maskedlm.pretrained_model_name_or_path=/nethome/jbjorner3/dev/diffusion-fun/text_diffusion/greek/greek/greek_runs/tlm_pp\=0_0_0_0"


try with convnetwork classifier (similar to other end2end alignment techniques, using convs after an initial alignment guess is made.)
python run.py -m run_type=classification +run_modifier=ClassificationTrainRunConfig classifier@trainer.model.classifier=ConvClassifierNetConfig seed=2 trainer.datasetloaders.batch_size=2 "preprocessing={prob_combine:1.0, prob_delete:1.0, prob_move:0.0, prob_swap:1.0}" dataset@trainer.datasetloaders.train_dataset=JaEnPPSupervisedAwesomeAlignDatasetTraining datasetmap@trainer.datasetloaders.val_datasets=JaEnPPSupervisedAwesomeAlignDatasetsMapEval trainer.max_epochs=50 trainer.model.layer_number=7,8,9 "trainer.model.maskedlm.pretrained_model_name_or_path=/nethome/jbjorner3/dev/diffusion-fun/text_diffusion/greek/greek/greek_runs/tlm_pp\=1_1_0_1" trainer.get_scheduler.warmup_steps=1000

try with uninitialized model (not awesome align) (doesn't work as well, Might not know the languages well?)
try along with tlm and mlm losses (with just TLM performs similar to awesome align, in corrupted setting much better!)
try with different layers (seems to like layer 8 and 9 if it is trained with just TLM loss)
try with data corruptions (promising, but some generalization issues with paratext)
try with less data 
try with different classification heads, 
try with so loss