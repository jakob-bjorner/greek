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

# corruption sweep
python run.py -m +run_modifier=FullTrainRunConfig trainer.datasetloaders.train_dataset.preprocessing.prob_combine=0.2,0.5,1.0 trainer.datasetloaders.train_dataset.preprocessing.prob_delete=0.0,0.2,0.5,1.0 trainer.datasetloaders.train_dataset.preprocessing.prob_swap=0.0,0.2,0.5,1.0