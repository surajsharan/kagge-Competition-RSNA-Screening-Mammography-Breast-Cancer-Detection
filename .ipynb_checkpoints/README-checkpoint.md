# kagge-Competition-RSNA-Screening-Mammography-Breast-Cancer-Detection
RSNA Screening Mammography Breast Cancer Detection
Things I tried 

1) ~~write a script to convert images to 1024 and 2048 using joblib parallel and dump in input dir 
2) ~~test ddp
3) ~~clear ml integration
4) ~~create the fresh dataset with cropped regions as heng cher keng kernal/ your own extaction
4) ~~balance smapler of data every bacth as atleast one positive sample
5) ~~investigate pos_weights how it effect the loss
6) ~~create inference kernel
7) ~~Train model with normal data with weighted loss
8) add additinal data and check 
9) run with ExhaustivRandomWeighted Sampler 
10) CostSentive loss function 
11) Model backbone change to timm
12) Model improvement 
13) Coatnet apply
14) Create fresh data with voi lut transformation , 1024x512 , 2048x1024 ,1536x756
15) Create a Dataloder to take atlest one postive in every batch
16) 


17) DO THINGS ASAP
- finalize the data asap ( apply voi lut )


-resent34d bbox segmentation
-run a model woth it COATNET would do better 
- effnetB4 model train


-- create a multilevel model with lama as well
-- probabilities with folds put that in LAMA and output final out come - design the pipeline for that as well
-- see LAMA 




## Labeller 

- using connected component first create all the labels possible with accept and reject mechanism or use mac to display all and check
- stacking model works fine implement pipeline and validate if it gives any boost
- train crop model and immediayely use it fo training 
- specific transformation for positive images so that they do not overfit in dataloader



ACTUAL THINGS TO TRY

- train a model wth lower res and then use that pretrained to upscale
- 1024x512 train on this and then upscale to 1536x768 and 2048x1024 three sizes