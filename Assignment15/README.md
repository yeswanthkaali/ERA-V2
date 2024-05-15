# Session 7 Assignment

This assignment is for training Yolov3 for custome dataset here we have picked a dataset of superman,batman and ironman
## Training Log
I


python train.py --batch 16 --epochs 25 --img 640 --device cpu --min-items 0 --close-mosaic 15 --data ./data/superheros.yaml --weights ./weights/gelan-c.pt --cfg models/detect/gelan-c.yaml --hyp hyp.scratch-high.yaml 
train: weights=./weights/gelan-c.pt, cfg=models/detect/gelan-c.yaml, data=./data/superheros.yaml, hyp=hyp.scratch-high.yaml, epochs=25, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=None, image_weights=False, device=cpu, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, flat_cos_lr=False, fixed_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, min_items=0, close_mosaic=15, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
/bin/sh: -c: line 0: syntax error near unexpected token `('
/bin/sh: -c: line 0: `git -C /Users/yeswanth/Library/CloudStorage/OneDrive-SoftwareAG/Desktop(2)/yolov9 describe --tags --long --always'
YOLOv5 🚀 2024-5-14 Python-3.10.14 torch-2.3.0 CPU

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, cls_pw=1.0, dfl=1.5, obj_pw=1.0, iou_t=0.2, anchor_t=5.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.9, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.15, copy_paste=0.3
ClearML: run 'pip install clearml' to automatically track, visualize and remotely train YOLO 🚀 in ClearML
Comet: run 'pip install comet_ml' to automatically track and visualize YOLO 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=3

                 from  n    params  module                                  arguments                     
  0                -1  1      1856  models.common.Conv                      [3, 64, 3, 2]                 
  1                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  2                -1  1    212864  models.common.RepNCSPELAN4              [128, 256, 128, 64, 1]        
  3                -1  1    164352  models.common.ADown                     [256, 256]                    
  4                -1  1    847616  models.common.RepNCSPELAN4              [256, 512, 256, 128, 1]       
  5                -1  1    656384  models.common.ADown                     [512, 512]                    
  6                -1  1   2857472  models.common.RepNCSPELAN4              [512, 512, 512, 256, 1]       
  7                -1  1    656384  models.common.ADown                     [512, 512]                    
  8                -1  1   2857472  models.common.RepNCSPELAN4              [512, 512, 512, 256, 1]       
  9                -1  1    656896  models.common.SPPELAN                   [512, 512, 256]               
 10                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 11           [-1, 6]  1         0  models.common.Concat                    [1]                           
 12                -1  1   3119616  models.common.RepNCSPELAN4              [1024, 512, 512, 256, 1]      
 13                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 14           [-1, 4]  1         0  models.common.Concat                    [1]                           
 15                -1  1    912640  models.common.RepNCSPELAN4              [1024, 256, 256, 128, 1]      
 16                -1  1    164352  models.common.ADown                     [256, 256]                    
 17          [-1, 12]  1         0  models.common.Concat                    [1]                           
 18                -1  1   2988544  models.common.RepNCSPELAN4              [768, 512, 512, 256, 1]       
 19                -1  1    656384  models.common.ADown                     [512, 512]                    
 20           [-1, 9]  1         0  models.common.Concat                    [1]                           
 21                -1  1   3119616  models.common.RepNCSPELAN4              [1024, 512, 512, 256, 1]      
 22      [15, 18, 21]  1   5492953  models.yolo.DDetect                     [3, [256, 512, 512]]          
gelan-c summary: 621 layers, 25439385 parameters, 25439369 gradients, 103.2 GFLOPs

Transferred 931/937 items from weights/gelan-c.pt
optimizer: SGD(lr=0.01) with parameter groups 154 weight(decay=0.0), 161 weight(decay=0.0005), 160 bias
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning /Users/yeswanth/Library/CloudStorage/OneDrive-SoftwareAG/Desktop(2)/yolov9/data/cust
val: Scanning /Users/yeswanth/Library/CloudStorage/OneDrive-SoftwareAG/Desktop(2)/yolov9/data/custom
Plotting labels to runs/train/exp5/labels.jpg... 
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to runs/train/exp5
Starting training for 25 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       0/24         0G      1.515      3.127      1.947         12        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.238      0.421      0.264      0.137

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/24         0G      1.512      2.355      1.841         12        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.548      0.513      0.536      0.302

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/24         0G      1.401       2.07       1.74         12        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.656      0.563       0.62      0.366

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/24         0G      1.312      1.964      1.618         20        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.669      0.587      0.659      0.416

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/24         0G      1.285      1.697       1.56         10        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170       0.76      0.639       0.74      0.469

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/24         0G      1.341      1.656      1.568         15        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.678      0.671      0.742      0.479

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/24         0G      1.336      1.521      1.572         14        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.761      0.744      0.803      0.534

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/24         0G      1.171      1.335      1.427          9        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.798      0.776       0.84      0.546

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/24         0G      1.157      1.282      1.404         14        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.788      0.783      0.814      0.521

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/24         0G      1.089      1.222      1.358         11        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.825      0.826      0.856      0.573
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/24         0G     0.9746       1.26      1.418          5        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.654      0.765      0.792      0.502

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/24         0G      1.126      1.373      1.535          5        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.834      0.753      0.885      0.598

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/24         0G      1.012      1.185      1.471          7        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.824      0.768       0.87      0.579

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/24         0G      1.104     0.9638      1.586          4        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.871      0.808        0.9      0.601

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/24         0G      1.011     0.9313      1.465          4        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.909       0.79      0.922      0.635

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/24         0G     0.9915     0.8982      1.404          5        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.919      0.848      0.922       0.64

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/24         0G      1.052     0.8936      1.538          3        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.904      0.834       0.92      0.633

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/24         0G     0.9382     0.8386      1.412          4        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.866      0.853      0.929      0.684

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/24         0G     0.9433     0.8433       1.41          4        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.915       0.87      0.947      0.699

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/24         0G     0.9043     0.7273      1.321          3        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.939      0.878      0.961      0.721

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/24         0G      1.017     0.7689      1.453          4        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.944      0.886      0.956      0.733

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/24         0G     0.8414      0.689      1.288          2        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.956      0.901      0.968       0.75

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/24         0G     0.8346       0.66      1.269          4        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.914       0.94      0.978      0.758

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/24         0G     0.9057     0.7738      1.364          4        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.942      0.937       0.98      0.765

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/24         0G     0.8136     0.6773      1.296          4        640: 100%|██████████| 8/8 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.962      0.949      0.985      0.781

25 epochs completed in 4.127 hours.
Optimizer stripped from runs/train/exp5/weights/last.pt, saved as runs/train/exp5/weights/last_striped.pt, 51.4MB
Optimizer stripped from runs/train/exp5/weights/best.pt, saved as runs/train/exp5/weights/best_striped.pt, 51.4MB

Validating runs/train/exp5/weights/best.pt...
Fusing layers... 
gelan-c summary: 467 layers, 25413273 parameters, 0 gradients, 102.5 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|█████
                   all        115        170      0.961      0.949      0.985      0.781
               ironman        115         71          1      0.949      0.993       0.79
                batman        115         52       0.98      0.961      0.992      0.763
              superman        115         47      0.904      0.936       0.97       0.79
Results saved to runs/train/exp5


## Test Output
![Group Norm](result.png)




**Yeswanth**
